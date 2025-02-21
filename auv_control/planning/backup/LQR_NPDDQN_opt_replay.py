import numpy as np
from auv_control import State
from AUVControl.auv_control.planning.base import BasePlanner
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from scipy.linalg import solve_continuous_are
from auv_control.control import LQR
from auv_control.estimation import InEKF
from auv_control import scenario
from auv_control.planning.astar import Astar
import sys
# Set up logging to output to the console and file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LQRNPDDQNPlanner(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=20, start=None, end=None, state_dim=24, action_dim=8, max_memory=1024, batch_size=512, gamma=0.99, lr=1e-3,alpha=0.6, beta_start=0.4, beta_frames=1000):
        # Parameters
        self.num_seconds = num_seconds
        self.state_dim = state_dim  # Update state_dim to include richer state information
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.epsilon = 1.0  # Exploration-exploitation balance
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995  # Adjusted epsilon decay rate
        self.current_state = None
        self.lr = lr
        # Priority replay buffer parameters
        self.alpha = alpha  # How much prioritization is used (0 = no prioritization, 1 = full prioritization)
        self.beta_start = beta_start  # Initial value of beta for importance sampling
        self.beta_frames = beta_frames  # Number of frames to anneal beta over
        self.beta = beta_start  # Start beta at beta_start
        self.frame_idx = 0  # To track the current frame for beta annealing

        # Memory for prioritized replay
        self.memory = []
        self.priorities = np.zeros((max_memory,), dtype=np.float32)  # Stores priorities for the experiences

        
        # Setup goal
        self.start = np.array([0, 0, 0]) if start is None else start
        self.end = np.array([40, 40, -20]) if end is None else end

        # Setup environment
        self.size = np.array([50, 50, 25])
        self.bottom_corner = np.array([-5, -5, -25])

        # Setup obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (num_obstacles, 3)) * self.size + self.bottom_corner
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner

        # Neural Networks
        self.policy_net = self._build_network().to(device)
        self.target_net = self._build_network().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Update target network initially
        self._update_target_network()

        # Setup base planner properties
        super().__init__()
        self.pos_func = self.pos_func
        self.rot_func = self.rot_func

        # Store previous action for smoothness reward
        self.previous_action = None
        

        # LQR parameters
        self.m = 31.02  # Mass of the AUV
        self.J = np.eye(3) * 2  # Moment of inertia (example values)
        self.Q_lqr = np.diag([100, 100, 100, 1, 1, 1])  # State cost matrix
        self.R_lqr = np.diag([0.01, 0.01, 0.01])  # Control cost matrix
        
    def extract_element(self,state):
        distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
        distance_to_goal = np.linalg.norm(state.vec[0:3] - self.end)
        return distance_to_goal,distance_to_nearest_obstacle
    
    def _build_network(self):
        """Builds a simple feedforward neural network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def _update_target_network(self):
        """Copies the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state_v, state_b, state_m, action, reward, real_next_sv, real_next_sb, real_next_sm, done):
        """Store the experience in memory with a priority."""
        max_prio = self.priorities.max() if self.memory else 1.0  # Assign max priority to new samples
        experience = (state_v, state_b, state_m, action, reward, real_next_sv, real_next_sb, real_next_sm, done)

        if len(self.memory) < self.max_memory:
            self.memory.append(experience)
        else:
            self.memory[self.frame_idx % self.max_memory] = experience  # Replace old experiences

        self.priorities[self.frame_idx % self.max_memory] = max_prio  # Assign max priority to new experience
        self.frame_idx += 1

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return None

        # Anneal beta over time
        self.beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        # Compute probabilities for sampling experiences
        prios = self.priorities[:len(self.memory)] ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)  # Sample experiences based on probabilities
        batch = [self.memory[idx] for idx in indices]

        # Separate the experience elements
        states_v, states_b, states_m, actions, rewards, next_sv, next_sb, next_sm, dones = zip(*batch)

        # Flatten and pad states for consistency
        states = [np.concatenate([v.flatten(), b.flatten(), m.flatten()]) for v, b, m in zip(states_v, states_b, states_m)]
        next_states = [np.concatenate([sv.flatten(), sb.flatten(), sm.flatten()]) for sv, sb, sm in zip(next_sv, next_sb, next_sm)]

        states = torch.FloatTensor(np.array([self.pad_state(s) for s in states])).to(device)
        next_states = torch.FloatTensor(np.array([self.pad_state(ns) for ns in next_states])).to(device)

        actions = torch.LongTensor([np.argmax(np.ravel(a)) for a in actions]).to(device)
        rewards = torch.FloatTensor(rewards).squeeze().to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get current Q-values from policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the next Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute TD errors and update priorities
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i] + 1e-5  # Update priority with small epsilon

        # Importance-sampling weights for gradient correction
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = torch.FloatTensor(weights).to(device)

        # Compute the loss
        loss = (weights * nn.SmoothL1Loss(reduction='none')(q_values, target_q_values)).mean()

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def pad_state(self, state, target_dim=24):
        """Pad the state to match the required dimension."""
        state_flat = np.ravel(state)
        if state_flat.shape[0] < target_dim:
            padded_state = np.pad(state_flat, (0, target_dim - state_flat.shape[0]), 'constant', constant_values=0)
        else:
            padded_state = state_flat[:target_dim]
        return padded_state


    def save_model(self, episode, path='npddqn_best_reward_model.pth'):
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logging.info(f"Model saved at episode {episode}" + str(path))

    def load_model(self, path='npddqn_best_reward_model.pth'):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Model loaded from {path}")    
    
    def calculate_reward(self, state, next_state, action):
        # Goal reward and progress reward
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        progress_reward = 0.1 * (self.prev_distance_to_goal - distance_to_goal)
        distance_reward = 1 / (distance_to_goal + 1)

        # Time penalty for every step
        time_penalty = -0.01

        # Obstacle avoidance penalty and safety reward
        distance_to_nearest_obstacle = np.min([np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc])
        obstacle_penalty = -5 / (distance_to_nearest_obstacle + 0.1)
        safety_reward = 0.5 if distance_to_nearest_obstacle > 5 else 0

        # Action smoothness penalty
        velocity_magnitude = np.linalg.norm(next_state.vec[3:6])
        action_smoothness_penalty = -0.05 * np.linalg.norm(action - self.previous_action) / (velocity_magnitude + 1)

        # Energy efficiency penalty
        energy_penalty = 0 # -0.01 * np.sum(np.abs(action))

        # Goal alignment reward
        goal_direction = (self.end - state.vec[0:3])
        goal_direction /= np.linalg.norm(goal_direction)
        velocity_direction = next_state.vec[3:6] / (np.linalg.norm(next_state.vec[3:6]) + 1e-5)
        alignment_reward = 1.0 * np.dot(goal_direction, velocity_direction)

        # Final reward combination
        reward = (
            progress_reward + distance_reward +
            time_penalty + obstacle_penalty + safety_reward +
            action_smoothness_penalty + energy_penalty +
            alignment_reward
        )

        # If goal is reached, add a substantial reward
        if distance_to_goal < 1:
            reward += 100

        # Update previous distance for progress calculation
        self.prev_distance_to_goal = distance_to_goal

        return reward

    def _select_action(self, state, lqr_action, weight,inference = False):
        """Selects action using epsilon-greedy policy and combines with LQR action."""
        action_dim = self.action_dim
        vertical_thruster_max = 50  # Max value for vertical thrusters
        angled_thruster_max = 60    # Max value for angled thrusters
        if random.random() < self.epsilon: #and weight < 0.4:
            # Exploration: Random action
            #print("debug _select_action training using")
            #action = np.random.uniform(-35, 35, self.action_dim)  
            # Exploration: Random action within specified ranges for each thruster type
            action = np.zeros(action_dim)
            # First 4 are vertical thrusters: [-50, 50]
            action[:4] = np.random.uniform(-vertical_thruster_max, vertical_thruster_max, 4)
            # Last 4 are angled thrusters: [0, 60]
            action[4:] = np.random.uniform(0, angled_thruster_max, 4)
        else:
            #print("debug _select_action evaluation using")
            # Exploitation: Use policy network
            if inference:
                self.policy_net.eval()
            state_tensor = torch.FloatTensor(self.pad_state(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = self.policy_net(state_tensor).cpu().numpy().flatten()
        # Assuming you have a training step counter or similar to adjust the weight over time
        lqr_weight = max(0, 0.4 - weight) 
        policy_weight = min(1, 0.6 + weight)

        # Combine LQR action with policy network action
        combined_action = lqr_weight * lqr_action + policy_weight * action


        self.previous_action = combined_action
        return combined_action
    
    def train(self, env, num_episodes=100, max_steps=3000, ctrain = False, model_path = "npddqn_best_reward_model.pth"):
        # Initialize wandb
        wandb.init(project="auv_control_project", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "epsilon_decay": self.epsilon_decay,
        })
        reward_ref = -float('inf')
        og_distance_to_goal = np.linalg.norm(self.start - self.end)
        distance_to_goal = np.linalg.norm(self.start - self.end)
        controller = LQR()
        observer = InEKF()
        ts = 1 / scenario["ticks_per_sec"]
        num_ticks = int(max_steps / ts)
        planner = Astar(max_steps)
        self.prev_distance_to_goal = og_distance_to_goal
        for episode in range(num_episodes):
            logging.info(f"Episode {episode + 1} starting")
            state_info = env.reset()
            state = State(state_info)  # Use State class to initialize state
            distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
            done = False
            total_reward = 0
            reward = 0 
            episode_loss = 0
            step_count = 0
            while not done and step_count < max_steps:
                # Get LQR action
                weight = 0.8 * step_count/max_steps
                # Get LQR action
                sensors = env.tick()
                # Pluck true state from sensors
                t = sensors["t"]
                true_state = State(sensors)
                #print("debug!True state sensors",sensors)
                # Estimate State
                est_state = observer.tick(sensors, ts)
                # Path planner
                des_state = planner.tick(t)
                # Autopilot Commands
                u = controller.u(est_state, des_state)
                lqr_action = u
                for_act_state = np.append(state.vec[0:], state.bias[0:])
                for_act_state = np.append(for_act_state, state.mat[0:])
                for_act_state = np.append(for_act_state, distance_to_goal)
                for_act_state = np.append(for_act_state, distance_to_nearest_obstacle)
                for_act_state = np.append(for_act_state, done)
                if ctrain:
                    weight = 0.4
                action = self._select_action(for_act_state, lqr_action, weight)
                #print("debugging action!!",action)
               
                next_state_info = env.step(action, ticks=1,publish=False)
                next_state = State(next_state_info)  # Use State class for next state
                #print("debugging position!!",next_state_info["PoseSensor"][:3,3]) #position info

                # Goal distance reward
                distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)

                # Add obstacle avoidance penalty
                distance_to_nearest_obstacle = np.min([np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc])
                real_next_state = np.append(next_state.vec[0:], next_state.bias[0:])
                real_next_state = np.append(real_next_state, next_state.mat[0:])
                real_next_state = np.append(real_next_state, distance_to_goal)
                real_next_state = np.append(real_next_state, distance_to_nearest_obstacle)
                real_next_state = State(real_next_state)
                done = distance_to_goal < 2
                # Store transition in memory
                self.remember(state.vec[0:],state.bias[0:],state.mat[0:], action, reward, real_next_state.vec[0:],real_next_state.bias[0:],real_next_state.mat[0:], done)
                state = real_next_state
                total_reward += self.calculate_reward(state, next_state, action) #reward
                self.prev_distance_to_goal = distance_to_goal
                # Perform replay and get loss value
                loss = self._replay()
                episode_loss += loss if loss is not None else 0
                step_count += 1

            # Log metrics to wandb
            wandb.log({
                "episode": episode + 1,
                "total_reward": total_reward,
                "average_loss": episode_loss / step_count if step_count > 0 else 0,
                "epsilon": self.epsilon
            })

            # Adjust epsilon

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % 4 == 0:
                self._update_target_network()
            #self._update_target_network()

            logging.info(f"Episode {episode + 1} completed - Total Reward: {total_reward}, Loss: {episode_loss / step_count if step_count > 0 else 0}")

            # Save the model with the best reward
            if distance_to_goal < og_distance_to_goal:
                self.save_model(episode + 1, path=model_path)
                og_distance_to_goal = distance_to_goal
                wandb.log({"best_distance_to_goal": distance_to_goal})
                print("Best distance to goal achieved", distance_to_goal)

    def pos_func(self, state,t):
        """
        Position function to calculate desired position at time t using the policy model.
        """
        with torch.no_grad():
            # Prepare the state input, assuming you have access to the state in this context (e.g., estimated state)
            state = self.pad_state(state)  # Replace current_state with your actual state variable
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get position values from the policy network
            predicted_pos = self.policy_net(state_tensor)[:, :3]  # First three values correspond to position
            return predicted_pos.cpu().numpy().flatten()

    def rot_func(self, state, t):
        """
        Rotation function to calculate desired rotation at time t using the policy model.
        """
        with torch.no_grad():
            # Prepare the state input, assuming you have access to the state in this context (e.g., estimated state)
            state = self.pad_state(state)  # Replace current_state with your actual state variable
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get rotation values from the policy network
            predicted_rot = self.policy_net(state_tensor)[:, 3:6]  # Values for rotation start from index 3
            return predicted_rot.cpu().numpy().flatten()



    @property
    def center(self):
        return self.bottom_corner + self.size / 2

    @property
    def top_corner(self):
        return self.bottom_corner + self.size

    def draw_traj(self, env, t):
        """Override super class to also make environment appear"""
        observer = InEKF()
        # Setup environment
        env.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0)
        for i in range(self.num_obstacles):
            loc = self.obstacle_loc[i].tolist()
            loc[1] *= -1
            env.spawn_prop('sphere', loc, [0, 0, 0], self.obstacle_size[i], False, "white")
        sensors = env.tick()
        # Pluck true state from sensors
        ts = 1 / 100
        t = sensors["t"]
        true_state = State(sensors)
        distance_to_nearest_obstacle = np.min([np.linalg.norm(true_state.vec[0:3] - obs) for obs in self.obstacle_loc])
        distance_to_goal = np.linalg.norm(true_state.vec[0:3] - self.end)    
        if distance_to_goal < 1:
            done = True
        else:
            done = False
        for_act_state = np.append(true_state.vec[0:], true_state.bias[0:])
        for_act_state = np.append(for_act_state, true_state.mat[0:])
        for_act_state = np.append(for_act_state, distance_to_goal)
        for_act_state = np.append(for_act_state, distance_to_nearest_obstacle)
        for_act_state = np.append(for_act_state, done)
        # Estimate State
        des_state = self._traj(for_act_state,t)

        # If des_state is 1D, make it 2D for consistency in indexing
        if des_state.ndim == 1:
            des_state = des_state.reshape(1, -1)

        des_pos = des_state[:, 0:3]

        # Draw line between each point
        for i in range(len(des_pos) - 1):
            env.draw_line(des_pos[i].tolist(), des_pos[i + 1].tolist(), thickness=5.0, lifetime=0.0)

    def _traj(self, state,t):
        """Get desired trajectory at time t"""
        eps = 1e-5

        pos = self.pos_func(state,t)
        rot = self.rot_func(state,t)

        lin_vel = (self.pos_func(state,t+eps) - pos) / eps
        ang_vel = (self.rot_func(state,t+eps) - rot) / eps

        return np.hstack((pos, lin_vel, rot, ang_vel))

    def tick(self, state,t):
        """Gets desired trajectory at time t, only as a state"""
        if not isinstance(t, float):
            raise ValueError("Can't tick with an array")

        return State(self._traj(state,t))