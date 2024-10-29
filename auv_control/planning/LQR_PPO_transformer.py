import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
from collections import deque
from auv_control import State
from .base import BasePlanner
from scipy.linalg import solve_continuous_are
from auv_control.control import LQR
from auv_control.estimation import InEKF
from auv_control import scenario
from auv_control.planning.astar import Astar
import logging
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerPPOPlanner(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=20, start=None, end=None, state_dim=21, action_dim=8, lr=1e-4, gamma=0.99, clip_epsilon=0.1, entropy_coef=0.01, value_loss_coef=0.1, ppo_epochs=10, batch_size=512):
        self.num_seconds = num_seconds
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.start = np.array([0, 0, 0]) if start is None else start
        self.end = np.array([40, 40, -20]) if end is None else end
        self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)

        # Setup obstacles
        self.num_obstacles = num_obstacles
        self.size = np.array([50, 50, 25])
        self.bottom_corner = np.array([-5, -5, -25])
        self.setup_obstacles()

        # Transformer Neural Networks
        self.policy_net = self._build_transformer_network().to(device)
        self.value_net = self._build_value_network().to(device)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

        # Memory for storing experience
        self.memory = []

        # Base planner setup
        super().__init__()
        self.pos_func = self.pos_func
        self.rot_func = self.rot_func

        # LQR setup
        self.m = 31.02  # AUV mass
        self.J = np.eye(3) * 2  # Example inertia values
        self.Q_lqr = np.diag([100, 100, 100, 1, 1, 1])  # State cost
        self.R_lqr = np.diag([0.01, 0.01, 0.01])  # Control cost

    def setup_obstacles(self):
        """Sets up obstacles with random positions and sizes."""
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (self.num_obstacles, 3)) * self.size + self.bottom_corner
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner

    def _build_transformer_network(self):
        """Construct the transformer-based policy network."""
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.state_dim, nhead=4, dim_feedforward=256)
        transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        return nn.Sequential(
            transformer,
            nn.Linear(self.state_dim, self.action_dim * 2)  # output: mean and std for the action distribution
        )

    def _build_value_network(self):
        """Construct the value network to estimate state values."""
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.state_dim, nhead=4, dim_feedforward=256)
        transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        return nn.Sequential(
            transformer,
            nn.Linear(self.state_dim, 1)  # output: value estimate
        )

    def select_action(self, state, lqr_action, weight):
        """Selects actions using the policy network and combines with LQR action."""
        state_tensor = torch.FloatTensor(self.pad_state(state)).unsqueeze(0).to(device)
        action_params = self.policy_net(state_tensor)
        mean = action_params[:, :self.action_dim]
        std = torch.clamp(action_params[:, self.action_dim:], min=1e-3)

        dist = D.Normal(mean, std)
        action = dist.sample()

        lqr_weight = max(0, 0.4 - weight)
        policy_weight = min(1, 0.6 + weight)

        combined_action = lqr_weight * torch.tensor(lqr_action, device=device, dtype=torch.float32) + policy_weight * action
        log_prob = dist.log_prob(action).sum(-1)

        return combined_action.cpu().numpy(), log_prob

    def pad_state(self, state, target_dim=21):
        """Pads the state to match the target dimension for input to the networks."""
        state_flat = np.ravel(state)
        if state_flat.shape[0] < target_dim:
            padded_state = np.pad(state_flat, (0, target_dim - state_flat.shape[0]), 'constant', constant_values=0)
        else:
            padded_state = state_flat[:target_dim]
        return padded_state

    def calculate_reward(self, state, next_state, action):
        """Calculates reward based on the progress towards the goal and other factors."""
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        progress_reward = 0.1 * (self.prev_distance_to_goal - distance_to_goal)
        distance_reward = np.log(1 / (distance_to_goal + 1))  # 使用对数平滑距离奖励
        time_penalty = -0.01 * np.linalg.norm(next_state.vec[0:3] - self.end)    
        distance_to_nearest_obstacle = np.min([np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc])
        obstacle_penalty = -np.exp(-distance_to_nearest_obstacle)
        safety_reward = 0.5 if distance_to_nearest_obstacle > 2 else 0

        velocity_magnitude = np.linalg.norm(next_state.vec[3:6])
        action_smoothness_penalty = -0.05 * np.linalg.norm(action - self.previous_action) / (velocity_magnitude + 1)

        energy_penalty = 0

        goal_direction = (self.end - state.vec[0:3])
        goal_direction /= np.linalg.norm(goal_direction)
        velocity_direction = next_state.vec[3:6] / (np.linalg.norm(next_state.vec[3:6]) + 1e-5)
        alignment_reward = np.clip(1.0 * np.dot(goal_direction, velocity_direction), 0, 1)

        reward = (
            progress_reward + distance_reward +
            time_penalty + obstacle_penalty + safety_reward +
            action_smoothness_penalty + energy_penalty +
            alignment_reward
        )

        if distance_to_goal < 1:
            reward += 100

        self.prev_distance_to_goal = distance_to_goal

        return reward

    # 其他方法（如train，update_policy等）可以保持不变

    def remember(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))

    def compute_gae(self, next_value, rewards, values, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return torch.FloatTensor(advantages).to(device) 

    def train(self, env, num_episodes=100, max_steps=3000,ctrain = False, model_path="ppo_best_model.pth"):
        wandb.init(project="auv_control_project", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": self.lr,
        })
        
        controller = LQR()
        observer = InEKF()
        planner = Astar(max_steps)
        ts = 1 / scenario["ticks_per_sec"]
        
        for episode in range(num_episodes):
            logging.info(f"Episode {episode + 1} starting")
            self.setup_obstacles()
            state_info = env.reset()
            state = State(state_info)
            done = False
            step_count = 0
            total_reward = 0
            distance_to_goal = np.linalg.norm(self.start - self.end)
            distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])

            while not done and step_count < max_steps:
                sensors = env.tick()
                t = sensors["t"]
                true_state = State(sensors)
                est_state = observer.tick(sensors, ts)
                des_state = planner.tick(t)
                lqr_action = controller.u(est_state, des_state)
                real_state = np.append(true_state.vec[0:], true_state.bias[0:])
                real_state = np.append(real_state, distance_to_goal)
                real_state = np.append(real_state, distance_to_nearest_obstacle)
                real_state = np.append(real_state, done)
                weight = 0.4 * step_count/max_steps
                if ctrain:
                    weight = 0.4
                combined_action, log_prob = self.select_action(real_state, lqr_action, weight)

                next_state_info = env.step(combined_action, ticks=1, publish=False)
                next_state = State(next_state_info)
                distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
                distance_to_nearest_obstacle = np.min([np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc])
                done = np.linalg.norm(next_state.vec[0:3] - self.end) < 1
                real_next_state = np.append(next_state.vec[0:], next_state.bias[0:])
                real_next_state = np.append(real_next_state, distance_to_goal)
                real_next_state = np.append(real_next_state, distance_to_nearest_obstacle)
                real_next_state = np.append(real_next_state, done)
                real_next_state = State(real_next_state)
                
                #done = distance_to_goal < 2
                reward = self.calculate_reward(true_state, next_state, combined_action)
                total_reward += reward

                value = self.value_net(torch.FloatTensor(self.pad_state(real_state)).unsqueeze(0).to(device)).item()
                
                self.remember(real_state, combined_action, log_prob.item(), reward, value, done)

                #state = next_state
                step_count += 1

                if len(self.memory) >= self.batch_size:
                    self.update_policy()
                    self.memory = []

            wandb.log({
                "episode": episode + 1,
                "total_reward": total_reward,
                "distance to goal": distance_to_goal

            })

            logging.info(f"Episode {episode + 1} completed - Total Reward: {total_reward}")
            self.save_model(episode + 1, model_path)

    def update_policy(self):
        states, actions, log_probs, rewards, values, dones = zip(*self.memory)

        states = torch.FloatTensor([self.pad_state(s) for s in states]).to(device)
        actions = torch.FloatTensor(actions).to(device)  # Changed to FloatTensor for continuous actions
        log_probs = torch.FloatTensor(log_probs).to(device)
        values = torch.FloatTensor(values).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        next_value = self.value_net(states[-1].unsqueeze(0)).item()
        advantages = self.compute_gae(next_value, rewards, values, dones)
        returns = advantages + values

        for _ in range(self.ppo_epochs):
            action_params = self.policy_net(states)
            means, stds = torch.chunk(action_params, 2, dim=-1)
            stds = torch.clamp(stds, min=1e-3)  # Ensure standard deviation is positive

            dist = D.Normal(means, stds)
            new_log_probs = dist.log_prob(actions).sum(-1)

            ratio = (new_log_probs - log_probs).exp()
            advantages_tensor = advantages
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            # Ensure returns_tensor is created on the correct device
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            value_preds = self.value_net(states).squeeze()
            critic_loss = nn.MSELoss()(value_preds, returns_tensor)

            entropy = dist.entropy().mean()
            total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

            wandb.log({
                "total_loss": total_loss / _ if _ > 0 else 0,
            })

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            
    def save_model(self, episode, path='ppo_best_model.pth'):
        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logging.info(f"Model saved at episode {episode}" + str(path))
        
    def load_model(self, path='ppo_best_model.pth'):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Model loaded from {path}")




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
                       
    def extract_element(self,state):
        distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
        distance_to_goal = np.linalg.norm(state.vec[0:3] - self.end)
        return distance_to_goal,distance_to_nearest_obstacle


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