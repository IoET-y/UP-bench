import pandas as pd
import numpy as np
from auv_control import State
from .base import BasePlanner
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from collections import deque
import random
import torch.distributions as D
import pickle
from auv_control.control import LQR
#from auv_control.estimation import InEKF
from auv_control import scenario
from auv_control.planning.astar import Astar

# Set up logging to output to the console and file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, transition, td_error):
        """Add a new experience and corresponding TD-error priority."""
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4,alpha = 0.6):
        """Sample a batch of experiences with prioritized weights."""
        if len(self.buffer) == 0:
            return [], []

        scaled_priorities = np.array(self.priorities) ** alpha
        probabilities = scaled_priorities / np.sum(scaled_priorities)

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        """Update the priorities of sampled experiences based on TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Avoid zero priority

    def clear(self):
        """Clear the replay buffer and priorities."""
        self.buffer = []
        self.priorities = deque(maxlen=self.capacity)
        self.position = 0

    def __len__(self):
        """Return the current number of elements in the buffer."""
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.max_action_normalize = max_action

        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        # )
        # Wider networks with layer normalization
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent numerical issues
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
    
        # Ensure max_action is a tensor and on the same device as y_t
        max_action_tensor = torch.tensor(self.max_action_normalize, device=y_t.device, dtype=y_t.dtype)
    
        action = y_t * max_action_tensor
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q_value = self.net(x)
        return q_value

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4  # 防止除以零

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / new_count
        new_var = m2 / new_count

        self.mean, self.var, self.count = new_mean, new_var, new_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class LQRSACPlanner(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=20, start=None, end=None, state_dim=9, action_dim=3, lr=2e-3, gamma=0.97, tau=0.005, batch_size=128, replay_buffer_size=1000000):
        #temperature and index 20241211
        self.target_entropy = -np.prod(action_dim)* 1  # 默认目标策略熵（动作维度负对数）
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)  # 温度系数 log(alpha)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)  # 优化器

        # Alpha 的初始化值emperature parameter for SAC
        self.alpha = self.log_alpha.exp().item()


        # Memory for storing experience
        self.replay_buffer_size = replay_buffer_size
        self.per_alpha = 0.6  # Prioritized experience replay alpha parameter
        self.memory = PrioritizedReplayBuffer(replay_buffer_size,alpha=self.per_alpha)
        self.beta = 0.4  # Importance sampling beta parameter
        self.beta_increment_per_sampling = 0.001

        self.reach_targe_times = 0
        # Rewards for episode
        self.episode_distance_reward = 0
        self.episode_align_reward = 0
        self.episode_current_utilization_reward = 0
        self.episode_safety_reward = 0
        self.episode_reach_target_reward = 0
        self.episode_out_box_penalty = 0

        # Parameters
        self.num_seconds = num_seconds
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size

        # Setup goal
        self.start = np.array([0, 0, -25]) if start is None else start
        self.end = np.array([50, 50, -40]) if end is None else end
        self.OG_distance_to_goal = np.linalg.norm(self.start - self.end)
        self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)

        # Setup environment
        self.size = np.array([120, 120, 50])
        self.bottom_corner = np.array([-70, -70, -70])

        # Setup obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_loc = []
        self.obstacle_size = np.random.uniform(1, 1, 20)

        # Max action values
        self.max_lin_accel = 20.0
        self.max_ang_accel = 2.0
        self.max_action = np.array([self.max_lin_accel]*3) #+ [self.max_ang_accel]*3)
        self.max_action_normalize = np.array([1]*3)

        # Temperature parameter for SAC
        
        # Neural Networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.max_action_normalize).to(device)
        self.q_net1 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q_net2 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_q_net1 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_q_net2 = QNetwork(self.state_dim, self.action_dim).to(device)

        # Apply weight initialization to networks
        self.policy_net.apply(self.initialize_weights)
        self.q_net1.apply(self.initialize_weights)
        self.q_net2.apply(self.initialize_weights)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)

        # Setup base planner properties
        super().__init__()
        self.pos_func = self.pos_func
        self.rot_func = self.rot_func

        # Store previous action for smoothness reward
        self.previous_action = np.zeros(action_dim)
        self.OG_distance_to_goal = np.linalg.norm(self.start - self.end)
        self.epsd = 0
        self.done = False
        self.first_reach_close = 0
        
        self.inx_current = 0.08
        self.inx_distance = 0.2

        # For static last state penalty use
        self.static_cnt = 0
        self.last_distance_to_goal = 0
        self.static_on = False
        self.closest_pos = 0

        # Time-related parameters
        self.ticks_per_sec = 100
        self.current_time = 0.0

        # Ocean current parameters
        self.current_strength = 0.5
        self.current_frequency = np.pi
        self.current_mu = 0.25
        self.current_omega = 2.0

        # Agent physical parameters
        self.gravity = 9.81
        self.cob = np.array([0, 0, 5.0]) / 100  # Center of buoyancy
        self.m = 31.02
        self.rho = 997
        self.V = self.m / self.rho
        self.J = np.eye(3) * 2  # Inertia tensor

                
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'out_features') and m.out_features == self.action_dim:
                # Policy network最后一层使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
            else:
                # 其他层（包括q_network）使用Kaiming初始化
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def calculate_ocean_current(self, position, time):
        """Calculate the ocean current's three-dimensional velocity."""
        x, y, z = position

        A = self.current_strength
        mu = self.current_mu
        omega = self.current_omega

        a_t = mu * np.sin(omega * time)
        b_t = 1 - 2 * mu * np.sin(omega * time)
        f_x_t = a_t * x**2 + b_t * x

        psi = A * np.sin(self.current_frequency * f_x_t) * np.sin(self.current_frequency * y)

        # Velocity components
        v_x = -self.current_frequency * A * np.sin(self.current_frequency * f_x_t) * np.cos(self.current_frequency * y)
        v_y = self.current_frequency * A * np.cos(self.current_frequency * f_x_t) * np.sin(self.current_frequency * y)
        v_z = 0  # Assuming vertical ocean current is negligible

        return np.array([v_x, v_y, v_z])

    def calculate_action_effect(self, action, ocean_current):
        """Calculate the effect of ocean current on the action."""
        area = 0.3 * 0.3  # Cross-sectional area of the AUV (assuming square)
        drag_coefficient = 1.0  # Drag coefficient

        # Relative velocity and drag force calculation
        relative_velocity = -ocean_current  # Assuming AUV is relatively stationary
        drag_force = 0.5 * drag_coefficient * self.rho * area * np.linalg.norm(relative_velocity) * relative_velocity

        # Acceleration due to drag
        acceleration_due_to_drag = drag_force / self.m
        # Update linear acceleration in action
        adjusted_action = np.copy(action)
        adjusted_action[:3] += acceleration_due_to_drag
        return adjusted_action
        
    def is_valid_position(self, pos, obstacle_locs, min_distance=5.0):
        """
        检查 pos 是否与 obstacle_locs 中所有障碍物都至少 min_distance 米。
        若都满足，则返回 True，否则返回 False。
        """
        for obs in obstacle_locs:
            if np.linalg.norm(pos - obs) < min_distance:
                return False
        return True
    
    def setup_start(self):
        lower_bound = np.array([-40, -40, -70])
        upper_bound = np.array([70, 70, -20])
        valid_position_found = False
    
        while not valid_position_found:
            candidate = lower_bound + np.random.rand(3) * (upper_bound - lower_bound)
            
            # 如果这个 candidate 与所有障碍物距离都 >= 5
            if self.is_valid_position(candidate, self.obstacle_loc, min_distance=5.0):
                self.start = candidate
                valid_position_found = True
        #self.start = random_position
        print("Start point:", self.start)

    def setup_end(self):
        # 确保终点满足与起点的距离条件
        distance_min = 40  # min距离
        distance_max = 90  # max距离
        valid_position_found = False
        lower_bound = np.array([-30, -30, -70])
        upper_bound = np.array([60, 60, -20])

        while not valid_position_found:
            candidate = lower_bound + np.random.rand(3) * (upper_bound - lower_bound)
            distance = np.linalg.norm(candidate - self.start)
            
            # 与障碍物 >= 5m
            # 和 start 距离在 [40, 100] 之间
            if self.is_valid_position(candidate, self.obstacle_loc, min_distance=5.0) \
               and (distance_min <= distance <= distance_max):
                self.end = candidate
                valid_position_found = True
    
        print("End point:", self.end)
        print(f"Distance between start and end: {np.linalg.norm(self.end - self.start):.2f}")    

            
    def get_start_end_obs(self,ind):
        self.obstacle_loc = self.predefined_obstacle_distributions[ind]
        self.obstacle_size = np.random.uniform(1, 1, 16)
        #print(self.obstacle_loc)
        return self.start, self.end, self.obstacle_loc, self.obstacle_size   
        
    def load_obstacle_distributions(self, filename="predefined_obstacles.pkl"):

        with open(filename, 'rb') as f:
            self.predefined_obstacle_distributions = pickle.load(f)
        print(f"Predefined obstacle distributions loaded from {filename}.")

    def save_obstacle_distributions(self, filename="predefined_obstacles.pkl"):

        with open(filename, 'wb') as f:
            pickle.dump(self.predefined_obstacle_distributions, f)
        print(f"Predefined obstacle distributions saved to {filename}.")
        
    def setup_obstacles(self, num_obstacles=20, grid_size=(4, 5, 2), train=True):
        #self.memory.clear()
        self.num_obstacles = num_obstacles
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)

        # Define grid
        x_bins = np.linspace(self.bottom_corner[0], self.bottom_corner[0] + self.size[0], grid_size[0] + 1)
        y_bins = np.linspace(self.bottom_corner[1], self.bottom_corner[1] + self.size[1], grid_size[1] + 1)
        z_bins = np.linspace(self.bottom_corner[2], self.bottom_corner[2] + self.size[2], grid_size[2] + 1)

        grid_cells = [(x_bins[i], x_bins[i+1], y_bins[j], y_bins[j+1], z_bins[k], z_bins[k+1])
                      for i in range(grid_size[0]) for j in range(grid_size[1]) for k in range(grid_size[2])]

        # Adjust grid cell selection for train/test scenarios
        if train:
            selected_cells = random.sample(grid_cells, k=len(grid_cells) // 2)  # Randomly select half of the grid cells
        else:
            selected_cells = grid_cells  # Use all cells for testing

        # Generate obstacles
        self.obstacle_loc = []
        for cell in selected_cells:
            x_min, x_max, y_min, y_max, z_min, z_max = cell
            num_obstacles_in_cell = num_obstacles // len(selected_cells)
            obstacles_in_cell = np.random.uniform(
                [x_min, y_min, z_min], [x_max, y_max, z_max], (num_obstacles_in_cell, 3)
            )
            self.obstacle_loc.extend(obstacles_in_cell)

        self.obstacle_loc = np.array(self.obstacle_loc)

        # Ensure obstacles are not too close to the start or end
        for i in range(len(self.obstacle_loc)):
            max_attempts = 10
            attempts = 0
            while (np.linalg.norm(self.obstacle_loc[i] - self.start) < 5 or
                   np.linalg.norm(self.obstacle_loc[i] - self.end) < 5):
                if attempts >= max_attempts:
                    break
                cell = random.choice(selected_cells)
                x_min, x_max, y_min, y_max, z_min, z_max = cell
                self.obstacle_loc[i] = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
                attempts += 1

        return self.obstacle_loc

    def spawn_obstacles(self,env, obstacle_locations):
        """
        Spawns obstacles in the simulation environment at specified locations.

        Args:
            env: The HoloOcean simulation environment.
            obstacle_locations (list of list of float): A list of [x, y, z] coordinates for each obstacle.
        """

        # Possible types of props
        prop_types = ["box", "sphere", "cylinder", "cone"]

        # Possible materials for props
        materials = ["white", "gold", "cobblestone", "brick", "wood", "grass", "steel", "black"]

        for i, location in enumerate(obstacle_locations):
            # Randomly select a prop type
            prop_type = "sphere"#random.choice(prop_types)

            # Randomly select a material
            material = "steel" #random.choice(materials)

            # Generate a random scale for the obstacle
            scale = random.uniform(2.0, 2.0)  # Scale between 0.5 and 2.0 meters

            # Set whether the object is affected by gravity
            sim_physics = True  # Objects are mobile and affected by gravity

            # Assign a tag to the prop (optional, for debugging or task management)
            tag = f"obstacle_{i}"

            # Spawn the prop
            env.spawn_prop(
                prop_type=prop_type,
                location=location.tolist(),  # Convert numpy array to list if needed
                scale=scale,
                sim_physics=sim_physics,
                material=material,
                tag=tag
            )
    def normalize_action(self, action):
        """Normalize action to [-1, 1] range."""
        return action / self.max_action

    def denormalize_action(self, normalized_action):
        """Convert normalized action back to original range."""
        return normalized_action * self.max_action
        
    def select_action(self, state, inference=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            normalized_action, _ = self.policy_net.sample(state_tensor)
        normalized_action = normalized_action.cpu().numpy()[0]
        action = np.clip(normalized_action, -self.max_action_normalize, self.max_action_normalize)

        return action

    def remember(self, state, action, adjust_action, reward, next_state, done):
        # Add to memory
        self.memory.add((state, action, adjust_action, reward, next_state, done), td_error=0.0)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from prioritized replay buffer
        samples, weights, indices = self.memory.sample(self.batch_size, beta=self.beta,alpha = self.per_alpha)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        states, actions, adjust_actions, rewards, next_states, dones = zip(*samples)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        adjust_actions = torch.FloatTensor(adjust_actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_pis = self.policy_net.sample(next_states)
            next_q1 = self.target_q_net1(next_states, next_actions)
            next_q2 = self.target_q_net2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pis
            target_q = rewards + self.gamma * (1 - dones) * next_q

        current_q1 = self.q_net1(states, actions) #adjust_actions
        current_q2 = self.q_net2(states, actions) #adjust_actions

        # TD-error for priority update
        td_errors = (target_q - current_q1).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors.flatten())

        # Loss calculation
        q_loss1 = (weights * (current_q1 - target_q).pow(2)).mean()
        q_loss2 = (weights * (current_q2 - target_q).pow(2)).mean()

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)
        self.q_optimizer2.step()

        # Policy loss
        new_actions, log_pis = self.policy_net.sample(states)
        q1_new_actions = self.q_net1(states, new_actions)
        q2_new_actions = self.q_net2(states, new_actions)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        policy_loss = (weights * (self.alpha * log_pis - q_new_actions)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()


        # 温度系数的损失函数20241211
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
        # 更新 alpha 的值20241211
        self.alpha = self.log_alpha.exp().item()
        
        wandb.log({
            "episode": self.epsd + 1,
            "q_loss1": q_loss1.item(),
            "q_loss2": q_loss2.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(), #20241211
            "alpha_value": self.alpha, #20241211
        })

        # Update target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, env, num_episodes=500, max_steps=3000, model_path="sac_best_model.pth",obsname = "new"):

        wandb.init(project="auv_RL_control_SAC_acceleration_1015", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "tau": self.tau,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "MAX_LIN_ACCEL": self.max_lin_accel,
            "maxstep": max_steps,
            "current align rd discount": self.inx_current,
            "distance rd discount": self.inx_distance,
        })

        ts = 1 / self.ticks_per_sec
        tmp = max_steps
        self.step_cnt = 0


        state_normalizer = RunningMeanStd(shape=self.state_dim)

        self.predefined_obstacle_distributions = [self.setup_obstacles(num_obstacles=20, train=True) for _ in range(100)]
        obstacle_index = 0  # Use obstacles sequentially
        self.obstacle_loc = self.predefined_obstacle_distributions[obstacle_index]
        self.spawn_obstacles(env, self.obstacle_loc)
        env.draw_box(center=[0, 0, -25], extent=[200, 200, 50], thickness=50, lifetime=0)
        env.draw_point(self.end, color=[0, 255, 0], thickness=100, lifetime=0)

        #print("onstacle dist: ", self.obstacle_loc)
        
        obs_file_path = obsname+"SAC_"+str(max_steps)+"_train_obstacles.pkl"
        self.save_obstacle_distributions(obs_file_path)
        initial_states = []
        for episode in range(num_episodes):
            self.episode_distance_reward = 0
            self.episode_align_reward = 0
            self.episode_current_utilization_reward = 0
            self.episode_safety_reward = 0
            self.episode_out_box_penalty = 0
            self.episode_reach_target_reward = 0
            self.static_on = False
            self.static_cnt = 0
            self.first_reach_close = 0
            self.epsd = episode
            
            logging.info(f"Episode {episode + 1} starting")

            if self.done:  # Introduce dynamic start and end points
                self.static_cnt = 0
                #self.memory.clear()
                obstacle_index = obstacle_index + 1
                self.obstacle_loc = self.predefined_obstacle_distributions[obstacle_index]

                self.setup_start()
                self.setup_end()
                
            state_info = env.reset()
            env.draw_box(center=[0, 0, -45], extent=[200, 200, 50], thickness=50, lifetime=0)
            env.draw_point( self.end, color=[0, 255, 0], thickness=100, lifetime=0)
            self.spawn_obstacles(env, self.obstacle_loc)


            env.agents["auv0"].teleport(self.start, [0, 0, 0])
            sensors = env.tick()
            state_info = sensors
            state = State(state_info)
            logging.info(f"Current Start: {self.start}, AUV position: is : {state.vec[0:3]}")
            distance_to_goal = np.linalg.norm(self.start - self.end)
            self.OG_distance_to_goal = distance_to_goal
            self.prev_distance_to_goal = distance_to_goal
            nm_distance_to_goal = distance_to_goal/self.OG_distance_to_goal
            self.closest_pos = distance_to_goal
            self.done = False
            done = False
            step_count = 0
            total_reward = 0
            added_steps = 0
            if distance_to_goal > 80:
                max_steps = tmp + 256
            else:
                max_steps = tmp
            self.distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])

            recurrent_states = initial_states
            initial_states = []
            early_break = False
            while not done and step_count < max_steps:
                early_break = False
                self.step_cnt += 1

                #CURRENT STEP
                sensors = env.tick()
                t = sensors["t"]
                true_state = State(sensors)
                relative_position_to_goal = self.end - true_state.vec[0:3]
                nm_relative_position_to_goal = relative_position_to_goal/(np.linalg.norm(relative_position_to_goal)+1e-8)
                sense_dto = self.distance_to_nearest_obstacle/10 if self.distance_to_nearest_obstacle < 10 else -1
                
                # 多个障碍物情况的表征 要考虑。
                ocean_current = self.calculate_ocean_current(true_state.vec[0:3], self.current_time)
                CURRENT_distance_to_goal = np.linalg.norm(true_state.vec[0:3] - self.end)
                nm_distance_to_goal = CURRENT_distance_to_goal/self.OG_distance_to_goal
 
                current_progress = self.prev_distance_to_goal - CURRENT_distance_to_goal
                nm_current_progress = current_progress/0.1
                real_state = np.concatenate([
                    #true_state.vec[:6], #[3:] #true_state.vec[3:]
                    ocean_current, nm_relative_position_to_goal,#self.end, np.array([current_alignment])
                    [nm_distance_to_goal, sense_dto,current_progress] # done
                ])


                self.prev_distance_to_goal = CURRENT_distance_to_goal
                
                #initial_states.append(real_state)
                #state_normalizer.update(np.array(initial_states))
                #normalized_real_state = state_normalizer.normalize(real_state)
                
                self.current_time += ts
                action = self.select_action(real_state)
                real_action = self.denormalize_action(action)
                
                adjusted_action = self.calculate_action_effect(real_action, ocean_current)
                normalized_adjusted_action = self.normalize_action(adjusted_action)

                # NEXT STEP 
                #Update environment
                used_action = np.append(adjusted_action, [0, 0, 0])
                next_state_info = env.step(used_action, ticks=1, publish=False)
                next_state = State(next_state_info)
                NEXT_distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
                
                nm_distance_to_goal = NEXT_distance_to_goal/self.OG_distance_to_goal
                    
                if (NEXT_distance_to_goal < 15) and (step_count >= max_steps - 1) and (added_steps <512) and (NEXT_distance_to_goal < self.prev_distance_to_goal):
                    max_steps += 2
                    added_steps += 2 
                    logging.info(f"Increasing max steps to {max_steps} for additional exploration.")
                    
                reward, aln_rd, safe_rd, reach_tg_rd, o_b_pnty = self.calculate_reward(true_state, next_state, action) #current_use_rd
                if (NEXT_distance_to_goal > self.prev_distance_to_goal) and (self.prev_distance_to_goal < 10):
                    logging.info(" early_break enable")
                    #reward -= 0.1
                    early_break = True
                    
                next_progress = self.prev_distance_to_goal- NEXT_distance_to_goal
                nm_next_progress = next_progress/ 0.1
                
                relative_position_to_goal_next = self.end - next_state.vec[0:3]
                nm_relative_position_to_goal_next = relative_position_to_goal_next/(np.linalg.norm(relative_position_to_goal_next)+1e-8)
                sense_dto_next = self.distance_to_nearest_obstacle/10 if self.distance_to_nearest_obstacle < 10 else -1
                ocean_current_next = self.calculate_ocean_current(next_state.vec[0:3], self.current_time)
                done = NEXT_distance_to_goal < 10
                self.done = done
                
                real_next_state = np.concatenate([
                    #next_state.vec[:6], #next_state.vec[3:],
                    ocean_current_next , nm_relative_position_to_goal_next, #np.array([current_alignment_next])
                    [nm_distance_to_goal, sense_dto_next, next_progress]
                ])
                
                #normalized_real_next_state = state_normalizer.normalize(real_next_state)

                self.closest_pos = min(distance_to_goal, self.closest_pos)
                total_reward += reward
                
                self.remember(real_state, action, normalized_adjusted_action, reward, real_next_state, done)
                self.update_policy()
                step_count += 1
                
                wandb.log({
                    "x_pos":true_state.vec[0],
                    "y_pos":true_state.vec[1],
                    "z_pos":true_state.vec[2],
                    "inlp_step_count": step_count,
                    "inlp_distance_to_nearest_obstacle": self.distance_to_nearest_obstacle,
                    "inlp_distance_to_goal": NEXT_distance_to_goal,
                    "inlp_aln_reward": aln_rd,
                    #"inlp_current_utilization_reward": current_use_rd,
                    "inlp_safety_reward": safe_rd,
                    "inlp_o_b_pnty": o_b_pnty,
                    "inlp_reach_target_reward": reach_tg_rd
                })

                if done:
                    self.reach_targe_times += 1
                    wandb.log({
                        "episode": episode + 1,
                        "reach_targe_times": self.reach_targe_times
                    })

                    success_path = f"successful_{episode}_{model_path}"
                    self.save_model(episode + 1, success_path)
                    break

            wandb.log({
                "episode": episode + 1,
                "eps_total_reward": total_reward,
                "eps_distance_to_goal": NEXT_distance_to_goal,
                "eps_align_reward": self.episode_align_reward,
                #"eps_current_utilization_reward": self.episode_current_utilization_reward,
                "eps_safety_reward": self.episode_safety_reward,
                "episode_out_box_penalty": self.episode_out_box_penalty,
                "eps_reach_target_reward": self.episode_reach_target_reward
            })

            logging.info(f"Episode {episode + 1} completed - Total Reward: {total_reward}")
            self.save_model(episode + 1, model_path)

    def calculate_reward(self, state, next_state, action):
        discount_factor = 0.01
        # Movement rewards
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        progress = self.prev_distance_to_goal - distance_to_goal
        distance_reward = progress #self.inx_distance * np.arctan(10 * (progress - 0.01)) #10 max acce : 1; 0.5 maxacce : 0.02   5->10

        action_direction = action[:3] / (np.linalg.norm(action[:3]) + 1e-8)
        goal_direction = (self.end - state.vec[0:3]) / (np.linalg.norm(self.end - state.vec[0:3]) + 1e-8)
        #计算动作方向与目标方向的对齐度
        alignment = np.dot(goal_direction, action_direction)
        #discount_factor = 2 if progress<0 else 1
        alignment_reward = alignment * abs(distance_reward)*discount_factor * 2
    
        # # Current alignment rewards and energy rewards
        # ocean_current = self.calculate_ocean_current(next_state.vec[0:3], self.current_time)
        # auv_velocity = next_state.vec[3:6]
        # utilization = np.dot(auv_velocity, ocean_current) / (np.linalg.norm(ocean_current) + 1e-5)
        # current_utilization_reward = self.inx_current * np.arctan(3 * (utilization - 0.5))
                      
        # Safety rewards
        distances_to_obstacles = [np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc]
        min_distance = np.min(distances_to_obstacles)
        closest_obstacle = self.obstacle_loc[np.argmin(distances_to_obstacles)]
        velocity_vector = next_state.vec[3:6]
        direction_to_obstacle = (closest_obstacle - next_state.vec[0:3]) / (min_distance + 1e-5)
        relative_speed = np.dot(velocity_vector, direction_to_obstacle)
        safe_distance_threshold = max(1, min(5, 3 + relative_speed * 0.1))

        safety_reward = -10 * np.exp(-min_distance) if min_distance < safe_distance_threshold else 0 #20
        safety_reward = safety_reward * discount_factor

        out_box_penaltyx = -0.015 if (next_state.vec[2] > 0 or next_state.vec[2] < -70 ) else 0
        out_box_penaltyy = -0.015 if (next_state.vec[1] > 100 or next_state.vec[1] < -100 ) else 0
        out_box_penaltyz = -0.015 if (next_state.vec[1] > 100 or next_state.vec[1] < -100) else 0
        out_box_penalty = 0 #out_box_penaltyx+out_box_penaltyy+out_box_penaltyz

        if distance_to_goal < 1:
            reach_target_reward = 100 * discount_factor
        else:
            reach_target_reward = 0

        if distance_to_goal < 10 and self.first_reach_close == 0:
            first_close_reward = 150 *discount_factor / (distance_to_goal + 1)
            self.first_reach_close += 1
        else:
            first_close_reward = 0

        #self.episode_distance_reward += distance_reward
        self.episode_align_reward += alignment_reward
        #self.episode_current_utilization_reward += current_utilization_reward
        self.episode_safety_reward += safety_reward
        self.episode_out_box_penalty += out_box_penalty
        self.episode_reach_target_reward += reach_target_reward
        self.previous_action = action
        #self.prev_distance_to_goal = distance_to_goal

        total_reward = (
            alignment_reward#distance_reward
            #+ current_utilization_reward
            + safety_reward
            + out_box_penalty
            + first_close_reward
            + reach_target_reward
        )
        return total_reward, alignment_reward, safety_reward, reach_target_reward, out_box_penalty#, current_utilization_reward
    
    def save_model(self, episode, path='sac_best_model.pth'):
        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'q1_state_dict': self.q_net1.state_dict(),
            'q2_state_dict': self.q_net2.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
        }, path)
        logging.info(f"Model saved at episode {episode} to {path}")

    def load_model(self, path='sac_best_model.pth'):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.q_net1.load_state_dict(checkpoint['q1_state_dict'])
        self.q_net2.load_state_dict(checkpoint['q2_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Model loaded from {path}")

    def pad_state(self, state, target_dim=25):
        state_flat = np.ravel(state)
        if state_flat.shape[0] < target_dim:
            padded_state = np.pad(state_flat, (0, target_dim - state_flat.shape[0]), 'constant', constant_values=0)
        else:
            padded_state = state_flat[:target_dim]
        return padded_state

    def pos_func(self, state, t):
        """
        Position function to calculate desired position at time t using the policy model.
        """
        with torch.no_grad():
            # 将 state 转换为适合 policy 网络输入的格式
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
            # 获取 policy 网络预测的位置部分 (前3个值对应位置)
            #predicted_action, _ = self.policy_net.sample(state_tensor)
            predicted_pos = state_tensor[:, :3]  # 提取前三个值作为位置
            return predicted_pos.cpu().numpy().flatten()
    
    def rot_func(self, state, t):
        """
        Rotation function to calculate desired rotation at time t using the policy model.
        """
        with torch.no_grad():
            # 将 state 转换为适合 policy 网络输入的格式
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
            # 获取 policy 网络预测的旋转部分 (第3到6个值对应旋转)
            #predicted_action, _ = self.policy_net.sample(state_tensor)
            predicted_rot = state_tensor[:, 3:6]  # 提取3-6值作为旋转
            return predicted_rot.cpu().numpy().flatten()
    
    def extract_element(self, state):
        """
        提取距离目标和障碍物的相关信息
        """
        distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
        distance_to_goal = np.linalg.norm(state.vec[0:3] - self.end)
        relative_position_to_goal = self.end - state.vec[0:3]
        return distance_to_goal, distance_to_nearest_obstacle, relative_position_to_goal
    
    @property
    def center(self):
        """
        返回环境中心位置
        """
        return self.bottom_corner + self.size / 2
    
    @property
    def top_corner(self):
        """
        返回环境顶部角落位置
        """
        return self.bottom_corner + self.size
    
    def draw_traj(self, env, t):
        """
        绘制轨迹，并在环境中显示障碍物和目标点
        """
        #observer = InEKF()
        # 绘制环境的边界框
        env.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0)
    
        # 在环境中绘制障碍物
        for i in range(self.num_obstacles):
            loc = self.obstacle_loc[i].tolist()
            loc[1] *= -1  # 翻转Y轴
            env.spawn_prop('sphere', loc, [0, 0, 0], self.obstacle_size[i], False, "white")
    
        # 获取传感器信息
        sensors = env.tick()
        ts = 1 / 100
        t = sensors["t"]
        true_state = State(sensors)
    
        # 计算与障碍物和目标点的距离
        distance_to_nearest_obstacle = np.min([np.linalg.norm(true_state.vec[0:3] - obs) for obs in self.obstacle_loc])
        distance_to_goal = np.linalg.norm(true_state.vec[0:3] - self.end)
        done = distance_to_goal < 1
    
        for_act_state = np.concatenate([
            true_state.vec, true_state.bias, true_state.mat.flatten(),
            [distance_to_goal, distance_to_nearest_obstacle, done]
        ])
    
        # 获取目标状态轨迹
        des_state = self._traj(for_act_state, t)
    
        # 确保 des_state 是二维
        if des_state.ndim == 1:
            des_state = des_state.reshape(1, -1)
    
        des_pos = des_state[:, 0:3]
    
        # 绘制轨迹的线条
        for i in range(len(des_pos) - 1):
            env.draw_line(des_pos[i].tolist(), des_pos[i + 1].tolist(), thickness=5.0, lifetime=0.0)
    
    def _traj(self, state, t):
        """
        获取时间 t 下的期望轨迹，包括位置、速度和旋转
        """
        eps = 1e-5
    
        pos = self.pos_func(state, t)
        rot = self.rot_func(state, t)
    
        lin_vel = (self.pos_func(state, t + eps) - pos) / eps
        ang_vel = (self.rot_func(state, t + eps) - rot) / eps
    
        return np.hstack((pos, lin_vel, rot, ang_vel))
    
    def tick(self, state, t):
        """
        在时间 t 下获取期望轨迹，仅返回状态
        """
        if not isinstance(t, float):
            raise ValueError("Tick 参数 t 必须是浮点数")
    
        return State(self._traj(state, t))
