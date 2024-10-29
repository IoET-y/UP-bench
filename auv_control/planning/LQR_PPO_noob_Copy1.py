import numpy as np
from auv_control import State
from .base import BasePlanner
from collections import deque
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
import torch.distributions as D
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'value', 'done'))



# Set up logging to output to the console and file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LQRPPOPlanner(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=10, start=None, end=None, state_dim=21, action_dim=8, lr=1e-3, gamma=0.95, clip_epsilon=0.1, entropy_coef=0.1, value_loss_coef=0.2, ppo_epochs=12, batch_size=128
                ): #	•	clip_epsilon 增加，允许更大幅度的策略更新，可能加速训练收敛.entropy_coef 提高至，鼓励更多探索，避免过早陷入局部最优。value_loss_coef 提升到，使得值函数的优化对总损失的贡献更大，增强价值估计的稳定性
        # Parameters
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

        # Setup goal
        self.start = np.array([0, 0, 0]) if start is None else start
        self.end = np.array([40, 40, -20]) if end is None else end
        self.og_distance_to_goal = np.linalg.norm(self.start - self.end)
        self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)
        # Setup environment
        self.size = np.array([50, 50, 25])
        self.bottom_corner = np.array([-5, -5, -25])
        self.__MAX_THRUST = 77
        #Setup obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (num_obstacles, 3)) * self.size + self.bottom_corner
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner

#         # Neural Networks
        self.policy_net = self._build_network().to(device)
        self.value_net = self._build_value_network().to(device)

        # Apply weight initialization to policy_net and value_net
        self.policy_net.apply(self.initialize_weights)
        self.value_net.apply(self.initialize_weights)
        
        # Optimizer setup    
        

                
        # Memory for storing experience
        self.memory = []
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # Setup base planner properties
        super().__init__()
        self.pos_func = self.pos_func
        self.rot_func = self.rot_func

        # Store previous action for smoothness reward
        self.previous_action = np.zeros(action_dim)

        # LQR parameters
        self.m = 31.02  # Mass of the AUV
        self.J = np.eye(3) * 2  # Moment of inertia (example values)
        self.Q_lqr = np.diag([100, 100, 100, 1, 1, 1])  # State cost matrix
        self.R_lqr = np.diag([0.01, 0.01, 0.01])  # Control cost matrix
        self.epsd = 0
        self.episode = 0
        self.fst_flag = 0
        self.scd_flag = 0
        self.trd_flag = 0


    def remember(self, state, action, log_prob, reward, value, done):
        """存储当前步骤的经验数据到 memory 中"""
        self.memory.append(Transition(state, action, log_prob, reward, value, done))

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # 或其他适合的初始化方法
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)     
                
    def setup_obstacles(self):
        """Sets up obstacles with random positions and sizes."""
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (self.num_obstacles, 3)) * self.size + self.bottom_corner
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner
        
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    
    def _build_value_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

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

    def train(self, env, num_episodes=500, max_steps=3000,ctrain = False, model_path="ppo_best_model.pth"):
        wandb.init(project="auv_RL_control_project_1e3", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef, 
            "value_loss_coef":self.value_loss_coef,
            "ppo_epochs":self.ppo_epochs,
            "entropy_coef":self.batch_size,
        })
        
        controller = LQR()
        observer = InEKF()
        planner = Astar(max_steps)
        ts = 1 / scenario["ticks_per_sec"]
        done = False
        for episode in range(num_episodes):
            
            self.epsd = episode
            logging.info(f"Episode {episode + 1} starting")
            if done == True:
                self.setup_obstacles()            
            state_info = env.reset()
            state = State(state_info)
            done = False
            step_count = 0
            total_reward = 0
            distance_to_goal = np.linalg.norm(self.start - self.end)
            distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
            self.fst_flag = 0
            self.scd_flag = 0
            self.trd_flag = 0
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
        
        
    def select_action(self, state, lqr_action, weight, inference=False):
        # 处理状态，确保其维度符合网络输入
        state_tensor = torch.FloatTensor(self.pad_state(state)).unsqueeze(0).to(device)
        
        # 通过策略网络生成动作输出
        action_logits = self.policy_net(state_tensor)[0]
        
        # 将策略网络的输出缩放至推力范围，使动作符合实际的推力限制
        policy_action = action_logits * self.__MAX_THRUST
        
        # 使用一个非线性权重调整策略，以提升动作组合的灵活性
        distance_to_goal = np.linalg.norm(state[:3] - self.end)
        lqr_weight = max(0.3, min(0.7, 1 - distance_to_goal / self.og_distance_to_goal))
        policy_weight = 1 - lqr_weight
        
        # 使用加权策略组合LQR动作和策略网络动作
        combined_action = lqr_weight * lqr_action + policy_weight * policy_action.detach().cpu().numpy()
    
        # 若非推理模式，则加入一个小的高斯噪声以提升探索能力
        if not inference:
            noise = np.random.normal(0, 0.1 * self.__MAX_THRUST, combined_action.shape)
            combined_action = np.clip(combined_action + noise, -self.__MAX_THRUST, self.__MAX_THRUST)
    
        return combined_action, torch.zeros(1)  # 返回动作和一个占位的log_prob
    
    def update_policy(self):
        transitions = Transition(*zip(*self.memory))
        states = torch.FloatTensor([self.pad_state(s) for s in transitions.state]).to(device)
        actions = torch.FloatTensor(transitions.action).to(device)
        values = torch.FloatTensor(transitions.value).to(device)
        rewards = torch.FloatTensor(transitions.reward).to(device)
        dones = torch.FloatTensor(transitions.done).to(device)
    
        next_value = self.value_net(states[-1].unsqueeze(0)).item()
        advantages = self.compute_gae(next_value, rewards, values, dones).detach()
        returns = advantages + values
    
        for _ in range(self.ppo_epochs):
            # 动作直接预测，无需分布采样
            predicted_actions = self.policy_net(states)
            actor_loss = nn.MSELoss()(50*predicted_actions, actions)  # 使用 MSE 损失
    
            value_preds = self.value_net(states).squeeze()
            critic_loss = nn.MSELoss()(value_preds, returns)
    
            total_loss = actor_loss + self.value_loss_coef * critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()
    
        self.memory.clear()
        wandb.log({
            'episode': self.epsd,
            "total_loss": total_loss,
        })                
        
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
        
    def pad_state(self, state, target_dim=21):
        state_flat = np.ravel(state)
        if state_flat.shape[0] < target_dim:
            padded_state = np.pad(state_flat, (0, target_dim - state_flat.shape[0]), 'constant', constant_values=0)
        else:
            padded_state = state_flat[:target_dim]
        return padded_state
        
    
    def calculate_reward(self, state, next_state, action):
        # Define the box boundaries
        box_x_min, box_y_min, box_z_min = self.bottom_corner
        box_x_max, box_y_max, box_z_max = self.bottom_corner + self.size

        # Check if the agent is outside the box
        is_outside_box = (
            next_state.vec[0] < box_x_min or next_state.vec[0] > box_x_max or
            next_state.vec[1] < box_y_min or next_state.vec[1] > box_y_max or
            next_state.vec[2] < box_z_min or next_state.vec[2] > box_z_max
        )
        outside_box_penalty = -2 if is_outside_box else 0

        # Calculate the distance to the target
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        progress_reward = 10 * (self.prev_distance_to_goal - distance_to_goal)
        distance_reward = 0.5 * np.log(1 / (distance_to_goal + 1))  # Logarithmic distance reward

        # Penalties for proximity to obstacles
        distance_to_nearest_obstacle = np.min([np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc])
        obstacle_penalty = -0.5 * np.exp(-distance_to_nearest_obstacle)
        safety_reward = 0.3 if distance_to_nearest_obstacle > 2 else 0  # Reward for maintaining safe distance

        # Static penalty: Encourage movement by penalizing small displacements over multiple steps
        static_penalty = -0.5 if np.linalg.norm(state.vec[0:3] - next_state.vec[0:3]) < 0.01 else 0

        # Action smoothness penalty: Encourage smooth control inputs to avoid drastic changes in action
        #velocity_magnitude = np.linalg.norm(next_state.vec[3:6])
        action_smoothness_penalty = 0 #-0.02 * np.linalg.norm(action - self.previous_action) / (velocity_magnitude + 1)

        # Alignment reward: Encourage movement in the direction of the goal
        goal_direction = (self.end - state.vec[0:3]) / (np.linalg.norm(self.end - state.vec[0:3]) + 1e-5)
        velocity_direction = next_state.vec[3:6] / (np.linalg.norm(next_state.vec[3:6]) + 1e-5)
        alignment_reward = np.clip(1.0 * np.dot(goal_direction, velocity_direction), 0, 1)

        # Reach target reward: High reward when agent is very close to the target
        self.fst_flag += 1 if distance_to_goal < 0.75 * self.og_distance_to_goal else 0
        self.scd_flag += 1 if distance_to_goal < 0.5 * self.og_distance_to_goal else 0
        self.trd_flag += 1 if distance_to_goal < 0.25 * self.og_distance_to_goal else 0
        reach_target_reward = 100 if distance_to_goal < 2 else 0
        half_reach_reward = 50 if self.trd_flag ==1 else 0
        quato_reach_reward = 75 if self.scd_flag ==1 else 0
        quato_back_reach_reward = 25 if self.fst_flag ==1 else 0
        # Incline penalty: Penalize for extreme pitch or roll angles
        roll, pitch = state.vec[6], state.vec[7]
        incline_penalty = 0.001 * (max(roll - 15, 0) + max(170 - pitch, 0))

        # Aggregate rewards and penalties
        total_reward = (
            progress_reward + distance_reward +
            obstacle_penalty + safety_reward +
            static_penalty + action_smoothness_penalty +
            alignment_reward + reach_target_reward +
            outside_box_penalty - incline_penalty + quato_back_reach_reward + quato_reach_reward + half_reach_reward
        )

        # Update previous distance to goal for future progress calculation
        self.prev_distance_to_goal = distance_to_goal

        # Normalize the total reward to avoid excessively large or small values
        normalization_factor = sum(abs(x) for x in [
            progress_reward, distance_reward, obstacle_penalty, 
            safety_reward, static_penalty, action_smoothness_penalty, 
            alignment_reward, reach_target_reward, outside_box_penalty, incline_penalty
        ]) + 1e-5

        normalized_reward = total_reward / normalization_factor
        return normalized_reward

    
    def pos_func(self, state,t):
        """
        Position function to calculate desired position at time t using the policy model.
        """
        with torch.no_grad():
            state = self.pad_state(state)  
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get position values from the policy network
            predicted_pos = self.policy_net(state_tensor)[:, :3]  # First three values correspond to position
            return predicted_pos.cpu().numpy().flatten()

    def rot_func(self, state, t):
        """
        Rotation function to calculate desired rotation at time t using the policy model.
        """
        with torch.no_grad():
            state = self.pad_state(state)  
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