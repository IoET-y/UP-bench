import pandas as pd
import numpy as np
from auv_control import State
from AUVControl.auv_control.planning.base import BasePlanner
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
import random
import torch.distributions as D
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import namedtuple
import holoocean
import pickle

# Set up logging to output to the console and file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LQRSACPlanner(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=20, start=None, end=None, state_dim=30, action_dim=6, lr=1e-4, gamma=0.95, tau=0.003, alpha=0.2, batch_size=512, replay_buffer_size=1000000): #bs up to 256
        # Memory for storing experience
        self.replay_buffer_size = replay_buffer_size
        # Memory for storing experience
        self.memory = deque(maxlen=self.replay_buffer_size)
        #rewards for episode
        self.episode_distance_reward = 0
        self.episode_current_utilization_reward = 0
        self.episode_safety_reward = 0
        self.episode_reach_target_reward = 0
        
        self.episode_memory = [] 
        self.episode_memory_success = []
        # for dynamic adjust env reset
        
        self.reach_targe_times = 0
        self.step_cnt = 0
        # Parameters
        self.num_seconds = num_seconds
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        
        # Setup goal
        self.start = np.array([0, 0, -5]) if start is None else start
        self.end = np.array([50, 50, -20]) if end is None else end
        self.og_distance_to_goal = np.linalg.norm(self.start - self.end)
        self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)
        # Setup environment
        self.size = np.array([80, 80, 25])
        self.bottom_corner = np.array([-10, -10, -25])
        self.__MAX_THRUST = 77
        # Setup obstacles
        self.num_obstacles = num_obstacles

        
        # self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        # self.obstacle_loc = np.random.beta(1.5, 1.5, (num_obstacles, 3)) * self.size + self.bottom_corner
        # for i in range(self.num_obstacles):
        #     while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
        #         self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner
        self.predefined_obstacle_distributions = []       
        self.distance_to_nearest_obstacle = 0
        self.obstacle_loc = []
        
        # Neural Networks
        self.policy_net = self._build_policy_network().to(device)
        self.q_net1 = self._build_q_network().to(device)
        self.q_net2 = self._build_q_network().to(device)
        self.target_q_net1 = self._build_q_network().to(device)
        self.target_q_net2 = self._build_q_network().to(device)

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
        self.previous_distance_to_goal = np.linalg.norm(self.start - self.end)
        self.epsd = 0
        self.done = False
        self.first_reach_close = 0
        self.inx_current = 0.6
        self.inx_distance = 1.23

        #for static last statue penaty use
        self.static_cnt = 0
        self.last_distance_to_goal = 0 
        self.static_on = False
        self.closest_pos = 0

        #action parameter
        self.action_dim = 6  # [lin_accel_x, lin_accel_y, lin_accel_z, ang_accel_x, ang_accel_y, ang_accel_z]
        self.__MAX_LIN_ACCEL = 0.5
        self.__MAX_ANG_ACCEL = 2
        self.limits_accel = [self.__MAX_LIN_ACCEL, self.__MAX_LIN_ACCEL, self.__MAX_LIN_ACCEL,self.__MAX_ANG_ACCEL, self.__MAX_ANG_ACCEL, self.__MAX_ANG_ACCEL]

        # 洋流模型参数 ref：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9887798
        self.current_strength = 0.5
        self.current_frequency = np.pi
        self.current_mu = 0.25
        self.current_omega = 2.0
        # 时间相关参数
        self.ticks_per_sec = 100  # 从配置文件读取
        self.current_time = 0.0  # 当前时间初始化为 0
        # agent物理参数
        self.gravity = 9.81
        self.cob = np.array([0, 0, 5.0]) / 100  # 中心浮力
        self.m = 31.02
        self.rho = 997
        self.V = self.m / self.rho
        self.J = np.eye(3) * 2  # 惯性张量
    
    def load_obstacle_distributions(self, filename="predefined_obstacles.pkl"):

        with open(filename, 'rb') as f:
            self.predefined_obstacle_distributions = pickle.load(f)
        print(f"Predefined obstacle distributions loaded from {filename}.")

    def save_obstacle_distributions(self, filename="predefined_obstacles.pkl"):

        with open(filename, 'wb') as f:
            pickle.dump(self.predefined_obstacle_distributions, f)
        print(f"Predefined obstacle distributions saved to {filename}.")
        
    def load_predefined_obstacles(self, index):

        if index < len(self.predefined_obstacle_distributions):
            self.obstacle_loc = self.predefined_obstacle_distributions[index]
        else:
            raise IndexError("Predefined obstacle index out of range.")

    def setup_obstacles(self, num_obstacles=20, grid_size=(4, 4, 2), train=True):
        self.memory.clear()
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
            max_attempts = 10  # 防止无限循环
            attempts = 0
            while (np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or 
                   np.linalg.norm(self.obstacle_loc[i] - self.end) < 10):
                if attempts >= max_attempts:
                    break  # 强制退出
                cell = random.choice(selected_cells)
                x_min, x_max, y_min, y_max, z_min, z_max = cell
                self.obstacle_loc[i] = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
                attempts += 1
    
        # Debug: Print generated obstacles
        print(f"Generated obstacles: {self.obstacle_loc}")
        print(f"Number of obstacles: {len(self.obstacle_loc)}")
    
        return self.obstacle_loc

    def calculate_ocean_current(self, position, time):
        if isinstance(time, torch.Tensor):
            time = time.cpu().numpy()  # Move to CPU and convert to NumPy
        """计算洋流的三维速度"""
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()  # Move to CPU and convert to NumPy
        x, y, z = position

        A = self.current_strength
        mu = self.current_mu
        omega = self.current_omega

        a_t = mu * np.sin(omega * time)
        b_t = 1 - 2 * mu * np.sin(omega * time)
        f_x_t = a_t * x**2 + b_t * x

        psi = A * np.sin(self.current_frequency * f_x_t) * np.sin(self.current_frequency * y)

        # 计算速度分量
        v_x = -self.current_frequency * A * np.sin(self.current_frequency * f_x_t) * np.cos(self.current_frequency * y)
        v_y = self.current_frequency * A * np.cos(self.current_frequency * f_x_t) * np.sin(self.current_frequency * y)
        v_z = 0  # 假设垂直方向的洋流速度较小

        return np.array([v_x, v_y, v_z])

    def calculate_action_effect(self, action, ocean_current):
        """计算洋流对动作的影响"""
        area = 0.3 * 0.3  # AUV的横截面积（假设为正方形）
        drag_coefficient = 1.0  # 阻力系数

        # 相对速度和阻力计算
        relative_velocity = -ocean_current  # 假设AUV相对静止
        drag_force = 0.5 * drag_coefficient * self.rho * area * np.linalg.norm(relative_velocity) * relative_velocity

        # 加速度
        acceleration_due_to_drag = drag_force / self.m
        
        # 更新动作中的线性加速度
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()  # Move to CPU and convert to NumPy
        adjusted_action = np.copy(action)
        adjusted_action[:3] += acceleration_due_to_drag  # 更新线性加速度部分
        print("linear acceleration_due_to_ocean",acceleration_due_to_drag)
        return adjusted_action

    
    def setup_start(self):
        lower_bound = np.array([0, 0, -5])
        upper_bound = np.array([20, 20, -10])
        random_position = lower_bound + np.random.rand(3) * (upper_bound - lower_bound)
        self.start = random_position
        print("start point:",self.start)
        
    def get_start_end_obs(self):
        return self.start, self.end, self.obstacle_loc, self.obstacle_size
        
    def setup_end(self):
        lower_bound = np.array([30, 30, -5])
        upper_bound = np.array([50, 50, -25])
        random_position = lower_bound + np.random.rand(3) * (upper_bound - lower_bound)
        self.end = random_position
        print("end point:",self.end)
        
    # def setup_obstacles(self,num_obstacles=20):
    #     """Sets up obstacles with random positions and sizes."""
    #     self.memory.clear()
    #     self.num_obstacles = num_obstacles
    #     self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
    #     self.obstacle_loc = np.random.beta(1.5, 1.5, (self.num_obstacles, 3)) * self.size + self.bottom_corner
    #     for i in range(self.num_obstacles):
    #         while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
    #             self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner
                
    def remember(self, state, action,adjust_action, reward, next_state, done,next_adjusted_action):
        self.memory.append((state, action,adjust_action, reward, next_state, done,next_adjusted_action))

            
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

    def _build_policy_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh(),  # Ensures output is between -1 and 1
        )

    def _build_q_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            #nn.Softplus(),#++++
        )

    
    def select_action(self, state, inference=False):
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.policy_net(state_tensor).cpu().numpy()[0]
        if not inference:
            if self.previous_distance_to_goal > 10:
                noise_scale = 0.3  # 较远距离，较大的噪声
            elif self.previous_distance_to_goal > 5:
                noise_scale = 0.2
            elif self.previous_distance_to_goal > 2:
                noise_scale = 0.15
            else:
                noise_scale = 0.1  # 靠近目标时，减少噪声
            #noise_scale = noise_scale * np.array(self.limits_accel)
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            
        if self.previous_distance_to_goal > 10:
            action[:3] = np.clip(action[:3] * self.__MAX_LIN_ACCEL, -self.__MAX_LIN_ACCEL, self.__MAX_LIN_ACCEL)  
            action[3:] = np.clip(action[3:] * self.__MAX_ANG_ACCEL, -self.__MAX_ANG_ACCEL, self.__MAX_ANG_ACCEL)  #
        elif self.previous_distance_to_goal > 5:
            action[:3] = np.clip(0.8 * action[:3] * self.__MAX_LIN_ACCEL, -self.__MAX_LIN_ACCEL, self.__MAX_LIN_ACCEL)
            action[3:] = np.clip(0.8 * action[3:] * self.__MAX_ANG_ACCEL, -self.__MAX_ANG_ACCEL, self.__MAX_ANG_ACCEL)
        elif self.previous_distance_to_goal > 2:
            action[:3] = np.clip(0.5 * action[:3] * self.__MAX_LIN_ACCEL, -self.__MAX_LIN_ACCEL, self.__MAX_LIN_ACCEL)
            action[3:] = np.clip(0.5 * action[3:] * self.__MAX_ANG_ACCEL, -self.__MAX_ANG_ACCEL, self.__MAX_ANG_ACCEL)
        else:
            action[:3] = np.clip(0.3 * action[:3] * self.__MAX_LIN_ACCEL, -self.__MAX_LIN_ACCEL, self.__MAX_LIN_ACCEL)
            action[3:] = np.clip(0.3 * action[3:] * self.__MAX_ANG_ACCEL, -self.__MAX_ANG_ACCEL, self.__MAX_ANG_ACCEL)

        if self.static_cnt >5 and self.previous_distance_to_goal < 10:
            action = 2*action
            action = np.clip(action, -np.array(self.limits_accel), np.array(self.limits_accel))

        return action
        
    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = list(zip(*random.sample(self.memory, self.batch_size)))
        
        states = torch.FloatTensor(transitions[0]).to(device)
        actions = torch.FloatTensor(transitions[1]).to(device)
        adjust_actions = torch.FloatTensor(transitions[2]).to(device)
        rewards = torch.FloatTensor(transitions[3]).to(device)
        next_states = torch.FloatTensor(transitions[4]).to(device)
        dones = torch.FloatTensor(transitions[5]).to(device)
        next_adjust_action =torch.FloatTensor(transitions[6]).to(device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states)
            #目标q值要反映下一状态的最优策略行为，所以我在这里使用未经过洋流调整的next_actions来保正策略学习的分布
            next_q1 = self.target_q_net1(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.target_q_net2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2).squeeze(-1)  # Ensure next_q is [batch_size]
            target_q = rewards + self.gamma * (1 - dones) * next_q  # Rewards, dones, and next_q must all be [batch_size]

        #而对于实际q值，环境的真实state变化由adjusted_action决定，因此用adjusted_action来学实际的q值
        current_q1 = self.q_net1(torch.cat([states, adjust_actions], dim=1)).squeeze()
        current_q2 = self.q_net2(torch.cat([states, adjust_actions], dim=1)).squeeze()
        q_loss1 = nn.MSELoss()(current_q1, target_q)
        q_loss2 = nn.MSELoss()(current_q2, target_q)
        
        # 梯度裁剪 20241104
        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)  # 添加梯度裁剪
        self.q_optimizer1.step()
    
        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)  # 添加梯度裁剪
        self.q_optimizer2.step()
    
        # Calculate policy loss and update policy network
        #policynet要maxmize Q(s, a)。所以用policynet输出的og action 而非 adjusted_action 来计算loss,让policynet去逐渐学会生成适合洋流环境的动作。
        policy_loss = -(self.q_net1(torch.cat([states, self.policy_net(states)], dim=1))).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # 添加梯度裁剪
        self.policy_optimizer.step()

        wandb.log({
            "episode": self.epsd + 1,
            "q_loss1": q_loss1,
            "q_loss2": q_loss2,
            "policy_loss": policy_loss,
        })
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, env, num_episodes=500, max_steps=3000, model_path="sac_best_model.pth"):
        log_file_path = "SAC_"+str(max_steps)+"_log.csv"
        log_file = log_file_path
        columns = ["episode", "step", "real_state", "movement_reward", "obstacle_reward", 
                   "out_of_box_penalty", "reach_target_reward", "action_smooth_reward","action_align_reward", "total_reward"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(log_file, index=False)
        max_step_increment = 1024
        self.maxstep = max_steps

        wandb.init(project="auv_RL_control_project_SAC_new", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "tau": self.tau,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "MAX_LIN_ACCEL":self.__MAX_LIN_ACCEL,
            "maxstep": max_steps,
            "current align rd discount": self.inx_current,
            "distance rd discount": self.inx_distance ,
        })


        ts = 1 / scenario["ticks_per_sec"]
        last_position = None  # 用于保存上一个 episode 的结束位置
        tmp = max_steps
        self.step_cnt = 0

        self.predefined_obstacle_distributions = [self.setup_obstacles(num_obstacles=20, train=True) for _ in range(100)]
        print("predefined_obstacle_distributions is",self.predefined_obstacle_distributions)
        obs_file_path = "SAC_"+str(max_steps)+"_train_obstacles.pkl"
        self.save_obstacle_distributions(obs_file_path)
        
        obstacle_index = 0  # 遍历使用障碍物分布
        self.obstacle_loc = self.predefined_obstacle_distributions[obstacle_index]
        
        for episode in range(num_episodes):
            episode_data = []
            self.episode_distance_reward = 0
            self.episode_current_utilization_reward = 0
            self.episode_safety_reward = 0
            self.episode_reach_target_reward = 0
            self.static_on = False
            self.static_cnt = 0
            self.first_reach_close = 0
            self.epsd = episode
            
            logging.info(f"Episode {episode + 1} starting")
            # 混合策略：判断是否达到episode_threshold，如果达到，则使用上一个结束位置   20241029

            if self.done == True: #20241030 introduce dynamic start end point
                #self.episode_memory_success.extend(self.episode_memory)
                self.static_cnt = 0
                self.setup_start()
                self.setup_end()
                obstacle_index += 1  # 遍历使用障碍物分布
                self.obstacle_loc = self.predefined_obstacle_distributions[obstacle_index]
                #self.setup_obstacles()
                #self.memory.extend(self.episode_memory_success) #1109
                
            state_info = env.reset()
            env.agents["auv0"].teleport(self.start,[0,0,0])
            sensors = env.tick()
            state_info=sensors
            state = State(state_info)
            print("self start:",self.start)
            print("auv position:",state.vec[0:3])
            distance_to_goal = np.linalg.norm(self.start - self.end)
            self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)
            self.og_distance_to_goal = np.linalg.norm(self.start - self.end)
            self.closest_pos =  self.prev_distance_to_goal
            self.done = False     
            done = False
            step_count = 0
            total_reward = 0
            added_steps = 0
            max_steps = tmp
            self.distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
            self.episode_memory = []
            while not done and step_count < max_steps:
                self.max_stp = max_steps
                self.step_cnt += 1
                sensors = env.tick()
                t = sensors["t"]
                true_state = State(sensors)
         
                relative_position_to_goal = self.end - true_state.vec[0:3]
                sense_dto = self.distance_to_nearest_obstacle if self.distance_to_nearest_obstacle < 10 else -1 # 声纳的可见范围一般为10米

                ocean_current = self.calculate_ocean_current(true_state.vec[0:3], self.current_time)
                
                real_state = np.append(true_state.vec[0:], true_state.bias[0:])
                real_state = np.append(real_state, true_state.add)
                real_state = np.append(real_state, ocean_current)
                real_state = np.append(real_state, relative_position_to_goal)
                real_state = np.append(real_state, distance_to_goal)
                real_state = np.append(real_state, sense_dto)
                real_state = np.append(real_state, done)
                
                self.current_time += 1 / self.ticks_per_sec
                
                action = self.select_action(real_state)
                adjusted_action = self.calculate_action_effect(action, ocean_current)
                
                ocean_current = self.calculate_ocean_current(true_state.vec[0:3], self.current_time)
                
                next_state_info = env.step(adjusted_action, ticks=1, publish=False)
                next_state = State(next_state_info)
                distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
                self.previous_distance_to_goal = distance_to_goal
                done = np.linalg.norm(next_state.vec[0:3] - self.end) < 1
                self.done = done
                reward,dst_rd,current_use_rd,safe_rd,reach_tg_rd =  self.calculate_reward(true_state, next_state, action)
                
                relative_position_to_goal_next = self.end - next_state.vec[0:3]
                sense_dto_next = self.distance_to_nearest_obstacle if self.distance_to_nearest_obstacle < 10 else -1 # 声纳的可见范围一般为10米
                real_next_state = np.append(next_state.vec[0:], next_state.bias[0:])
                real_next_state = np.append(real_next_state, next_state.add)
                real_next_state = np.append(real_next_state, ocean_current)
                real_next_state = np.append(real_next_state, relative_position_to_goal_next)
                real_next_state = np.append(real_next_state, distance_to_goal)
                real_next_state = np.append(real_next_state, sense_dto_next) #distance_to_nearest_obstacle   感知范围强相关 ！！！
                real_next_state = np.append(real_next_state, done)
                next_action = self.select_action(real_next_state)
                next_adjusted_action = self.calculate_action_effect(next_action, ocean_current)
                
                self.closest_pos = distance_to_goal if distance_to_goal < self.closest_pos else self.closest_pos
                #total_reward += reward
                if done:
                    total_reward = reach_tg_rd
                else:
                    total_reward += reward

                if distance_to_goal < 15 and step_count >= max_steps - 1 and added_steps <max_step_increment:
                    max_steps += 64
                    added_steps += 64
                    logging.info(f"Increasing max steps to {max_steps} for additional exploration.")
                self.episode_memory.append((real_state, action, adjusted_action, reward, real_next_state, done))
                self.remember(real_state, action, adjusted_action, reward, real_next_state, done,next_adjusted_action)
                self.update_policy()
                
                step_count += 1
                wandb.log({
                    "inloop_step_count": step_count,
                    "inloop_distance_to_nearest_obstacle": self.distance_to_nearest_obstacle,
                    "inloop_distance_to_goal":distance_to_goal,
                    "inloop_distance_reward":dst_rd,
                    "inloop_current_utilization_reward":current_use_rd,
                    "inloop_safety_reward":safe_rd,
                    "inloop_reach_target_reward":reach_tg_rd
                })
                wandb.log({
                    "step_count_keepon": self.step_cnt ,
                    "distance_to_goal_keepon":distance_to_goal,
                    "inloop_distance_reward":dst_rd,
                    "current_utilization_reward_keepon":current_use_rd,
                    "safety_reward_keepon":safe_rd,
                    "reach_target_reward_keepon":reach_tg_rd,
                })
                if done:
                    self.reach_targe_times += 1
                    wandb.log({
                        "episode": episode + 1,
                        "reach_targe_times": self.reach_targe_times
                    })
                    
                    success_path = "successful_"+str(episode)+"_"+model_path
                    self.save_model(episode + 1, success_path)
                    break

            wandb.log({
                "episode": episode + 1,
                "epsd_total_reward": total_reward,
                "epsd_distance_to_goal": distance_to_goal,
                "epsd_distance_reward":self.episode_distance_reward,
                "epsd_current_utilization_reward":self.episode_current_utilization_reward,
                "epsd_safety_reward":self.episode_safety_reward,
                "epsd_reach_target_reward":self.episode_reach_target_reward
            })
            if abs(self.last_distance_to_goal - distance_to_goal) < 10 and episode > 300 and distance_to_goal < 10:
                self.static_cnt =  self.static_cnt + 1
                self.static_on = True
            else:
                self.static_cnt = 0
                self.static_on = False

            self.last_distance_to_goal = distance_to_goal

            logging.info(f"Episode {episode + 1} completed - Total Reward: {total_reward}")
            self.save_model(episode + 1, model_path)
        
    def calculate_reward(self, state, next_state, action):
        #movement rewards
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        progress = self.prev_distance_to_goal - distance_to_goal
        distance_reward = self.inx_distance * np.arctan(10 * (progress - 0.015)) #10 max acce : 1; 0.5 maxacce : 0.02   5->10

        # current align rewards
        ocean_current = self.calculate_ocean_current(next_state.vec[0:3], self.current_time)
        auv_velocity = next_state.vec[3:6]
        utilization = np.dot(auv_velocity, ocean_current) / (np.linalg.norm(ocean_current) + 1e-5)
        current_utilization_reward = self.inx_current * np.arctan(3 * (utilization - 0.5)) #5->3

        # safty rewards
        distances_to_obstacles = [np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc]
        min_distance = np.min(distances_to_obstacles)
        closest_obstacle = self.obstacle_loc[np.argmin(distances_to_obstacles)]
        velocity_vector = next_state.vec[3:6]
        direction_to_obstacle = (closest_obstacle - next_state.vec[0:3]) / (min_distance + 1e-5)
        relative_speed = np.dot(velocity_vector, direction_to_obstacle)
        # 动态调整安全距离阈值，基于相对速度，值范围在1到5之间
        safe_distance_threshold = max(1, min(5, 3 + relative_speed * 0.1))
        
        safety_reward = -20 * np.exp(-min_distance) if min_distance < safe_distance_threshold else 0

        if distance_to_goal < 10 and self.first_reach_close <= 1:
            self.first_reach_close = self.first_reach_close + 1 #. new try 20241029  
            
        if distance_to_goal < 1:
            reach_target_reward = 500
        else:
            reach_target_reward = 0
            
        if distance_to_goal < 10 and self.first_reach_close == 1:
            first_close_reward = 200 / (distance_to_goal + 1)
        else:
            first_close_reward = 0
        reach_target_reward = reach_target_reward 

        self.episode_distance_reward += distance_reward
        self.episode_current_utilization_reward += current_utilization_reward
        self.episode_safety_reward += safety_reward
        self.episode_reach_target_reward += reach_target_reward

        self.previous_action = action
        self.prev_distance_to_goal = distance_to_goal

        total_reward = distance_reward + current_utilization_reward + safety_reward+first_close_reward # +reach_target_reward
        return total_reward,distance_reward,current_utilization_reward,safety_reward,reach_target_reward
    
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

    def pad_state(self, state, target_dim=30):
        state_flat = np.ravel(state)
        if state_flat.shape[0] < target_dim:
            padded_state = np.pad(state_flat, (0, target_dim - state_flat.shape[0]), 'constant', constant_values=0)
        else:
            padded_state = state_flat[:target_dim]
        return padded_state

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
        relative_position_to_goal = self.end - state.vec[0:3]        
        return distance_to_goal,distance_to_nearest_obstacle,relative_position_to_goal

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
