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
import random
import torch.distributions as D
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import namedtuple
import holoocean

# Set up logging to output to the console and file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LQRSACPlanner(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=20, start=None, end=None, state_dim=21, action_dim=8, lr=3e-4, gamma=0.95, tau=0.003, alpha=0.2, batch_size=512, replay_buffer_size=10000000): #bs up to 256


        #observation rewards
        self.move_reward = 0
        self.obs_reward = 0
        self.outbox_reward = 0
        self.reachtarget_reward = 0
        self.action_reward = 0
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
        self.replay_buffer_size = replay_buffer_size

        # Setup goal
        self.start = np.array([0, 0, -5]) if start is None else start
        self.end = np.array([50, 50, -20]) if end is None else end
        self.og_distance_to_goal = np.linalg.norm(self.start - self.end)
        self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)
        # Setup environment
        self.size = np.array([60, 60, 25])
        self.bottom_corner = np.array([-5, -5, -25])
        self.__MAX_THRUST = 77
        # Setup obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (num_obstacles, 3)) * self.size + self.bottom_corner
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner
                
        self.distance_to_nearest_obstacle = np.min([np.linalg.norm(self.start - obs) for obs in self.obstacle_loc])
        
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

        # Memory for storing experience
        self.memory = deque(maxlen=self.replay_buffer_size)

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
        # LQR parameters
        self.m = 31.02  # Mass of the AUV
        self.J = np.eye(3) * 2  # Moment of inertia (example values)
        self.Q_lqr = np.diag([100, 100, 100, 1, 1, 1])  # State cost matrix
        self.R_lqr = np.diag([0.01, 0.01, 0.01])  # Control cost matrix

        #for static last statue penaty use
        self.static_cnt = 0
        self.last_distance_to_goal = 0 
        self.static_on = False
        self.closest_pos = 0
        
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
        
    def setup_obstacles(self,num_obstacles=20):
        """Sets up obstacles with random positions and sizes."""
        self.memory.clear()
        self.num_obstacles = num_obstacles
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (self.num_obstacles, 3)) * self.size + self.bottom_corner
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner
                
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

            
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
            nn.Linear(256, self.action_dim),
            nn.Tanh(),  # Ensures output is between -1 and 1
        )

    def _build_q_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def select_action(self, state, inference=False):
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
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            
        if self.previous_distance_to_goal > 10:
            action = np.clip(action * self.__MAX_THRUST, -self.__MAX_THRUST, self.__MAX_THRUST)
        elif self.previous_distance_to_goal > 5:
            action = np.clip(action * 0.8 * self.__MAX_THRUST, -0.5*self.__MAX_THRUST, 0.5*self.__MAX_THRUST)
        elif self.previous_distance_to_goal > 2:
            action = np.clip(action * 0.5 * self.__MAX_THRUST, -0.3*self.__MAX_THRUST, 0.3*self.__MAX_THRUST)
        else:
            action = np.clip(action * 0.3 * self.__MAX_THRUST, -0.15*self.__MAX_THRUST, 0.15*self.__MAX_THRUST)
        if self.static_cnt >5 and elf.previous_distance_to_goal < 10:
            action = 2*action
            action = np.clip(action * self.__MAX_THRUST, -self.__MAX_THRUST, self.__MAX_THRUST)
            
        return action

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = list(zip(*random.sample(self.memory, self.batch_size)))
        states = torch.FloatTensor(transitions[0]).to(device)
        actions = torch.FloatTensor(transitions[1]).to(device)
        rewards = torch.FloatTensor(transitions[2]).to(device)
        next_states = torch.FloatTensor(transitions[3]).to(device)
        dones = torch.FloatTensor(transitions[4]).to(device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states)
            next_q1 = self.target_q_net1(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.target_q_net2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2).squeeze(-1)  # Ensure next_q is [batch_size]
            target_q = rewards + self.gamma * (1 - dones) * next_q  # Rewards, dones, and next_q must all be [batch_size]


        current_q1 = self.q_net1(torch.cat([states, actions], dim=1)).squeeze()
        current_q2 = self.q_net2(torch.cat([states, actions], dim=1)).squeeze()
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
        })

        controller = LQR()
        observer = InEKF()
        planner = Astar(max_steps)
        ts = 1 / scenario["ticks_per_sec"]
        last_position = None  # 用于保存上一个 episode 的结束位置
        tmp = max_steps
        self.step_cnt = 0
        for episode in range(num_episodes):
             
            self.move_reward = 0
            self.obs_reward = 0
            self.outbox_reward = 0
            self.reachtarget_reward = 0
            self.action_reward = 0
            
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
                self.setup_obstacles()
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
                self.step_cnt = step_count + 1
                sensors = env.tick()
                t = sensors["t"]
                true_state = State(sensors)
                est_state = observer.tick(sensors, ts)
                des_state = planner.tick(t)
                lqr_action = controller.u(est_state, des_state)
                real_state = np.append(true_state.vec[0:], true_state.bias[0:])
                real_state = np.append(real_state, distance_to_goal)
                real_state = np.append(real_state, self.distance_to_nearest_obstacle)
                real_state = np.append(real_state, done)
                combined_action = self.select_action(real_state)

                next_state_info = env.step(combined_action, ticks=1, publish=False)
                next_state = State(next_state_info)
                distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
                self.previous_distance_to_goal = distance_to_goal
                done = np.linalg.norm(next_state.vec[0:3] - self.end) < 1
                self.done = done
                
                reward = self.calculate_reward(true_state, next_state, combined_action)
                
                real_next_state = np.append(next_state.vec[0:], next_state.bias[0:])
                real_next_state = np.append(real_next_state, distance_to_goal)
                real_next_state = np.append(real_next_state, self.distance_to_nearest_obstacle)
                real_next_state = np.append(real_next_state, done)

                
                self.closest_pos = distance_to_goal if distance_to_goal < self.closest_pos else self.closest_pos
                total_reward += reward
               # 当接近终点但未达到目标，增加 max_steps
                if distance_to_goal < 15 and step_count >= max_steps - 1 and added_steps <max_step_increment:
                    max_steps += 64
                    added_steps += 64
                    logging.info(f"Increasing max steps to {max_steps} for additional exploration.")
                self.episode_memory.append((real_state, combined_action, reward, real_next_state, done))
                self.remember(real_state, combined_action, reward, real_next_state, done)
                self.update_policy()
                
                step_count += 1
                wandb.log({
                    "step_count": step_count,
                    "distance_to_nearest_obstacle": self.distance_to_nearest_obstacle,
                    "everystep_distance_to_goal":distance_to_goal
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
                "total_reward": total_reward,
                "distance to goal": distance_to_goal,
                "episode_movement_reward": self.move_reward,
                "episode_obstacle_reward": self.obs_reward,
                "episode_out_of_box_penalty":  self.outbox_reward,
                "episode_reach_target_reward": self.reachtarget_reward,
                "episode_action_reward": self.action_reward #1109
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
        # 计算各类奖励和惩罚
        movement_reward = self._movement_reward(state, next_state, action)
        obstacle_reward = self._obstacle_reward(next_state)
        out_of_box_penalty = self._out_of_box_penalty(next_state)
        reach_target_reward = self._reach_target_reward(next_state)
        action_smooth_reward = self._action_smooth_reward(state,action)
        self.move_reward += movement_reward
        self.obs_reward += obstacle_reward
        self.outbox_reward += out_of_box_penalty
        self.reachtarget_reward += reach_target_reward
        self.action_reward += action_smooth_reward
        wandb.log({
            "step_count":self.step_cnt,
            "movement_reward": movement_reward,
            "obstacle_reward": obstacle_reward,
            "out_of_box_penalty": out_of_box_penalty,
            "reach_target_reward": reach_target_reward,
            "action_smooth_reward": action_smooth_reward
        })

        total_reward = movement_reward + obstacle_reward + out_of_box_penalty + reach_target_reward
        
        self.prev_distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        self.previous_action = action
        return total_reward
    
    #movement
    def _action_smooth_reward(self, state, action):
# 这是环境中agent的configuration里关于推进器位置和方向的描述：
#         # True thruster locations
#         self.thruster_p = np.array([[18.18, -22.14, -4],
#                                     [18.18, 22.14, -4],
#                                     [-31.43, 22.14, -4], # First element should be -31.43
#                                     [-31.43, -22.14, -4], # First element should be -31.43
#                                     [7.39, -18.23, -0.21], # First element should be 7.39
#                                     [7.39, 18.23, -0.21], # First element should be 7.39
#                                     [-20.64, 18.23, -0.21],
#                                     [-20.64, -18.23, -0.21]])/100

#         # Thruster directions
#         self.thruster_d = np.array([[0, 0, 1],
#                                     [0, 0, 1],
#                                     [0, 0, 1],
#                                     [0, 0, 1],
#                                     [np.sqrt(2), np.sqrt(2), 0],
#                                     [np.sqrt(2), -np.sqrt(2), 0],
#                                     [np.sqrt(2), np.sqrt(2), 0],
#                                     [np.sqrt(2), -np.sqrt(2), 0]])
        
        # 垂直方向推力为 Thrusters 1-4，主要影响 z 轴
        vertical_thrust = np.array([action[0], action[1], action[2], action[3]])
        z_movement = np.mean(vertical_thrust)  # z 轴运动由垂直推进器贡献
    
        # 水平方向推力为 Thrusters 5-8，影响 x 和 y 轴
        x_movement = (action[4] * np.sqrt(2) + action[5] * np.sqrt(2) + action[6] * np.sqrt(2) + action[7] * np.sqrt(2)) / 4
        y_movement = (action[4] * np.sqrt(2) - action[5] * np.sqrt(2) + action[6] * np.sqrt(2) - action[7] * np.sqrt(2)) / 4
    
        # 三维合成动作向量
        action_direction = np.array([x_movement, y_movement, z_movement])
        action_direction /= np.linalg.norm(action_direction) + 1e-5  # 归一化
    
        # 计算目标方向向量
        goal_direction = (self.end - state.vec[0:3]) / (np.linalg.norm(self.end - state.vec[0:3]) + 1e-5)
    
        # 计算动作与目标方向的对齐度
        alignment = np.dot(goal_direction, action_direction)
    
        # 动作变化量，用于鼓励动作平滑
        action_change = np.linalg.norm(action - self.previous_action)
    
        # 计算对齐和平滑的综合奖励
        smooth_alignment_reward = alignment - 0.01 * action_change
        # 更新上一动作
        return smooth_alignment_reward
        
    def _movement_reward(self, state, next_state, action):
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        distance_reward = 15 * (self.prev_distance_to_goal - distance_to_goal)
        energy_penalty = -0.1 #* (1 + self.step_cnt / 100) # 每一步都产生一个小的负奖励####pending
        static_penalty = -0.3 if np.linalg.norm(state.vec[0:3] - next_state.vec[0:3]) < 0.02 else 0
        return distance_reward  + energy_penalty  + static_penalty 
    
    def _obstacle_reward(self, next_state):
        # 找到距离最近的障碍物位置和最小距离
        distances_to_obstacles = [np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc]
        min_distance = np.min(distances_to_obstacles)
        closest_obstacle = self.obstacle_loc[np.argmin(distances_to_obstacles)]
        # 获取智能体速度向量
        velocity_vector = next_state.vec[3:6]
        # 计算智能体指向最近障碍物的方向向量并归一化
        direction_to_obstacle = (closest_obstacle - next_state.vec[0:3]) / (min_distance + 1e-5)
        # 计算指向最近障碍物的相对速度
        relative_speed = np.dot(velocity_vector, direction_to_obstacle)
        # 动态调整安全距离阈值，基于相对速度，值范围在1到5之间
        safe_distance_threshold = max(1, min(5, 3 + relative_speed * 0.1))
        # 计算障碍物惩罚
        if min_distance < safe_distance_threshold:
            obstacle_penalty = -5 * np.exp(-min_distance)
        else:
            obstacle_penalty = 0.05  # 超出安全距离时给予小的正奖励
        
        return obstacle_penalty
    
    # 出界类惩罚
    def _out_of_box_penalty(self, next_state): #   分级 考虑 碰撞 剐蹭。。。

        is_outside_box = (
            next_state.vec[0] < self.bottom_corner[0] or next_state.vec[0] > self.top_corner[0] or
            next_state.vec[1] < self.bottom_corner[1] or next_state.vec[1] > self.top_corner[1] or
            next_state.vec[2] < self.bottom_corner[2] or next_state.vec[2] > self.top_corner[2]
        )
        return -3 if is_outside_box else 0  # 2--3--1
    
    def _reach_target_reward(self, next_state):
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        if distance_to_goal < 10 and self.first_reach_close <= 1:
            self.first_reach_close = self.first_reach_close + 1 #. new try 20241029  
            
        if distance_to_goal < 1:
            reach_target_reward = 20000
        else:
            reach_target_reward = 0
            
        if distance_to_goal < 10 and self.first_reach_close == 1:
            first_close_reward = 2000 / (distance_to_goal + 1)
        else:
            first_close_reward = 0
        return reach_target_reward + first_close_reward

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

    def pad_state(self, state, target_dim=21):
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
