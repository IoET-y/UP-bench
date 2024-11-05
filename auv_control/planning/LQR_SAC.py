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
    def __init__(self, num_seconds, num_obstacles=20, start=None, end=None, state_dim=21, action_dim=8, lr=3e-4, gamma=0.95, tau=0.003, alpha=0.2, batch_size=512, replay_buffer_size=1000000): #bs up to 256
        # for dynamic adjust env reset
        self.episode_cnt = 0  # 设置混合策略切换的数值
        self.reach_targe_times = 0

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
                noise_scale = 0.1
            else:
                noise_scale = 0.05  # 靠近目标时，减少噪声
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            
        if self.previous_distance_to_goal > 10:
            action = np.clip(action * self.__MAX_THRUST, -self.__MAX_THRUST, self.__MAX_THRUST)
        elif self.previous_distance_to_goal > 5:
            action = np.clip(action * 0.5 * self.__MAX_THRUST, -0.5*self.__MAX_THRUST, 0.5*self.__MAX_THRUST)
        elif self.previous_distance_to_goal > 2:
            action = np.clip(action * 0.3 * self.__MAX_THRUST, -0.3*self.__MAX_THRUST, 0.3*self.__MAX_THRUST)
        else:
            action = np.clip(action * 0.15 * self.__MAX_THRUST, -0.15*self.__MAX_THRUST, 0.15*self.__MAX_THRUST)
        if self.static_cnt >5:
            action = 2*action
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
        # self.q_optimizer1.zero_grad()
        # q_loss1.backward()
        # self.q_optimizer1.step()

        # self.q_optimizer2.zero_grad()
        # q_loss2.backward()
        # self.q_optimizer2.step()

        # policy_loss = -(self.q_net1(torch.cat([states, self.policy_net(states)], dim=1))).mean()

        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()
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

        
        self.episode_cnt = 0
        wandb.init(project="auv_RL_control_project_SAC", name=model_path)
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

        for episode in range(num_episodes):
            self.static_on = False
            self.static_cnt = 0
            self.first_reach_close = 0
            self.epsd = episode
            logging.info(f"Episode {episode + 1} starting")
            # 混合策略：判断是否达到episode_threshold，如果达到，则使用上一个结束位置   20241029
            self.episode_cnt = self.episode_cnt + 1
            if self.done == True: #20241030 introduce dynamic start end point
                self.episode_cnt = 0
                self.static_cnt = 0
                self.setup_start()
                self.setup_end()
                self.setup_obstacles()
                
            if self.episode_cnt in range(10,15) and ((self.done == False) and self.prev_distance_to_goal<10):#20241030
                # 使用上一个 episode 的结束位置继续
                state = next_state
                print("auv position:",state.vec[0:3])
                distance_to_goal = np.linalg.norm(state.vec[0:3] - self.end)
                self.prev_distance_to_goal = distance_to_goal
            else:
                #self.episode_cnt = 0
                state_info = env.reset()
                env.agents["auv0"].teleport(self.start,[0,0,0])
                sensors = env.tick()
                state_info=sensors
                state = State(state_info)
                print("self start:",self.start)
                print("auv position:",state.vec[0:3])
                distance_to_goal = np.linalg.norm(self.start - self.end)
                self.prev_distance_to_goal = np.linalg.norm(self.start - self.end)
                

            self.done = False     
            done = False
            step_count = 0
            total_reward = 0
            self.distance_to_nearest_obstacle = np.min([np.linalg.norm(state.vec[0:3] - obs) for obs in self.obstacle_loc])
            while not done and step_count < max_steps:
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
                
                real_next_state = np.append(next_state.vec[0:], next_state.bias[0:])
                real_next_state = np.append(real_next_state, distance_to_goal)
                real_next_state = np.append(real_next_state, self.distance_to_nearest_obstacle)
                real_next_state = np.append(real_next_state, done)

                reward = self.calculate_reward(true_state, next_state, combined_action)
                total_reward += reward
   

                
                self.remember(real_state, combined_action, reward, real_next_state, done)
                self.update_policy()

                step_count += 1
                wandb.log({
                    "step_count": step_count,
                    "distance_to_nearest_obstacle": self.distance_to_nearest_obstacle,
                })
                
                if done:
                    wandb.log({
                        "episode": episode + 1,
                        "reach_targe_times": self.reach_targe_times + 1

                    })
                    
                    success_path = "successful_"+str(episode)+"_"+model_path
                    self.save_model(episode + 1, success_path)
                    break
                    
            wandb.log({
                "episode": episode + 1,
                "total_reward": total_reward,
                "distance to goal": distance_to_goal
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
        
        # add step penalty ......
        
        
        # Calculate the distance to the target
        distance_to_goal = np.linalg.norm(next_state.vec[0:3] - self.end)
        progress_reward = 15 * (self.prev_distance_to_goal - distance_to_goal) #20241029 10->20
        distance_reward = 1 * (1 / (distance_to_goal + 1))  # Logarithmic distance reward  1030 0.5->1
        
        if distance_to_goal < 10 and progress_reward > 0: # add 20241104
            distance_reward = (11-distance_to_goal)*distance_reward
        else:
            distance_reward = distance_reward
            
        # if self.static_on:
        #     distance_reward = 0
            
        # Penalties for proximity to obstacles
        self.distance_to_nearest_obstacle = np.min([np.linalg.norm(next_state.vec[0:3] - obs) for obs in self.obstacle_loc])
        obstacle_penalty = 0
        if self.distance_to_nearest_obstacle < 3: 
            obstacle_penalty = -5 * np.exp(-self.distance_to_nearest_obstacle)   #20241029 0.5 -> 0.1

        safety_reward = 0.3 if self.distance_to_nearest_obstacle > 2 else 0  # Reward for maintaining safe distance

        # Static penalty: Encourage movement by penalizing small displacements over multiple steps
        static_penalty = -0.5 if np.linalg.norm(state.vec[0:3] - next_state.vec[0:3]) < 0.01 else 0 #20241030
        if distance_to_goal < 2: #20241101 add
            obstacle_penalty = 0
            static_penalty = 0
        
        #last_static_penalty = -200*self.static_cnt if (self.static_cnt > 5 and self.static_on) else 0
        #last_static_penalty = -500*self.static_cnt if (self.static_cnt > 10 and self.static_on) else 0

        # Action smoothness penalty: Encourage smooth control inputs to avoid drastic changes in action
        action_smoothness_penalty = -0.02 * np.linalg.norm(action - self.previous_action)

        # Alignment reward: Encourage movement in the direction of the goal
        goal_direction = (self.end - state.vec[0:3]) / (np.linalg.norm(self.end - state.vec[0:3]) + 1e-5)
        velocity_direction = next_state.vec[3:6] / (np.linalg.norm(next_state.vec[3:6]) + 1e-5)
        alignment_reward = np.clip(1.0 * np.dot(goal_direction, velocity_direction), 0, 1)

        # Reach target reward: High reward when agent is very close to the target
        if distance_to_goal < 10 and self.first_reach_close <= 1:
            self.first_reach_close = self.first_reach_close + 1 #. new try 20241029
            
        reach_close_reward = 100 if self.first_reach_close == 1 else 0 #. new try 20241029
        reach_target_reward = 500 if distance_to_goal < 1.5 else 0

        # Incline penalty: Penalize for extreme pitch or roll angles
        roll, pitch = state.vec[6], state.vec[7]
        incline_penalty = 0.001 * (max(roll - 15, 0) + max(170 - pitch, 0))

        # Aggregate rewards and penalties
        total_reward = (
            progress_reward + distance_reward +
            obstacle_penalty + safety_reward +
            static_penalty + action_smoothness_penalty +
            alignment_reward + reach_target_reward +
            outside_box_penalty - incline_penalty + reach_close_reward #+ last_static_penalty
            )
        # Update previous distance to goal for future progress calculation
        self.prev_distance_to_goal = distance_to_goal

        self.previous_action = action
        return total_reward # normalized_reward # cancel normalize

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



    #parts below are utils function, can ignore them while training and inference.
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
