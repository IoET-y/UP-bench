
import os
import math
import time
import heapq
import random
import logging
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import scipy.linalg
from scipy.interpolate import splprep, splev
from collections import deque
import wandb
from .rl_utils import (calculate_ocean_current, calculate_action_effect,
                       normalize_action, denormalize_action)
from .rl_rewards import calculate_reward

# 假设 BasePlanner 定义在同目录下的 base 模块中
from .base import BasePlanner


# ---------------------------
# 定义辅助状态类，与老 SAC 算法一致
class EnvState:
    def __init__(self, location, rotation, velocity, lasers):
        # 拼接各传感器数据为一个向量
        self.vec = np.concatenate([location, rotation, velocity, lasers])


# ---------------------------
# 优先经验回放缓冲区，与老 SAC 算法一致
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, transition, td_error):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, alpha=0.6):
        if len(self.buffer) == 0:
            return [], [], []
        scaled_priorities = np.array(self.priorities) ** alpha
        probabilities = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def clear(self):
        self.buffer = []
        self.priorities = deque(maxlen=self.capacity)
        self.position = 0

    def __len__(self):
        return len(self.buffer)


# ---------------------------
# 策略网络，与老 SAC 算法一致
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.max_action = max_action  # 在此处代表局部子目标偏移的最大幅度（例如 10 米）
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        max_action_tensor = torch.tensor(self.max_action, device=y_t.device, dtype=y_t.dtype)
        action = y_t * max_action_tensor
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob


# ---------------------------
# Q网络，与老 SAC 算法一致
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# ---------------------------
# SAC+LQR 分层规划器
class SACLQRPlanner(BasePlanner):
    def __init__(self, num_seconds, state_dim=27, action_dim=3,
                 sensor_range=10.0, grid_resolution=0.5, max_steps = 2000, lr=2e-3, gamma=0.95, tau=0.02, batch_size=128,
                 replay_buffer_size=100000, config_file="./config_all.yaml"):
        """
        参数说明：
          - num_seconds: 总运行时间
          - state_dim: 用于策略输入的状态维度
          - action_dim: SAC agent 输出的局部目标偏移维度（3D）
          - lr, gamma, tau, batch_size, replay_buffer_size: SAC 相关超参数
          - config_file: 配置文件路径（用于设置环境种子等参数）
        """
        # 加载配置文件（若有）
        config_file = os.path.join(os.path.dirname(__file__), "config_all.yaml")
        with open(config_file, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        seed = self.config.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.max_steps = max_steps
        # SAC 超参数
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.per_alpha = 0.6
        self.memory = PrioritizedReplayBuffer(replay_buffer_size, alpha=self.per_alpha)
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        # 高层 SAC 目标：选择局部子目标，局部目标偏移范围设定为感知范围（10 米）
        self.max_local_offset = np.array([3.0, 3.0, 3.0])
        # 终点位置将在每个 episode 开始时从环境中获取
        self.end = None

        # SAC 网络及优化器
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_entropy = -np.prod(action_dim) * 1
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self._get_device())
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)
        self.alpha = self.log_alpha.exp().item()

        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.max_local_offset).to(self._get_device())
        self.q_net1 = QNetwork(self.state_dim, self.action_dim).to(self._get_device())
        self.q_net2 = QNetwork(self.state_dim, self.action_dim).to(self._get_device())
        self.target_q_net1 = QNetwork(self.state_dim, self.action_dim).to(self._get_device())
        self.target_q_net2 = QNetwork(self.state_dim, self.action_dim).to(self._get_device())

        self.policy_net.apply(self.initialize_weights)
        self.q_net1.apply(self.initialize_weights)
        self.q_net2.apply(self.initialize_weights)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)

        # 初始化 LQR 控制器参数（AUV 动力学采用双积分模型）
        self.ticks_per_sec = 100
        self.ts = 1.0 / self.ticks_per_sec
        I3 = np.eye(3)
        A = np.block([
            [np.eye(3), self.ts * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        B = np.block([
            [0.5 * (self.ts ** 2) * np.eye(3)],
            [self.ts * np.eye(3)]
        ])
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1, 0.1])
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # 规划网格参数（用于 A*），这里参考 AStarPlanner 的设置
        self.grid_resolution = grid_resolution  # 可以调小以适应局部范围
        self.x_min = 0
        self.x_max = 300
        self.y_min = 0
        self.y_max = 300
        self.z_min = -100
        self.z_max = 0
        self.obstacle_radius = 5  # 障碍“膨胀”半径

        # 感知范围（局部规划时只考虑 10 米内的障碍物）
        self.sensor_range = sensor_range

        # 记录评价指标（路径长度、碰撞数、能耗、平滑度等）
        self.episode_out_of_box_penalty = 0.0
        self.episode_energy_penalty = 0.0
        self.episode_smoothness_penalty = 0.0
        self.episode_time_penalty = 0.0
        self.reach_targe_times = 0
        self.episode_distance_reward = 0
        self.episode_align_reward = 0
        self.episode_current_utilization_reward = 0
        self.episode_safety_reward = 0
        self.episode_reach_target_reward = 0

        self.total_length = 0
        self.episode_path_length = 0
        self.episode_collisions = 0
        self.episode_energy = 0
        self.episode_smoothness = 0
        self.static_counter = 0

        # 其他状态记录
        self.previous_action = np.zeros(action_dim)
        self.current_time = 0.0

        super().__init__()

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'out_features') and m.out_features == self.action_dim:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ---------------------------
    # 以下为 A* 规划相关函数（与 AStarPlanner 类似）
    def world_to_index(self, pos):
        ix = int((pos[0] - self.x_min) / self.grid_resolution)
        iy = int((pos[1] - self.y_min) / self.grid_resolution)
        iz = int((pos[2] - self.z_min) / self.grid_resolution)
        nx = int((self.x_max - self.x_min) / self.grid_resolution)
        ny = int((self.y_max - self.y_min) / self.grid_resolution)
        nz = int((self.z_max - self.z_min) / self.grid_resolution)
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        iz = min(max(iz, 0), nz - 1)
        return (ix, iy, iz)

    def index_to_world(self, idx):
        x = self.x_min + idx[0] * self.grid_resolution + self.grid_resolution / 2.0
        y = self.y_min + idx[1] * self.grid_resolution + self.grid_resolution / 2.0
        z = self.z_min + idx[2] * self.grid_resolution + self.grid_resolution / 2.0
        return np.array([x, y, z])

    def create_obstacle_grid(self, obstacles):
        nx = int((self.x_max - self.x_min) / self.grid_resolution)
        ny = int((self.y_max - self.y_min) / self.grid_resolution)
        nz = int((self.z_max - self.z_min) / self.grid_resolution)
        grid = np.zeros((nx, ny, nz), dtype=int)
        for obs in obstacles:
            # 仅对在规划区域内的障碍物标记
            if not (self.x_min <= obs[0] <= self.x_max and
                    self.y_min <= obs[1] <= self.y_max and
                    self.z_min <= obs[2] <= self.z_max):
                continue
            obs_idx = self.world_to_index(obs)
            radius_in_cells = int(math.ceil(self.obstacle_radius / self.grid_resolution))
            for i in range(max(0, obs_idx[0] - radius_in_cells), min(nx, obs_idx[0] + radius_in_cells + 1)):
                for j in range(max(0, obs_idx[1] - radius_in_cells), min(ny, obs_idx[1] + radius_in_cells + 1)):
                    for k in range(max(0, obs_idx[2] - radius_in_cells), min(nz, obs_idx[2] + radius_in_cells + 1)):
                        cell_center = self.index_to_world((i, j, k))
                        if np.linalg.norm(cell_center - np.array(obs)) <= self.obstacle_radius:
                            grid[i, j, k] = 1
        return grid

    class Node:
        def __init__(self, x, y, z, g=0, h=0, parent=None):
            self.x = x
            self.y = y
            self.z = z
            self.g = g
            self.h = h
            self.f = g + h
            self.parent = parent

        def __lt__(self, other):
            return self.f < other.f

    def plan_path(self, start, goal, obstacles):
        """
        使用 A* 算法规划路径
        """
        grid = self.create_obstacle_grid(obstacles)
        nx, ny, nz = grid.shape
        start_idx = self.world_to_index(start)
        goal_idx = self.world_to_index(goal)
        open_list = []
        closed_set = set()
        h_start = math.sqrt((goal_idx[0]-start_idx[0])**2 +
                            (goal_idx[1]-start_idx[1])**2 +
                            (goal_idx[2]-start_idx[2])**2)
        start_node = self.Node(start_idx[0], start_idx[1], start_idx[2], g=0, h=h_start, parent=None)
        heapq.heappush(open_list, start_node)
        node_map = {(start_idx[0], start_idx[1], start_idx[2]): start_node}
        found = False
        while open_list:
            current = heapq.heappop(open_list)
            if (current.x, current.y, current.z) == goal_idx:
                found = True
                goal_node = current
                break
            closed_set.add((current.x, current.y, current.z))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nx_idx = current.x + dx
                        ny_idx = current.y + dy
                        nz_idx = current.z + dz
                        if nx_idx < 0 or nx_idx >= nx or ny_idx < 0 or ny_idx >= ny or nz_idx < 0 or nz_idx >= nz:
                            continue
                        if grid[nx_idx, ny_idx, nz_idx] == 1:
                            continue
                        neighbor_index = (nx_idx, ny_idx, nz_idx)
                        if neighbor_index in closed_set:
                            continue
                        move_cost = math.sqrt(dx**2 + dy**2 + dz**2)
                        g_new = current.g + move_cost
                        h_new = math.sqrt((goal_idx[0]-nx_idx)**2 +
                                          (goal_idx[1]-ny_idx)**2 +
                                          (goal_idx[2]-nz_idx)**2)
                        f_new = g_new + h_new
                        if neighbor_index in node_map:
                            if g_new < node_map[neighbor_index].g:
                                node_map[neighbor_index].g = g_new
                                node_map[neighbor_index].f = f_new
                                node_map[neighbor_index].parent = current
                        else:
                            neighbor_node = self.Node(nx_idx, ny_idx, nz_idx, g=g_new, h=h_new, parent=current)
                            node_map[neighbor_index] = neighbor_node
                            heapq.heappush(open_list, neighbor_node)
        if not found:
            return None
        # 回溯路径
        path_indices = []
        node = goal_node
        while node is not None:
            path_indices.append((node.x, node.y, node.z))
            node = node.parent
        path_indices.reverse()
        path = [self.index_to_world(idx) for idx in path_indices]
        return path
    #
    # def smooth_path(self, path, smoothing_factor=2.0, num_points=100):
    #     """
    #     对离散路径进行样条平滑
    #     """
    #     if len(path) < 3:
    #         return path
    #
    #     path_array = np.array(path).T  # shape (3, n)
    #     tck, u = splprep(path_array, s=smoothing_factor)
    #     u_new = np.linspace(0, 1, num_points)
    #     smooth_points = splev(u_new, tck)
    #     smooth_path = np.vstack(smooth_points).T
    #     return [pt for pt in smooth_path]
    def smooth_path(self, path, smoothing_factor=1.0, num_points=100):
        """
        对路径进行样条插值平滑
        """
        if len(path) < 4:  # 只有 3 个点以下时，不进行平滑
            return path

        path_array = np.array(path).T  # shape: (3, n)
        try:
            tck, u = splprep(path_array, s=smoothing_factor, k=min(3, len(path) - 1))  # 确保 k < len(path)
        except TypeError as e:
            print(f"[Warning] Spline fitting failed: {e}")
            return path  # 返回原路径

        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]
    # ---------------------------
    # LQR 控制器：根据当前状态和期望状态（当前位置及期望速度对应于当前规划路径的目标点），计算控制输入
    def lqr_control(self, current_pos, current_velocity, waypoint, desired_velocity):
        # 构造状态向量：x = [position, velocity]
        x_current = np.hstack([current_pos, current_velocity])
        x_des = np.hstack([waypoint, desired_velocity])
        error_state = x_current - x_des
        u = -self.K.dot(error_state)
        # 限制控制输入
        u = np.clip(u, -20, 20)  # 这里使用 20 为最大线性加速度，可根据需要调整
        return u

    # ---------------------------
    # 局部路径跟踪：利用 LQR 控制器沿规划好的局部路径跟踪，直到接近局部目标或超步长
    def follow_local_path(self, env, path, local_goal, max_steps=50):
        """
        参数：
          - env: 环境对象
          - path: 局部平滑路径（列表，每个为 np.array([x,y,z])）
          - local_goal: 本次局部规划的目标点（绝对坐标）
          - max_steps: 本次局部控制最大步数
        返回：
          - final_state: 跟踪结束后的 AUV 位置
          - total_reward: 累计奖励
          - steps_taken: 实际控制步数
        """
        step_count = 0
        total_reward = 0.0
        prev_u = None
        # 获取当前状态
        current_pos = env.location.copy()
        current_velocity = env.velocity.copy()
        path_idx = 0
        while step_count < max_steps:
            # 若已到达局部目标（距离小于 2 米），则结束
            if np.linalg.norm(current_pos - local_goal) < 1:
                break
            # 若路径规划完毕，则以局部目标作为期望点
            if path_idx >= len(path):
                waypoint = local_goal
                v_des = np.zeros(3)
            else:
                waypoint = path[path_idx]
                # 设置期望速度（例如 3 m/s）沿路径方向
                if path_idx < len(path) - 1:
                    desired_speed = 2.0
                    direction = path[path_idx+1] - waypoint
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-6:
                        direction = direction / norm_dir
                    else:
                        direction = np.zeros(3)
                    v_des = desired_speed * direction
                else:
                    v_des = np.zeros(3)
                # 当接近当前路径点时，切换下一个
                if np.linalg.norm(current_pos - waypoint) < 1:
                    path_idx += 1
                    continue

            # 计算 LQR 控制输入
            u = self.lqr_control(current_pos, current_velocity, waypoint, v_des)
            # 构造完整动作：前三个为线性加速度，后三个角加速度置 0
            action = np.concatenate([u, np.zeros(3)])
            sensors = env.tick(action)
            env.update_state(sensors)
            new_pos = env.location.copy()
            next_dist_to_goal = np.linalg.norm(self.end - new_pos)

            # 计算奖励（调用下文的 calculate_reward 函数）
            state_vec = np.hstack([current_pos, env.rotation, current_velocity, env.lasers])
            next_state_vec = np.hstack([new_pos, env.rotation, env.velocity, env.lasers])
            r, pr, ar, sp, bonus, sm_pen = calculate_reward(self,state_vec, next_state_vec, u)
            total_reward += r
            # 更新评价指标
            distance_moved = np.linalg.norm(new_pos - current_pos)

            if next_dist_to_goal < 2:
                self.reach_targe_times += 1
                self.done = 1
                break

            # 碰撞检测：若距离任一障碍物小于 5 米，则记一次碰撞
            for obs in env.obstacles:
                if np.linalg.norm(new_pos - np.array(obs)) < 2.0:
                    break

            current_pos = new_pos
            current_velocity = env.velocity.copy()
            step_count += 1
            self.current_time += self.ts
        return current_pos, total_reward, step_count


    # ---------------------------
    # SAC 相关方法
    def select_action(self, state, inference=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._get_device())
        with torch.no_grad():
            normalized_action, _ = self.policy_net.sample(state_tensor)
        normalized_action = normalized_action.cpu().numpy()[0]
        # 保证局部偏移在 [-max_local_offset, max_local_offset] 内
        action = np.clip(normalized_action, -self.max_local_offset, self.max_local_offset)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done), td_error=0.0)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        update_times = 1
        for _ in range(update_times):
            samples, weights, indices = self.memory.sample(self.batch_size, beta=self.beta, alpha=self.per_alpha)
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
            states, actions, rewards, next_states, dones = zip(*samples)
            states = torch.FloatTensor(states).to(self._get_device())
            actions = torch.FloatTensor(actions).to(self._get_device())
            rewards = torch.FloatTensor(rewards).to(self._get_device()).unsqueeze(1)
            next_states = torch.FloatTensor(next_states).to(self._get_device())
            dones = torch.FloatTensor(dones).to(self._get_device()).unsqueeze(1)
            weights = torch.FloatTensor(weights).to(self._get_device()).unsqueeze(1)
            with torch.no_grad():
                next_actions, next_log_pis = self.policy_net.sample(next_states)
                next_q1 = self.target_q_net1(next_states, next_actions)
                next_q2 = self.target_q_net2(next_states, next_actions)
                next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pis
                target_q = rewards + self.gamma * (1 - dones) * next_q
            current_q1 = self.q_net1(states, actions)
            current_q2 = self.q_net2(states, actions)
            td_errors = (target_q - current_q1).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors.flatten())
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
            new_actions, log_pis = self.policy_net.sample(states)
            q1_new_actions = self.q_net1(states, new_actions)
            q2_new_actions = self.q_net2(states, new_actions)
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)
            policy_loss = (weights * (self.alpha * log_pis - q_new_actions)).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.policy_optimizer.step()
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        # 更新目标网络
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        wandb.log({
            "q_loss1": q_loss1.item(),
            "q_loss2": q_loss2.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha_value": self.alpha,
        })

    def save_model(self, episode, model_path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'q_net1': self.q_net1.state_dict(),
            'q_net2': self.q_net2.state_dict(),
            'target_q_net1': self.target_q_net1.state_dict(),
            'target_q_net2': self.target_q_net2.state_dict(),
            'optimizer_policy': self.policy_optimizer.state_dict(),
            'optimizer_q1': self.q_optimizer1.state_dict(),
            'optimizer_q2': self.q_optimizer2.state_dict(),
        }, model_path)

    # ---------------------------
    # 主训练流程：高层 SAC 与低层 LQR 控制协同工作
    def train(self, env, num_episodes=500, max_macro_steps= 512, model_path="sac_lqr_best_model.pth"):
        """
        训练流程：
          1. 重置环境，获取初始状态（起点及终点）
          2. 循环：高层 SAC 根据当前状态选择局部子目标（局部目标 = 当前位置信息 + 输出偏移，偏移范围受限于感知范围 10 米）
          3. 低层：利用 A*（仅考虑感知范围内障碍）规划局部路径，然后使用 LQR 控制器跟踪局部路径，累计步级奖励
          4. 将该“宏步”作为一次高层 transition 存入经验池，并更新 SAC 网络
          5. 重复以上过程直至 AUV 到达终点（目标距离小于 2 米）
        """
        wandb.init(project="auv_SAC_LQR_planning", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "tau": self.tau,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "sensor_range": self.sensor_range,
            "max_local_offset": self.max_local_offset.tolist(),
        })
        episode = 0
        self.done = 0
        while self.reach_targe_times < 10:
            episode_start_time = time.time()
            logging.info(f"Episode {episode+1} starting")
            env.reset()
            if self.done == 1:
                env.set_current_target(env.choose_next_target())
                env.draw_targets()
                self.done = 0

            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            current_pos = env.location.copy()
            current_velocity = env.velocity.copy()
            # 获取终点（全局目标）
            self.end = np.array(env.get_current_target())
            logging.info(f"Start: {current_pos}, Goal: {self.end}")

            macro_step = 0
            episode_reward = 0.0

            # 重置本 episode 评价指标
            self.episode_out_of_box_penalty = 0.0
            self.episode_energy_penalty = 0.0
            self.episode_smoothness_penalty = 0.0
            self.episode_time_penalty = 0.0
            self.episode_distance_reward = 0
            self.episode_align_reward = 0
            self.episode_current_utilization_reward = 0
            self.episode_safety_reward = 0
            self.episode_reach_target_reward = 0
            self.total_length = 0
            self.episode_path_length = 0
            self.episode_collisions = 0
            self.episode_energy = 0
            self.episode_smoothness = 0
            self.static_counter = 0

            while macro_step < max_macro_steps:

                # 构造当前状态，用于高层 SAC（状态包含：位置、rotation、速度、激光传感器数据，加上海流信息和全局目标）
                # 这里简化处理，直接将各传感器数据拼接
                pre_state = EnvState(current_pos, env.rotation.copy(), current_velocity, env.lasers.copy())
                # 假设海流信息置0（或可调用相应函数），这里简化
                pre_state_vec = np.append(pre_state.vec, [0.0, *self.end])
                # SAC agent 选择局部偏移（局部子目标 = current_pos + offset）；注意：偏移必须在感知范围内
                local_offset = self.select_action(pre_state_vec)
                # 限制偏移范数不超过 sensor_range
                if np.linalg.norm(local_offset) > self.sensor_range:
                    local_offset = local_offset / np.linalg.norm(local_offset) * self.sensor_range
                local_goal = current_pos + local_offset
                logging.info(f"Macro step {macro_step+1}: local goal = {local_goal}")

                # 规划局部路径：仅使用感知范围内的障碍物
                local_obstacles = []
                for obs in env.obstacles:
                    if np.linalg.norm(np.array(obs) - current_pos) <= self.sensor_range:
                        local_obstacles.append(obs)
                path = self.plan_path(current_pos, local_goal, local_obstacles)
                if path is None:
                    logging.info("Local path planning failed. Penalizing and break.")
                    reward_penalty = -100
                    episode_reward += reward_penalty

                    self.remember(pre_state_vec, local_offset, reward_penalty, pre_state_vec, 1)
                    break

                # 平滑局部路径
                path = self.smooth_path(path, smoothing_factor=1.0, num_points=100)
                # （可选）在环境中绘制局部路径
                for i in range(len(path) - 1):
                    env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[0,100,0], thickness=3, lifetime=0)

                # 低层跟踪局部路径，返回新状态、累计奖励、步数
                new_pos, macro_reward, steps_taken = self.follow_local_path(env, path, local_goal, max_steps=100)


                episode_reward += macro_reward
                if episode_reward < -1000 or self.episode_out_of_box_penalty <-1000:
                    break
                macro_step += 1

                # 构造下一个状态
                post_state = EnvState(new_pos, env.rotation.copy(), env.velocity.copy(), env.lasers.copy())
                post_state_vec = np.append(post_state.vec, [0.0, *self.end])
                # 存储高层 transition（用累计奖励作为本宏步奖励）
                self.remember(pre_state_vec, local_offset, macro_reward, post_state_vec, self.done)
                # 更新 SAC agent
                self.update_policy()
                current_pos = new_pos.copy()
                current_velocity = env.velocity.copy()
                if self.done == 1:
                    break

                # 日志记录
                wandb.log({
                    "episode": episode+1,
                    "macro_step": macro_step,
                    "current_x": current_pos[0],
                    "current_y": current_pos[1],
                    "current_z": current_pos[2],
                    "macro_reward": macro_reward,
                    "steps_in_macro": steps_taken,
                    "distance_to_global_goal": np.linalg.norm(current_pos - self.end),
                })
            episode_duration = time.time() - episode_start_time
            wandb.log({
                "episode": episode + 1,
                "eps_reach_target": self.reach_targe_times,
                "eps_total_reward": episode_reward,
                "eps_align_reward": self.episode_align_reward,
                "eps_safety_reward": self.episode_safety_reward,
                "eps_reach_target_reward": self.episode_reach_target_reward,
                "eps_out_of_box_penalty": self.episode_out_of_box_penalty,
                "eps_energy_penalty ": self.episode_energy_penalty,
                "eps_smoothness_penalty": self.episode_smoothness_penalty,
                "eps_time_penalty": self.episode_time_penalty,
                "eps_ave_length_per_step": self.episode_path_length / macro_step if macro_step > 0 else 0,
                "episode_path_length": self.episode_path_length,
                "episode_collisions": self.episode_collisions,
                "episode_energy": self.episode_energy,
                "episode_smoothness": self.episode_smoothness,
                "episode_duration": episode_duration
            })
            logging.info(f"Episode {episode+1} completed - Total Reward: {episode_reward}")
            self.save_model(episode+1, model_path)
            episode += 1

        logging.info("SAC+LQR Planning finished training.")
        return
