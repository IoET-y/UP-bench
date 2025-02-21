#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sac_lqr_planner.py

该代码实现了一个基于SAC高层决策与A*+LQR低层规划控制的分层路径规划方法，
用于海底环境中在线规划：AUV仅感知10米内障碍物信息和终点位置，
利用SAC输出局部子目标，再用A*规划局部路径、LQR跟踪路径并获得reward，
迭代直到全局终点到达。
"""

import os
import time
import math
import yaml
import random
import heapq
import logging
import numpy as np
import scipy.linalg
from scipy.interpolate import splprep, splev

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

import wandb

# 假设 BasePlanner 在同目录下的 base 模块中定义
from .base import BasePlanner

###########################
# 优先经验回放缓冲区
###########################
from collections import deque

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

###########################
# SAC 策略网络和Q网络
###########################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.max_action = max_action
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

###########################
# SACLQRPlanner
###########################

class SACLQRPlanner(BasePlanner):
    def __init__(self, num_seconds, state_dim=29, action_dim=3,
                 sensor_range=10.0, grid_resolution=0.5, max_steps=2000,
                 lr=2e-3, gamma=0.95, tau=0.02, batch_size=128,
                 replay_buffer_size=100000, collision_threshold=5.0,
                 config_file="./config_all.yaml"):
        """
        参数说明：
          - num_seconds: 总运行时间
          - state_dim: SAC 状态维度（可根据实际情况调整）
          - action_dim: SAC 高层动作维度，本例为 3（表示局部子目标相对偏移）
          - sensor_range: AUV感知范围（单位米，本例10米）
          - grid_resolution: 局部A*规划的网格分辨率
          - max_steps: 每个episode最大步数
          - collision_threshold: 用于检测碰撞的距离阈值
        """
        self.num_seconds = num_seconds
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sensor_range = sensor_range
        self.grid_resolution = grid_resolution
        self.max_steps = max_steps
        self.collision_threshold = collision_threshold

        # SAC相关超参数
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size

        # SAC 目标熵与温度参数
        self.target_entropy = -np.prod(action_dim) * 1
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self._get_device())
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)
        self.alpha = self.log_alpha.exp().item()

        self.replay_buffer_size = replay_buffer_size
        self.memory = PrioritizedReplayBuffer(replay_buffer_size, alpha=0.6)
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        # 低层LQR控制器设计（假设AUV动力学为双积分模型）
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

        # 用于A*局部规划的一些参数：
        self.obstacle_radius = 1.0  # 局部规划中给障碍适当“膨胀”

        # 全局规划区域信息（可按需要调整，此处仅作为参考）
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0

        # SAC网络及优化器
        self.max_action = np.array([self.sensor_range] * 3)  # 子目标最大偏移不超过感知范围
        self.max_action_normalize = np.array([1] * 3)

        self.device = self._get_device()
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.max_action_normalize).to(self.device)
        self.q_net1 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q_net2 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net1 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net2 = QNetwork(self.state_dim, self.action_dim).to(self.device)

        self.policy_net.apply(self.initialize_weights)
        self.q_net1.apply(self.initialize_weights)
        self.q_net2.apply(self.initialize_weights)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)

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

    ###########################
    # A* 局部规划相关函数
    ###########################
    class Node:
        def __init__(self, x, y, z, g=0, h=0, parent=None):
            self.x = x
            self.y = y
            self.z = z
            self.g = g      # 起点到当前节点的代价
            self.h = h      # 启发式代价（欧氏距离）
            self.f = g + h  # 总代价
            self.parent = parent

        def __lt__(self, other):
            return self.f < other.f

    def world_to_index(self, pos, region_min, resolution, grid_size):
        # 将连续坐标转为局部网格索引
        ix = int((pos[0] - region_min[0]) / resolution)
        iy = int((pos[1] - region_min[1]) / resolution)
        iz = int((pos[2] - region_min[2]) / resolution)
        ix = min(max(ix, 0), grid_size[0]-1)
        iy = min(max(iy, 0), grid_size[1]-1)
        iz = min(max(iz, 0), grid_size[2]-1)
        return (ix, iy, iz)

    def index_to_world(self, idx, region_min, resolution):
        # 将局部网格索引转为世界坐标（单元中心）
        x = region_min[0] + idx[0] * resolution + resolution / 2.0
        y = region_min[1] + idx[1] * resolution + resolution / 2.0
        z = region_min[2] + idx[2] * resolution + resolution / 2.0
        return np.array([x, y, z])

    def create_local_obstacle_grid(self, current_pos, obstacles):
        """
        构造以current_pos为中心、边长为 2*sensor_range 的立方体局部网格，
        标记障碍物（障碍物在局部范围内并膨胀 obstacle_radius）
        """
        region_min = current_pos - self.sensor_range
        region_max = current_pos + self.sensor_range
        grid_size = (
            int((2 * self.sensor_range) / self.grid_resolution),
            int((2 * self.sensor_range) / self.grid_resolution),
            int((2 * self.sensor_range) / self.grid_resolution)
        )
        grid = np.zeros(grid_size, dtype=int)

        for obs in obstacles:
            # 只处理在局部区域内的障碍物
            if np.all(obs >= region_min) and np.all(obs <= region_max):
                obs_idx = self.world_to_index(obs, region_min, self.grid_resolution, grid_size)
                radius_in_cells = int(math.ceil(self.obstacle_radius / self.grid_resolution))
                for i in range(max(0, obs_idx[0] - radius_in_cells), min(grid_size[0], obs_idx[0] + radius_in_cells + 1)):
                    for j in range(max(0, obs_idx[1] - radius_in_cells), min(grid_size[1], obs_idx[1] + radius_in_cells + 1)):
                        for k in range(max(0, obs_idx[2] - radius_in_cells), min(grid_size[2], obs_idx[2] + radius_in_cells + 1)):
                            cell_center = self.index_to_world((i, j, k), region_min, self.grid_resolution)
                            if np.linalg.norm(cell_center - obs) <= self.obstacle_radius:
                                grid[i, j, k] = 1
        return grid, region_min, grid_size

    def plan_local_path(self, start, goal, obstacles):
        """
        利用A*规划从start到goal的局部路径（规划区域为以start为中心、边长2*sensor_range的区域）
        obstacles：局部障碍物（已经是世界坐标）
        返回：路径为连续世界坐标列表
        """
        grid, region_min, grid_size = self.create_local_obstacle_grid(start, obstacles)
        nx, ny, nz = grid_size

        start_idx = self.world_to_index(start, region_min, self.grid_resolution, grid_size)
        goal_idx = self.world_to_index(goal, region_min, self.grid_resolution, grid_size)

        open_list = []
        closed_set = set()
        h_start = math.sqrt((goal_idx[0]-start_idx[0])**2 + (goal_idx[1]-start_idx[1])**2 + (goal_idx[2]-start_idx[2])**2)
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
                        h_new = math.sqrt((goal_idx[0]-nx_idx)**2 + (goal_idx[1]-ny_idx)**2 + (goal_idx[2]-nz_idx)**2)
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

        # 回溯构造路径
        path_indices = []
        node = goal_node
        while node is not None:
            path_indices.append((node.x, node.y, node.z))
            node = node.parent
        path_indices.reverse()

        # 转换为世界坐标
        path = []
        for idx in path_indices:
            pos = self.index_to_world(idx, region_min, self.grid_resolution)
            path.append(pos)
        return path

    def smooth_path(self, path, smoothing_factor=1.0, num_points=100):
        """
        对路径进行样条平滑
        """
        if len(path) < 3:
            return path
        path_array = np.array(path).T  # shape (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    ###########################
    # 低层 LQR 控制段执行（沿局部规划路径跟踪子目标）
    ###########################
    def lqr_control_segment(self, env, path, sub_goal, max_segment_steps=200):
        """
        沿给定的局部平滑路径，用LQR控制器跟踪直到到达子目标或超步。
        返回：最终状态（env.location）、累积reward、done标志
        """
        step_count = 0
        total_reward = 0.0
        path_idx = 0
        current_pos = env.location.copy()

        while step_count < max_segment_steps:
            # 若全局目标已到达，则退出
            if np.linalg.norm(current_pos - np.array(env.get_current_target())) < 2:
                return current_pos, total_reward, True

            # 若到达子目标（局部规划终点），退出本段控制
            if np.linalg.norm(current_pos - sub_goal) < 1.0:
                return current_pos, total_reward, False

            # 确定当前目标点（沿路径）
            if path_idx >= len(path):
                waypoint = sub_goal
                v_des = np.zeros(3)
            else:
                waypoint = path[path_idx]
                # 计算期望速度（朝向下一个点，设定目标速度）
                if path_idx < len(path) - 1:
                    desired_speed = 2.0  # 可调整
                    direction = path[path_idx+1] - waypoint
                    norm_dir = np.linalg.norm(direction)
                    direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                    v_des = desired_speed * direction
                else:
                    v_des = np.zeros(3)
                if np.linalg.norm(current_pos - waypoint) < 0.5:
                    path_idx += 1
                    continue

            # 构造状态
            x_current = np.hstack([current_pos, env.velocity.copy()])
            x_des = np.hstack([waypoint, v_des])
            error_state = x_current - x_des
            # LQR 控制
            u = -self.K.dot(error_state)
            u = np.clip(u, -20.0, 20.0)  # 限制控制输入幅值
            action = np.concatenate([u, np.zeros(3)])  # 后三维角加速度置0

            sensors = env.tick(action)
            env.update_state(sensors)
            new_pos = env.location.copy()

            # 简单的reward设计：进展奖励 - 能量惩罚 - 碰撞惩罚
            progress_reward = np.linalg.norm(current_pos - sub_goal) - np.linalg.norm(new_pos - sub_goal)
            energy_penalty = 0.01 * np.linalg.norm(u)**2
            collision_penalty = 0.0
            for obs in env.obstacles:
                if np.linalg.norm(new_pos - np.array(obs)) < self.collision_threshold:
                    collision_penalty = -5.0
                    break
            step_reward = progress_reward - energy_penalty + collision_penalty

            total_reward += step_reward
            current_pos = new_pos
            step_count += 1

        return current_pos, total_reward, False

    ###########################
    # SAC 相关函数
    ###########################
    def select_action(self, state, inference=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state_tensor)
        action = action.cpu().numpy()[0]
        # 限制在 [-1,1]（归一化后），由此映射到实际子目标偏移范围
        action = np.clip(action, -self.max_action_normalize, self.max_action_normalize)
        # 实际子目标偏移 = action * sensor_range
        return action * self.sensor_range

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done), td_error=0.0)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        update_times = 1
        for _ in range(update_times):
            samples, weights, indices = self.memory.sample(self.batch_size, beta=self.beta)
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
            states, actions, rewards, next_states, dones = zip(*samples)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
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

            wandb.log({
                "q_loss1": q_loss1.item(),
                "q_loss2": q_loss2.item(),
                "policy_loss": policy_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha_value": self.alpha,
            })

        # 更新目标网络
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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

    ###########################
    # 整体训练流程
    ###########################
    def train(self, env, num_episodes=500, max_steps_episode=3000, model_path="sac_lqr_best_model.pth"):
        """
        整个训练流程：
          1. 重置环境并获得初始状态（使用传感器数据构造状态向量，例如：位置、速度、局部障碍物信息、全局目标）
          2. 高层 SAC 根据当前状态选择局部子目标（相对偏移，映射到绝对子目标）
          3. 根据当前传感器（10米内）获得障碍物列表，利用 A* 局部规划生成路径，并平滑
          4. 使用 LQR 控制器沿平滑路径执行，直至到达子目标（或遇到特殊情况）
          5. 累计该段 reward，将 transition 存入经验池，更新 SAC 策略
          6. 重复直到全局目标到达
        """
        wandb.init(project="auv_SAC_LQR_planning", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "sensor_range": self.sensor_range,
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "tau": self.tau,
        })

        episode = 0
        total_episodes_reward = 0

        while episode < num_episodes:
            episode_start_time = time.time()
            logging.info(f"Episode {episode+1} starting")
            env.reset()
            # 用零动作获得初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            current_state = self._construct_state(env)
            total_episode_reward = 0.0
            step_count = 0

            # 只要全局目标未到达，就进行局部规划与控制
            while step_count < max_steps_episode:
                current_pos = env.location.copy()
                global_goal = np.array(env.get_current_target())
                # 如果全局目标在感知范围内，则直接将子目标设为全局目标
                if np.linalg.norm(current_pos - global_goal) <= self.sensor_range:
                    sub_goal = global_goal
                else:
                    # SAC输出局部子目标偏移（绝对子目标 = 当前坐标 + 子目标偏移）
                    action_offset = self.select_action(current_state)
                    sub_goal = current_pos + action_offset

                # 从传感器获得的障碍物：只选取10米内的障碍物（假设 env.obstacles 为全局障碍列表）
                obstacles_local = [np.array(obs) for obs in env.obstacles if np.linalg.norm(np.array(obs) - current_pos) <= self.sensor_range]

                # 利用A*规划局部路径
                path = self.plan_local_path(current_pos, sub_goal, obstacles_local)
                if path is None:
                    # 无法规划路径，给予较大负奖励后退出本段
                    segment_reward = -20.0
                    next_state = self._construct_state(env)
                    done = False
                    logging.info("Local path planning failed, applying penalty.")
                    self.remember(current_state, action_offset, segment_reward, next_state, done)
                    total_episode_reward += segment_reward
                    break

                # 平滑路径
                smooth_path = self.smooth_path(path, smoothing_factor=1.0, num_points=100)

                # 可选：在环境中绘制规划路径
                for i in range(len(smooth_path)-1):
                    env.env.draw_line(smooth_path[i].tolist(), smooth_path[i+1].tolist(), color=[30,50,0], thickness=3, lifetime=0)

                # 使用LQR沿局部路径执行
                new_pos, segment_reward, reached_goal = self.lqr_control_segment(env, smooth_path, sub_goal, max_segment_steps=200)
                total_episode_reward += segment_reward
                step_count += 1  # 这里简单计步，可按实际步数累加

                next_state = self._construct_state(env)
                done = True if np.linalg.norm(new_pos - global_goal) < 2.0 else False

                # 存储高层 transition，并更新策略
                self.remember(current_state, action_offset, segment_reward, next_state, done)
                self.update_policy()

                wandb.log({
                    "episode": episode+1,
                    "step_count": step_count,
                    "x_pos": new_pos[0],
                    "y_pos": new_pos[1],
                    "z_pos": new_pos[2],
                    "segment_reward": segment_reward,
                    "global_distance": np.linalg.norm(new_pos - global_goal)
                })

                current_state = next_state

                if done:
                    logging.info("Global goal reached!")
                    break

            episode_duration = time.time() - episode_start_time
            wandb.log({
                "episode": episode+1,
                "total_episode_reward": total_episode_reward,
                "episode_duration": episode_duration,
                "steps_in_episode": step_count
            })
            logging.info(f"Episode {episode+1} completed - Total Reward: {total_episode_reward}, Steps: {step_count}")
            self.save_model(episode+1, model_path)
            episode += 1

        logging.info("Training finished.")
        return

    def _construct_state(self, env):
        """
        构造 SAC 输入状态。可以包含：当前位置、速度、传感器（障碍物）信息、全局目标位置等。
        这里简单拼接：位置、速度、以及全局目标相对位置
        """
        pos = env.location.copy()
        vel = env.velocity.copy()
        global_goal = np.array(env.get_current_target())
        rel_goal = global_goal - pos
        # 如果有激光或其他传感器数据也可拼接（需保证维度与self.state_dim匹配）
        state = np.concatenate([pos, vel, rel_goal])
        # 若维度不足，可用零填充
        if len(state) < self.state_dim:
            state = np.concatenate([state, np.zeros(self.state_dim - len(state))])
        return state

