# ppo_planner.py

import os
import time
import yaml
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from collections import deque
import wandb

# 假设 BasePlanner 在同目录下的 base 模块中定义
from .base import BasePlanner

# 从工具模块中导入海流、动作相关工具函数和奖励函数
from .rl_utils import (calculate_ocean_current, calculate_action_effect,
                       normalize_action, denormalize_action)
from .rl_rewards_PPO import calculate_reward

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# 工具函数：反 tanh
#########################################
def atanh(x):
    # 为保证数值稳定，将输入 clamp 到 (-1+ε,1-ε)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


#########################################
# 定义 PPO 所使用的网络
#########################################

class PolicyNetworkPPO(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, init_log_std=0.0):
        """
        策略网络，输出动作均值和对数标准差
        参数 max_action 用于后续缩放动作（期望为归一化尺度，如 [1,1,1]）
        init_log_std：用于初始化 log_std_layer 的偏置（例如 0.0 对应初始标准差 1.0）
        """
        super(PolicyNetworkPPO, self).__init__()
        self.max_action_normalize = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        # 初始化 log_std_layer 的偏置为 init_log_std，使得初始标准差为 exp(init_log_std)
        nn.init.constant_(self.log_std_layer.bias, init_log_std)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 修改 clamping 范围，从 [-2, 2] 改为 [-1, 2]
        log_std = torch.clamp(log_std, min=-1, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # 这里 max_action_normalize 通常为 [1,1,1]
        max_action_tensor = torch.tensor(self.max_action_normalize, device=y_t.device, dtype=y_t.dtype)
        action = y_t * max_action_tensor
        # 计算对数概率，并减去 tanh 的雅可比项
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def evaluate_log_prob(self, state, action):
        action = torch.clamp(action, -0.999, 0.999)
        pre_tanh = atanh(action)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        """
        价值网络，输出状态值
        """
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)


#########################################
# PPO 算法的 Planner 类
#########################################

class PPOPlanner(BasePlanner):
    def __init__(self, num_seconds, state_dim=26, action_dim=3,
                 lr=4e-5, gamma=0.99, clip_ratio=0.2, ppo_epochs=10, lam=0.95,
                 batch_size=128, config_file="./config_all.yaml"):
        """
        初始化 PPOPlanner
        """
        # 加载配置文件
        config_file = os.path.join(os.path.dirname(__file__), "config_all.yaml")
        with open(config_file, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        # 设置全局随机种子
        seed = self.config.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # 当前环境级别及相关参数
        self.env_level = self.config["environment"]["level"]
        self.current_scene_index = 0

        # PPO 相关超参数
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lr = lr

        # 记录指标（部分变量可能需要根据具体奖励函数进一步更新）
        self.reach_targe_times = 0
        self.episode_align_reward = 0
        self.episode_safety_reward = 0
        self.episode_reach_target_reward = 0
        self.episode_out_of_box_penalty = 0
        self.episode_energy_penalty = 0
        self.episode_smoothness_penalty = 0
        self.episode_time_penalty = 0
        self.total_length = 0
        self.episode_path_length = 0
        self.episode_collisions = 0
        self.episode_energy = 0
        self.episode_smoothness = 0
        self.static_counter = 0

        self.num_seconds = num_seconds
        self.state_dim = state_dim   # 与 SAC 中保持一致
        self.action_dim = action_dim

        # 动作限制参数
        self.max_lin_accel = 10.0
        self.max_ang_accel = 2.0
        self.max_action = np.array([self.max_lin_accel] * 3)
        self.max_action_normalize = np.array([1] * 3)

        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetworkPPO(self.state_dim, self.action_dim, self.max_action_normalize).to(device)
        self.value_net = ValueNetwork(self.state_dim).to(device)

        self.policy_net.apply(self.initialize_weights)
        self.value_net.apply(self.initialize_weights)

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

        # 其他变量（例如上一动作、时间计数等）
        super().__init__()
        self.previous_action = np.zeros(action_dim)
        self.current_time = 0.0
        self.ticks_per_sec = 100

        # 关于海流的参数（保留原有逻辑）
        self.current_strength = 0.5
        self.current_frequency = np.pi
        self.current_mu = 0.25
        self.current_omega = 2.0
        self.gravity = 9.81
        self.cob = np.array([0, 0, 5.0]) / 100
        self.m = 31.02
        self.rho = 997
        self.V = self.m / self.rho
        self.J = np.eye(3) * 2

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'out_features') and m.out_features == self.action_dim:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def select_action(self, state):
        """
        根据当前状态选择动作，同时返回动作的对数概率和状态值
        注意：取消了额外添加噪声的操作，保证采样动作与记录的对数概率一致。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.sample(state_tensor)
            value = self.value_net(state_tensor)
        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().numpy()[0]
        value = value.cpu().numpy()[0]
        return action, log_prob, value

    def compute_returns_and_advantages(self, rewards, values, dones, last_value):
        """
        利用广义优势估计（GAE）计算每个时间步的优势和折扣回报
        """
        T = len(rewards)
        returns = np.zeros(T)
        advantages = np.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0
            else:
                next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return returns, advantages

    def update_policy(self, trajectories):
        """
        根据收集到的轨迹数据进行 PPO 更新
        """
        states = torch.FloatTensor(trajectories['states']).to(device)
        actions = torch.FloatTensor(trajectories['actions']).to(device)
        old_log_probs = torch.FloatTensor(trajectories['old_log_probs']).to(device)
        returns = torch.FloatTensor(trajectories['returns']).to(device)
        advantages = torch.FloatTensor(trajectories['advantages']).to(device)

        dataset_size = states.shape[0]
        value_coef = 0.5
        entropy_coef = 0.1

        for epoch in range(self.ppo_epochs):
            permutation = np.random.permutation(dataset_size)
            for i in range(0, dataset_size, self.batch_size):
                indices = permutation[i: i + self.batch_size]
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                # 使用 evaluate_log_prob 计算当前策略下，采样时记录的动作对数概率
                new_log_probs = self.policy_net.evaluate_log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.value_net(batch_states)
                value_loss = ((values - batch_returns.unsqueeze(1)) ** 2).mean()

                entropy = (-new_log_probs).mean()

                total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                wandb.log({
                    "ppo_policy_loss": policy_loss.item(),
                    "ppo_value_loss": value_loss.item(),
                    "ppo_entropy": entropy.item(),
                    "ppo_total_loss": total_loss.item(),
                })

    def save_model(self, episode, model_path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.policy_optimizer.state_dict(),
            'optimizer_value': self.value_optimizer.state_dict(),
        }, model_path)

    def train(self, env, num_episodes=500, max_steps=3000, model_path="ppo_best_model.pth"):
        """
        使用 custom_environment 进行训练
        """
        wandb.init(project="auv_RL_control_PPO_acceleration", name=model_path)
        wandb.config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "clip_ratio": self.clip_ratio,
            "ppo_epochs": self.ppo_epochs,
            "lam": self.lam,
            "batch_size": self.batch_size,
            "MAX_LIN_ACCEL": self.max_lin_accel,
            "Default_maxstep": max_steps,
        })

        ts = 1 / self.ticks_per_sec
        self.step_cnt = 0
        done = 0
        episode = 0
        while self.reach_targe_times < 10:
            episode_start_time = time.time()
            logging.info(f"PPO Episode {episode + 1} starting")
            env.reset()
            # 当上个 episode 结束时更新目标
            if done == 1:
                env.set_current_target(env.choose_next_target())
                env.draw_targets()

            # 用零动作获得初始传感器数据并更新状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            self.start = env.location.copy()
            self.end = np.array(env.get_current_target())
            OG_distance_to_goal = np.linalg.norm(self.start - self.end)
            print(f"start: {self.start}, end: {self.end}, distance: {OG_distance_to_goal}")
            self.current_time = 0.0

            # 重置 episode 内部记录
            self.episode_align_reward = 0
            self.episode_safety_reward = 0
            self.episode_reach_target_reward = 0
            self.episode_out_of_box_penalty = 0
            self.episode_energy_penalty = 0
            self.episode_smoothness_penalty = 0
            self.episode_time_penalty = 0
            self.total_length = 0
            self.episode_path_length = 0
            self.episode_collisions = 0
            self.episode_energy = 0
            self.episode_smoothness = 0
            self.static_counter = 0

            # 用于存储本 episode 内的轨迹数据
            traj_states = []
            traj_actions = []
            traj_old_log_probs = []
            traj_rewards = []
            traj_values = []
            traj_dones = []

            step_count = 0
            total_reward = 0
            done = 0

            while done == 0 and step_count < max_steps:
                current_pos = env.location.copy()
                # 计算当前状态特征（例如目标相对位置等）
                relative_position_to_goal = self.end - current_pos
                dist_to_goal = np.linalg.norm(relative_position_to_goal)

                pre_state_obj = EnvState(current_pos, env.velocity.copy(), env.lasers.copy())
                ocean_current = calculate_ocean_current(self, current_pos, self.current_time)
                pre_state = np.append(pre_state_obj.vec, [ocean_current, self.end])

                # 注意：pre_state 的维度应与 self.state_dim 保持一致
                #
                # current_s =  env.observation_space.copy()
                # pre_state = np.concatenate([current_s, ocean_current, self.end])

                # 选择动作（此处返回的动作与记录的对数概率应一致）
                action, log_prob, value = self.select_action(pre_state)
                # 根据海流影响调整动作
                real_action = denormalize_action(self, action)
                adjusted_action = calculate_action_effect(self, real_action, ocean_current)
                normalized_adjusted_action = normalize_action(self, adjusted_action)
                # 补充控制指令（例如附加 0 分量，以符合环境要求）
                use_action = np.append(adjusted_action, [0, 0, 0])

                # 执行动作，tick 环境并更新状态
                sensors = env.tick(use_action)
                env.update_state(sensors)
                next_pos = env.location.copy()
                next_dist_to_goal = np.linalg.norm(self.end - next_pos)
                # 判断是否到达目标
                done = 1 if next_dist_to_goal < 2 else 0
                self.current_time += ts

                post_ocean_current = calculate_ocean_current(self, next_pos, self.current_time)
                post_state_obj = EnvState(next_pos, env.velocity.copy(), env.lasers.copy())
                post_state = np.append(post_state_obj.vec, [post_ocean_current, self.end])
                # post_s = env.observation_space.copy()
                # post_state = np.concatenate([post_s, ocean_current, self.end])

                # 计算奖励
                reward, pg_rd, aln_rd, safe_pnt, reach_tg_rd, smt_pnt = calculate_reward(self, pre_state, post_state, action)
                total_reward += reward

                # 记录当前 step 数据
                traj_states.append(pre_state)
                traj_actions.append(action)
                traj_old_log_probs.append(log_prob)
                traj_rewards.append(reward)
                traj_values.append(value)
                traj_dones.append(done)

                # 记录 wandb 信息（例如距离障碍物、目标距离等）
                if env.obstacles:
                    distances = [np.linalg.norm(current_pos - np.array(obs)) for obs in env.obstacles]
                    distance_to_nearest_obstacle = min(distances)
                else:
                    distance_to_nearest_obstacle = 100.0

                wandb.log({
                    "x_pos": next_pos[0],
                    "y_pos": next_pos[1],
                    "z_pos": next_pos[2],
                    "step_count": step_count,
                    "distance_to_nearest_obstacle": distance_to_nearest_obstacle,
                    "distance_to_goal": next_dist_to_goal,
                    "align_reward": aln_rd,
                    "safety_reward": safe_pnt,
                    "reach_target_reward": reach_tg_rd,
                    "smoothness_penalty": smt_pnt,
                    "progress_reward": pg_rd,
                    "step_total_reward": reward
                })

                # 如果奖励过低则提前结束当前 episode（此处逻辑根据需求调整）
                if  total_reward < -1:
                    break

                step_count += 1

            # episode 结束，计算最后一步的 value（若 done==1，则置 0）
            final_state = traj_states[-1]
            final_state_tensor = torch.FloatTensor(final_state).unsqueeze(0).to(device)
            with torch.no_grad():
                last_value = self.value_net(final_state_tensor).cpu().numpy()[0] if done == 0 else 0

            traj_rewards = np.array(traj_rewards, dtype=np.float32)
            traj_values = np.array(traj_values, dtype=np.float32)
            traj_dones = np.array(traj_dones, dtype=np.float32)


            returns, advantages = self.compute_returns_and_advantages(traj_rewards, traj_values, traj_dones, last_value)

            trajectories = {
                'states': np.array(traj_states, dtype=np.float32),
                'actions': np.array(traj_actions, dtype=np.float32),
                'old_log_probs': np.array(traj_old_log_probs, dtype=np.float32),
                'returns': returns,
                'advantages': advantages,
            }

            # 使用 PPO 更新策略
            self.update_policy(trajectories)

            # 记录 episode 统计信息
            episode_duration = time.time() - episode_start_time
            wandb.log({
                "episode": episode + 1,
                "reach_target_times": self.reach_targe_times,
                "eps_total_reward": total_reward,
                "eps_distance_to_goal": next_dist_to_goal,
                "eps_align_reward": self.episode_align_reward,
                "eps_safety_reward": self.episode_safety_reward,
                "eps_reach_target_reward": self.episode_reach_target_reward,
                "eps_out_of_box_penalty": self.episode_out_of_box_penalty,
                "eps_energy_penalty": self.episode_energy_penalty,
                "eps_smoothness_penalty": self.episode_smoothness_penalty,
                "eps_time_penalty": self.episode_time_penalty,
                "eps_ave_length_per_step": self.episode_path_length / step_count if step_count > 0 else 0,
                "episode_path_length": self.episode_path_length,
                "episode_collisions": self.episode_collisions,
                "episode_energy": self.episode_energy,
                "episode_smoothness": self.episode_smoothness,
                "episode_duration": episode_duration
            })

            logging.info(f"Episode {episode + 1} completed - Total Reward: {total_reward}")
            if episode % 40 == 0:
                self.save_model(episode + 1, model_path)
            if done == 1:
                self.reach_targe_times += 1
                self.save_model(episode +1, model_path)
                wandb.log({"episode": episode + 1, "reach_target_times": self.reach_targe_times})
            episode += 1


#########################################
# 辅助状态类（与 SAC 中保持一致）
#########################################

class EnvState:
    def __init__(self, location, velocity, lasers):
        # 将各传感器数据拼接为一个向量
        self.vec = np.concatenate([location, velocity, lasers])