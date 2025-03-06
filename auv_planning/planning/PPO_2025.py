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

# Assuming BasePlanner is defined in base.py
from .base import BasePlanner

# Import utility functions for ocean current, action effects, and rewards
from .rl_utils import (calculate_ocean_current, calculate_action_effect,
                       normalize_action, denormalize_action)
from .rl_rewards_PPO import calculate_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# Utility Function: Inverse tanh (atanh)
#########################################
def atanh(x):
    # Clamp x to (-1+ε, 1-ε) for numerical stability
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

#########################################
# Policy Network for PPO-LSTM
#########################################
class PolicyNetworkPPO(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, init_log_std=0.70):
        """
        初始化带 LSTM 的策略网络
        - state_dim: 状态维度
        - action_dim: 动作维度
        - max_action: 最大动作（如 [1,1,1]）
        - init_log_std: 初始对数标准差
        """
        super(PolicyNetworkPPO, self).__init__()
        self.max_action_normalize = max_action
        # 先用 MLP 提取特征
        # Wider network with more units
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),  # Increase from 64 to 128
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        # LSTM with more units
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)
        nn.init.constant_(self.log_std_layer.bias, init_log_std)

        # Initialize the mean layer with smaller values for better initial behavior
        nn.init.uniform_(self.mean_layer.weight, -0.003, 0.003)
        nn.init.constant_(self.mean_layer.bias, 0)

    def forward(self, state, hidden=None):
        """
        state 可以为 2D (batch, state_dim) 或 3D (batch, seq_len, state_dim)
        如果为 2D，则自动扩展 seq_len=1
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # 变为 (batch, 1, state_dim)
        x = self.mlp(state)  # (batch, seq_len, 64)
        if hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, 128, device=x.device)
            c0 = torch.zeros(1, batch_size, 128, device=x.device)
            hidden = (h0, c0)
        x, hidden = self.lstm(x, hidden)  # (batch, seq_len, 64)
        mean = self.mean_layer(x)         # (batch, seq_len, action_dim)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-3, max=1)
        # 如果序列长度为1，则 squeeze 掉 seq 维度
        if mean.size(1) == 1:
            mean = mean.squeeze(1)
            log_std = log_std.squeeze(1)
        return mean, log_std, hidden

    def sample(self, state, hidden=None):
        """
        采用 reparameterization trick 采样动作，并返回动作、对数概率以及 hidden state
        """
        mean, log_std, hidden = self.forward(state, hidden)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        max_action_tensor = torch.tensor(self.max_action_normalize, device=y_t.device, dtype=y_t.dtype)
        action = y_t * max_action_tensor
        # 调整 log probability 以考虑 tanh 的压缩效应
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, hidden

    def evaluate_log_prob(self, state, action, hidden=None):
        """
        计算给定 state 和 action 下的对数概率
        """
        action = torch.clamp(action, -0.999, 0.999)
        pre_tanh = atanh(action)
        mean, log_std, _ = self.forward(state, hidden)
        std = log_std.exp()
        normal = D.Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob

#########################################
# Value Network for PPO-LSTM
#########################################
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        """
        初始化带 LSTM 的价值网络
        """
        super(ValueNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, state, hidden=None):
        """
        state 可以为 2D (batch, state_dim) 或 3D (batch, seq_len, state_dim)
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)
        x = self.mlp(state)
        if hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, 128, device=x.device)
            c0 = torch.zeros(1, batch_size, 128, device=x.device)
            hidden = (h0, c0)
        x, hidden = self.lstm(x, hidden)
        value = self.value_layer(x)
        if value.size(1) == 1:
            value = value.squeeze(1)
        return value, hidden

#########################################
# Optimized PPO-LSTM Planner
#########################################
class PPOPlanner(BasePlanner):
    def __init__(self, num_seconds, state_dim=23, action_dim=3,
                 lr=1e-3,  # 将学习率调低
                 gamma=0.99, clip_ratio=0.1, ppo_epochs=10, lam=0.95,
                 batch_size=64, config_file="./config_all.yaml"):
        """
        初始化 PPO-LSTM 规划器
        """
        config_file = os.path.join(os.path.dirname(__file__), "config_all.yaml")
        with open(config_file, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        # 设置随机种子
        seed = self.config.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.env_level = self.config["environment"]["level"]
        self.current_scene_index = 0

        # PPO 超参数
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Add learning rate warmup and decay
        self.lr_start = lr
        self.lr_end = lr * 0.1
        self.warmup_steps = 100

        # Increase reward scale for better signal
        self.reward_scale = 5.0

        # Stronger gradient clipping - prevent large updates
        self.max_grad_norm = 0.5


        # 其他性能指标初始化
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
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 动作限制与归一化参数
        self.max_lin_accel = 2.0
        self.max_ang_accel = 2.0
        self.max_action = np.array([self.max_lin_accel] * 3)
        self.max_action_normalize = np.array([1] * 3)

        # 初始化网络（采用 PPO-LSTM 结构）
        self.policy_net = PolicyNetworkPPO(self.state_dim, self.action_dim, self.max_action_normalize).to(device)
        self.value_net = ValueNetwork(self.state_dim).to(device)
        self.policy_net.apply(self.initialize_weights)
        self.value_net.apply(self.initialize_weights)

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

        # 学习率调度器（可选）
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=200, gamma=0.95)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=200, gamma=0.95)

        super().__init__()
        self.previous_action = np.zeros(action_dim)
        self.current_time = 0.0
        self.ticks_per_sec = 100

        # Ocean current parameters (保持不变)
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
        """
        自定义权重初始化
        """
        if isinstance(m, nn.Linear):
            if hasattr(m, 'out_features') and m.out_features == self.action_dim:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def select_action(self, state):
        """
        给定当前 state 选择动作，并返回动作、对数概率及价值
        注意：这里每次输入单步 state（shape: [state_dim]），内部自动扩展为 [1, state_dim]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _ = self.policy_net.sample(state_tensor)
            value, _ = self.value_net(state_tensor)
        return (action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
                value.cpu().numpy()[0])

    def compute_returns_and_advantages(self, rewards, values, dones, last_value):
        """
        使用 GAE 计算 returns 和 advantages
        """
        T = len(rewards)
        returns = np.zeros(T)
        advantages = np.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            next_value = 0 if dones[t] else (last_value if t == T - 1 else values[t + 1])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return returns, advantages

    # def update_policy(self, trajectories):
    #     """
    #     使用 PPO 更新策略和价值网络
    #     修改部分：将整个 episode 作为一个序列（batch_size=1）输入 LSTM
    #     """
    #     # 将数据 reshape 成 (1, T, feature_dim)
    #     states = torch.FloatTensor(trajectories['states']).unsqueeze(0).to(device)           # (1, T, state_dim)
    #     actions = torch.FloatTensor(trajectories['actions']).unsqueeze(0).to(device)          # (1, T, action_dim)
    #     old_log_probs = torch.FloatTensor(trajectories['old_log_probs']).unsqueeze(0).to(device)  # (1, T, 1)
    #     returns = torch.FloatTensor(trajectories['returns']).unsqueeze(0).to(device)          # (1, T)
    #     advantages = torch.FloatTensor(trajectories['advantages']).unsqueeze(0).to(device)    # (1, T)
    #
    #     value_coef = 0.5
    #     entropy_coef = 0.01
    #
    #     # 由于数据为序列，这里不进行随机 mini-batch 打乱
    #     for epoch in range(self.ppo_epochs):
    #         new_log_probs = self.policy_net.evaluate_log_prob(states, actions)
    #         ratio = torch.exp(new_log_probs - old_log_probs)
    #         # 调整 advantage 维度 (1, T, 1)
    #         advantages_ = advantages.unsqueeze(-1)
    #         surr1 = ratio * advantages_
    #         surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_
    #         policy_loss = -torch.min(surr1, surr2).mean()
    #
    #         values, _ = self.value_net(states)
    #         value_loss = ((values.squeeze(-1) - returns)**2).mean()
    #
    #         entropy = (-new_log_probs).mean()
    #         total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    #
    #         self.policy_optimizer.zero_grad()
    #         self.value_optimizer.zero_grad()
    #         total_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
    #         torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
    #         self.policy_optimizer.step()
    #         self.value_optimizer.step()
    #
    #         wandb.log({
    #             "ppo_policy_loss": policy_loss.item(),
    #             "ppo_value_loss": value_loss.item(),
    #             "ppo_entropy": entropy.item(),
    #             "ppo_total_loss": total_loss.item(),
    #         })
    #
    #     self.policy_scheduler.step()
    #     self.value_scheduler.step()
    def update_policy(self, trajectories):
        # Scale rewards for better signal
        scaled_returns = trajectories['returns'] * self.reward_scale
        scaled_advantages = trajectories['advantages'] * self.reward_scale

        # Reshape data
        states = torch.FloatTensor(trajectories['states']).unsqueeze(0).to(device)
        actions = torch.FloatTensor(trajectories['actions']).unsqueeze(0).to(device)
        old_log_probs = torch.FloatTensor(trajectories['old_log_probs']).unsqueeze(0).to(device)
        returns = torch.FloatTensor(scaled_returns).unsqueeze(0).to(device)
        advantages = torch.FloatTensor(scaled_advantages).unsqueeze(0).to(device)

        # Adaptive value coefficient based on returns magnitude
        value_coef = 0.5
        entropy_coef = 0.01  # Maintain exploration

        # Implement early stopping based on KL divergence
        target_kl = 0.015

        for epoch in range(self.ppo_epochs):
            new_log_probs = self.policy_net.evaluate_log_prob(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages_ = advantages.unsqueeze(-1)

            # Standard PPO objective
            surr1 = ratio * advantages_
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_
            policy_loss = -torch.min(surr1, surr2).mean()

            values, _ = self.value_net(states)
            value_loss = ((values.squeeze(-1) - returns) ** 2).mean()

            entropy = (-new_log_probs).mean()
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            # Stricter gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=self.max_grad_norm)

            self.policy_optimizer.step()
            self.value_optimizer.step()
            wandb.log({
                "ppo_policy_loss": policy_loss.item(),
                "ppo_value_loss": value_loss.item(),
                "ppo_entropy": entropy.item(),
                "ppo_total_loss": total_loss.item(),
            })

            # Calculate approximate KL for early stopping
            approx_kl = ((old_log_probs - new_log_probs) ** 2).mean().item()
            if approx_kl > target_kl:
                break  # Early stopping if updates are too large


    def save_model(self, episode, model_path):
        """
        保存模型参数
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.policy_optimizer.state_dict(),
            'optimizer_value': self.value_optimizer.state_dict(),
        }, model_path)

    def train(self, env, num_episodes=500, max_steps=3000, model_path="ppo_best_model.pth"):
        """
        主要训练循环（保持大部分逻辑不变，只是利用 PPO-LSTM 进行时序建模）
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

        while self.reach_targe_times < 10 and episode < num_episodes:
            episode_start_time = time.time()
            logging.info(f"PPO Episode {episode + 1} starting")
            env.reset()
            if done == 1:
                env.set_current_target(env.choose_next_target())
                env.draw_targets()

            # 初始化 state
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            self.start = env.location.copy()
            self.end = np.array(env.get_current_target())
            OG_distance_to_goal = np.linalg.norm(self.start - self.end)
            print(f"start: {self.start}, end: {self.end}, distance: {OG_distance_to_goal}")
            self.current_time = 0.0

            # 重置 episode 统计量
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

            # 记录本 episode 的轨迹
            traj_states, traj_actions = [], []
            traj_old_log_probs, traj_rewards = [], []
            traj_values, traj_dones = [], []

            step_count = 0
            total_reward = 0
            done = 0

            while done == 0 and step_count < max_steps:
                current_pos = env.location.copy()
                relative_position_to_goal = self.end - current_pos
                dist_to_goal = np.linalg.norm(relative_position_to_goal)

                # 构造 state（保持不变）
                pre_state_obj = EnvState(current_pos/10, env.velocity.copy()/10, env.lasers.copy()/10)
                ocean_current = calculate_ocean_current(self, current_pos, self.current_time)
                pre_state = np.append(pre_state_obj.vec, [ocean_current])

                action, log_prob, value = self.select_action(pre_state)
                real_action = denormalize_action(self, action)
                adjusted_action = calculate_action_effect(self, real_action, ocean_current)
                normalized_adjusted_action = normalize_action(self, adjusted_action)
                use_action = np.append(adjusted_action, [0, 0, 0])

                sensors = env.tick(use_action)
                env.update_state(sensors)
                next_pos = env.location.copy()
                next_dist_to_goal = np.linalg.norm(self.end - next_pos)
                done = 1 if next_dist_to_goal < 2 else 0
                self.current_time += ts

                post_ocean_current = calculate_ocean_current(self, next_pos, self.current_time)
                post_state_obj = EnvState(next_pos/10, env.velocity.copy(), env.lasers.copy()/10)
                post_state = np.append(post_state_obj.vec, [post_ocean_current])

                reward, pg_rd, aln_rd, safe_pnt, reach_tg_rd, smt_pnt = calculate_reward(self, pre_state, post_state, action)
                total_reward += reward

                traj_states.append(pre_state)
                traj_actions.append(action)
                traj_old_log_probs.append(log_prob)
                traj_rewards.append(reward)
                traj_values.append(value)
                traj_dones.append(done)

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

                if total_reward < -1:
                    print("early stop")
                    break

                step_count += 1

            # 计算最后一个 state 的 value
            final_state = traj_states[-1]
            final_state_tensor = torch.FloatTensor(final_state).unsqueeze(0).to(device)
            with torch.no_grad():
                last_value = self.value_net(final_state_tensor)[0].cpu().numpy()[0] if done == 0 else 0

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

            # 更新策略
            self.update_policy(trajectories)

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
                self.save_model(episode + 1, model_path)
                wandb.log({"episode": episode + 1, "reach_target_times": self.reach_targe_times})
            episode += 1

#########################################
# Environment State Helper Class
#########################################
class EnvState:
    def __init__(self, location, velocity, lasers):
        # 将传感器数据拼接成一个向量
        self.vec = np.concatenate([location, velocity, lasers])