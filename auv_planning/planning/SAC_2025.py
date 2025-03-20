# sac_planner.py

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

from .base import BasePlanner


# Import ocean current, action related tool functions and reward functions from the tool module
from .rl_utils import (calculate_ocean_current, calculate_action_effect,
                       normalize_action, denormalize_action)
from .rl_rewards import calculate_reward

# # Import a custom environment

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("training.log"),
    logging.StreamHandler()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Auxiliary state class, used to construct the state object required by the reward function (provides .vec attribute)
class EnvState:
    def __init__(self, location, rotation, velocity, lasers):

        self.vec = np.concatenate([location, rotation, velocity, lasers])


# Prioritize experience replay buffer
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


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.max_action_normalize = max_action
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
        max_action_tensor = torch.tensor(self.max_action_normalize, device=y_t.device, dtype=y_t.dtype)
        action = y_t * max_action_tensor
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob


# Q net
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


# SAC-planner
class SACPlanner(BasePlanner):
    def __init__(self, num_obstacles=20, state_dim=29, action_dim=3,
                 lr=2e-3, gamma=0.95, tau=0.02, batch_size=128,
                 replay_buffer_size=100000, config_file="./config_all.yaml"):
        config_file = os.path.join(os.path.dirname(__file__), "config_all.yaml")
        with open(config_file, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

          seed = self.config.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.env_level = self.config["environment"]["level"]
        self.current_scene_index = 0

        # SAC parameter
        self.target_entropy = -np.prod(action_dim) * 1
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)
        self.alpha = self.log_alpha.exp().item()

        self.replay_buffer_size = replay_buffer_size
        self.per_alpha = 0.6
        self.memory = PrioritizedReplayBuffer(replay_buffer_size, alpha=self.per_alpha)
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.reach_targe_times = 0
        self.episode_distance_reward = 0
        self.episode_align_reward = 0
        self.episode_current_utilization_reward = 0
        self.episode_safety_reward = 0
        self.episode_reach_target_reward = 0
                   
        # Benchmark 
        self.static_counter = 0
        self.total_length = 0
        self.episode_path_length = 0
        self.episode_collisions = 0
        self.episode_energy = 0
        self.episode_smoothness = 0
        self.episode_out_of_box_penalty = 0
        self.episode_energy_penalty = 0
        self.episode_smoothness_penalty = 0
        self.episode_time_penalty = 0




        self.state_dim = state_dim   
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size


        self.start = None
        self.end = None
        self.OG_distance_to_goal = None
        self.prev_distance_to_goal = None


        self.max_lin_accel = 20.0
        self.max_ang_accel = 2.0
        self.max_action = np.array([self.max_lin_accel] * 3)
        self.max_action_normalize = np.array([1] * 3)


        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.max_action_normalize).to(device)
        self.q_net1 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q_net2 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_q_net1 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_q_net2 = QNetwork(self.state_dim, self.action_dim).to(device)

        self.policy_net.apply(self.initialize_weights)
        self.q_net1.apply(self.initialize_weights)
        self.q_net2.apply(self.initialize_weights)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)

        # lr-schedular
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=768, gamma=0.95)
        self.q_scheduler1 = optim.lr_scheduler.StepLR(self.q_optimizer1, step_size=768, gamma=0.95)
        self.q_scheduler2 = optim.lr_scheduler.StepLR(self.q_optimizer2, step_size=768, gamma=0.95)

        super().__init__()

        self.previous_action = np.zeros(action_dim)
        self.epsd = 0
        self.done = 0
        self.first_reach_close = 0
        self.static_cnt = 0
        self.last_distance_to_goal = 0
        self.static_on = False
        self.closest_pos = 0
        self.ticks_per_sec = 100
        self.current_time = 0.0

        # ocean parameter
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

    def select_action(self, state, inference=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            normalized_action, _ = self.policy_net.sample(state_tensor)
        normalized_action = normalized_action.cpu().numpy()[0]
        action = np.clip(normalized_action, -self.max_action_normalize, self.max_action_normalize)
        return action

    def remember(self, state, action, adjust_action, reward, next_state, done):
        self.memory.add((state, action, adjust_action, reward, next_state, done), td_error=0.0)

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        update_times = 1  
        for _ in range(update_times):
            samples, weights, indices = self.memory.sample(self.batch_size, beta=self.beta, alpha=self.per_alpha)
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
            states, actions, adjust_actions, rewards, next_states, dones = zip(*samples)
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
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
      
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # 
        # self.policy_scheduler.step()
        # self.q_scheduler1.step()
        # self.q_scheduler2.step()
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

    def train(self, env , num_episodes=500, max_steps=3000, model_path="sac_best_model.pth"):
      """
      Training with custom_environment:
      - Call env.reset() to reset the environment (draw targets and obstacles internally)
      - Call env.update_state() to update internal states (location, rotation, velocity, lasers) after getting initial sensor data
      - Use env.get_current_target() to get the target position
      - Call env.tick(adjusted_action) at each step to execute the action and update the state
      """
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
            "Default_maxstep": max_steps,
        })
        ts = 1 / self.ticks_per_sec
        self.step_cnt = 0
        done = 0
        episode = 0

        while self.reach_targe_times < 10: #episode in range(num_episodes):

            episode_start_time = time.time()
            logging.info(f"Episode {episode + 1} starting")
          
            env.reset()
            if done == 1:
                env.set_current_target(env.choose_next_target())
                env.draw_targets()

     
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            self.start = env.location.copy()
            self.end = np.array(env.get_current_target())
            self.OG_distance_to_goal = np.linalg.norm(self.start - self.end)
            print(f"start point at:{self.start} end point at: {self.end}, distance is : {self.OG_distance_to_goal}")

            self.prev_distance_to_goal = self.OG_distance_to_goal
            self.closest_pos = self.OG_distance_to_goal

            self.episode_distance_reward = 0
            self.episode_align_reward = 0
            self.episode_current_utilization_reward = 0
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

            step_count = 0
            total_reward = 0
            done = 0
            max_steps_episode = max_steps
            added_steps = 0

            while  done == 0 and step_count < max_steps_episode:
                current_pos = env.location.copy()

                relative_position_to_goal = self.end - current_pos
                dist_to_goal = np.linalg.norm(relative_position_to_goal)

                self.prev_distance_to_goal = dist_to_goal

                if env.obstacles:
                    distances = [np.linalg.norm(current_pos - np.array(obs)) for obs in env.obstacles]
                    distance_to_nearest_obstacle = min(distances)
                else:
                    distance_to_nearest_obstacle = 100.0

                pre_state = EnvState(current_pos, env.rotation.copy(), env.velocity.copy(), env.lasers.copy())
                ocean_current = calculate_ocean_current(self, current_pos, self.current_time)
                pre_state =  np.append(pre_state.vec, [ocean_current,self.end])

                action = self.select_action(pre_state)
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

                post_state = EnvState(next_pos, env.rotation.copy(), env.velocity.copy(), env.lasers.copy())
                post_state =  np.append(post_state.vec, [post_ocean_current,self.end])

                reward, pg_rd, aln_rd, safe_pnt, reach_tg_rd, smt_pnt = calculate_reward(self, pre_state, post_state, action)
                total_reward += reward
                if total_reward < -1000 or self.episode_out_of_box_penalty <-1000:
                    break


                self.remember(pre_state, action, normalized_adjusted_action, reward, post_state, done)
                # if next_pos[2] > 0 or distance_to_nearest_obstacle < 1.5:
                #     break
                self.update_policy()

                if next_dist_to_goal < 10 and step_count >= max_steps_episode - 1 and added_steps < 512 and next_dist_to_goal < dist_to_goal:
                    max_steps_episode += 1
                    added_steps += 1
                    logging.info(f"Increasing max steps to {max_steps_episode} for additional exploration.")

                step_count += 1


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
                    "every_rewards_all":reward
                })

                if done == 1:
                    self.reach_targe_times += 1
                    wandb.log({"episode": episode + 1, "reach_targe_times": self.reach_targe_times})
                    self.save_model(episode + 1, model_path)
                    break

            episode_duration = time.time() - episode_start_time
            wandb.log({
                "episode": episode + 1,
                "eps_reach_target": self.reach_targe_times,
                "eps_total_reward": total_reward,
                "eps_distance_to_goal": next_dist_to_goal,
                "eps_align_reward": self.episode_align_reward,
                "eps_safety_reward": self.episode_safety_reward,
                "eps_reach_target_reward": self.episode_reach_target_reward,
                "eps_out_of_box_penalty": self.episode_out_of_box_penalty,
                "eps_energy_penalty ":self.episode_energy_penalty ,
                "eps_smoothness_penalty":self.episode_smoothness_penalty ,
                "eps_time_penalty":self.episode_time_penalty ,
                "eps_ave_length_per_step": self.episode_path_length / step_count if step_count > 0 else 0,
                "episode_path_length": self.episode_path_length,
                "episode_collisions": self.episode_collisions,
                "episode_energy": self.episode_energy,
                "episode_smoothness": self.episode_smoothness,
                "episode_duration": episode_duration
            })
            logging.info(f"Episode {episode + 1} completed - Total Reward: {total_reward}")
            self.save_model(episode + 1, model_path)
            episode += 1

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


