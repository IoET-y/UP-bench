#auv_planning/planning/PFM_2025.py

import numpy as np
import math
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner


class PFMPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 k_att=1.0, k_rep=100.0, d0=10.0, desired_speed=3.0):
        """
        参数说明：
          - grid_resolution: 障碍检测时的离散化分辨率（单位与环境一致）
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率
          - k_att: 吸引力系数
          - k_rep: 排斥力系数
          - d0: 排斥影响距离阈值（当障碍距离小于 d0 时产生排斥力）
          - desired_speed: 期望运动速度（用于生成期望状态）

        同时记录 wandb 的评价指标：
          - ave_path_length, ave_excu_time, ave_smoothness, ave_energy, ave_plan_time
        """
        # EVALUATION METRICS
        self.ave_path_length = 0
        self.ave_excu_time = 0
        self.ave_smoothness = 0
        self.ave_energy = 0
        self.ave_plan_time = 0

        self.grid_resolution = grid_resolution
        self.max_steps = max_steps
        self.max_lin_accel = max_lin_accel
        self.collision_threshold = collision_threshold
        self.ticks_per_sec = ticks_per_sec
        self.ts = 1.0 / self.ticks_per_sec
        self.current_time = 0.0

        # 规划区域设置（假定区域：x:0~100, y:0~100, z:-100~0）
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0
        self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
        self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
        self.nz = int((self.z_max - self.z_min) / self.grid_resolution)

        # 离线设计 LQR 控制器（与 D* Lite 版本一致）
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

        # 潜力场参数
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.desired_speed = desired_speed

        # 障碍地图：字典，key 为网格索引，value 为占据状态（1:障碍，0:自由）
        self.obstacle_map = {}

        super().__init__()

    # --- 网格坐标转换 ---
    def world_to_index(self, pos):
        ix = int((pos[0] - self.x_min) / self.grid_resolution)
        iy = int((pos[1] - self.y_min) / self.grid_resolution)
        iz = int((pos[2] - self.z_min) / self.grid_resolution)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        iz = min(max(iz, 0), self.nz - 1)
        return (ix, iy, iz)

    def index_to_world(self, idx):
        x = self.x_min + idx[0] * self.grid_resolution + self.grid_resolution / 2.0
        y = self.y_min + idx[1] * self.grid_resolution + self.grid_resolution / 2.0
        z = self.z_min + idx[2] * self.grid_resolution + self.grid_resolution / 2.0
        return np.array([x, y, z])

    # --- 障碍地图更新 ---
    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        根据传感器读数更新障碍地图：
          - sensor_readings: 长度为14的列表或数组；若读数 < 10，则视为障碍
          - 根据预设传感器方向（假设 agent 姿态为零），将测量转换为世界坐标，
            再转换为网格索引进行标记；同时沿射线采样，将视野内其他单元标记为空闲（0）
        """
        directions = []
        # 8个水平激光（角度0,45,...,315度）
        for i in range(8):
            angle = math.radians(i * 45)
            directions.append(np.array([math.cos(angle), math.sin(angle), 0]))
        # UpRangeSensor
        directions.append(np.array([0, 0, 1]))
        # DownRangeSensor
        directions.append(np.array([0, 0, -1]))
        # UpInclinedRangeSensor：两束
        directions.append(np.array([math.cos(math.radians(45)), 0, math.sin(math.radians(45))]))
        directions.append(np.array([0, math.cos(math.radians(45)), math.sin(math.radians(45))]))
        # DownInclinedRangeSensor：两束
        directions.append(np.array([math.cos(math.radians(45)), 0, -math.sin(math.radians(45))]))
        directions.append(np.array([0, math.cos(math.radians(45)), -math.sin(math.radians(45))]))

        updated_cells = []
        max_range = 10.0
        for reading, direction in zip(sensor_readings, directions):
            if reading < max_range:
                obstacle_pos = current_pos + reading * direction
                cell = self.world_to_index(obstacle_pos)
                if self.obstacle_map.get(cell, 0) != 1:
                    self.obstacle_map[cell] = 1
                    updated_cells.append(cell)
            num_samples = int(reading / self.grid_resolution)
            for s in range(num_samples):
                sample_distance = s * self.grid_resolution
                sample_pos = current_pos + sample_distance * direction
                sample_cell = self.world_to_index(sample_pos)
                if self.obstacle_map.get(sample_cell, 0) != 0:
                    self.obstacle_map[sample_cell] = 0
        return updated_cells

    # --- 潜力场力计算 ---
    def compute_potential_force(self, current_pos, goal_pos):
        """
        计算当前点的总潜力场力，包括吸引力与排斥力。
        吸引力：F_attr = k_att * (goal - current)
        排斥力：对每个障碍，如果距离小于 d0，则 F_rep = k_rep * (1/d - 1/d0) / d^2 * (current - obs)/d
        """
        # 吸引力
        F_attr = self.k_att * (goal_pos - current_pos)
        F_rep = np.zeros(3)
        # 遍历障碍地图中标记为障碍的单元
        for cell, occ in self.obstacle_map.items():
            if occ == 1:
                obs_pos = self.index_to_world(cell)
                diff = current_pos - obs_pos
                d = np.linalg.norm(diff)
                if d < self.d0 and d > 1e-3:
                    rep_force = 1.5*self.k_rep * (1.0 / d - 1.0 / self.d0) / (d ** 2) * (diff / d)
                    F_rep += rep_force
        F_total = F_attr + F_rep
        return F_total

    # --- 路径平滑（用于可视化） ---
    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        if path is None or len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        # 对每个维度进行边界约束
        smooth_path[:, 0] = np.clip(smooth_path[:, 0], self.x_min, self.x_max)
        smooth_path[:, 1] = np.clip(smooth_path[:, 1], self.y_min, self.y_max)
        smooth_path[:, 2] = np.clip(smooth_path[:, 2], self.z_min, self.z_max)
        return [pt for pt in smooth_path]

    # --- 主训练及规划循环 ---
    def train(self, env, num_episodes=10):
        wandb.init(project="auv_PFM_planning", name="PFM_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max]
            },
            "k_att": self.k_att,
            "k_rep": self.k_rep,
            "d0": self.d0,
            "desired_speed": self.desired_speed
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            logging.info(f"PFM Episode {episode + 1} starting")
            env.reset()

            # 获取初始状态与目标
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  # 3D 起点
            target = env.get_current_target()
            goal_pos = np.array(target)  # 3D 目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 重置障碍地图
            self.obstacle_map = {}

            step_count = 0
            total_path_length = 0.0
            collisions = 0
            energy = 0.0
            smoothness = 0.0
            prev_u = None
            current_pos = start_pos.copy()

            episode_start_time = time.time()
            episode_start_running_time = time.time()

            # 控制循环：每个周期基于当前状态利用潜力场方法计算期望运动方向
            while step_count < self.max_steps:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # 更新障碍地图
                sensor_readings = env.lasers.copy()  # 14 个传感器读数
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # 计算潜力场力
                F_total = self.compute_potential_force(current_pos, goal_pos)
                norm_F = np.linalg.norm(F_total)
                if norm_F > 1e-6:
                    direction = F_total / norm_F
                else:
                    direction = np.zeros(3)
                # 根据期望速度生成期望速度向量
                v_des = self.desired_speed * direction
                # 计算期望短期目标位置：当前位置 + v_des * ts
                pos_des = current_pos + v_des * self.ts

                # 构造当前状态与期望状态（状态由位置与速度构成）
                x_current = np.hstack([current_pos, env.velocity.copy()])
                x_des = np.hstack([pos_des, v_des])
                error_state = x_current - x_des

                # LQR 控制计算控制输入
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                action = np.concatenate([u, np.zeros(3)])
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()

                # 更新评价指标
                distance_moved = np.linalg.norm(new_pos - current_pos)
                total_path_length += distance_moved
                energy += np.linalg.norm(u) ** 2
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

                for obs in env.obstacles:
                    if np.linalg.norm(new_pos - np.array(obs)) < self.collision_threshold:
                        collisions += 1
                        break

                env.env.draw_line(current_pos.tolist(), new_pos.tolist(), color=[0,100,0], thickness=3, lifetime=0)
                current_pos = new_pos
                step_count += 1
                self.current_time += self.ts

                wandb.log({
                    "x_pos": current_pos[0],
                    "y_pos": current_pos[1],
                    "z_pos": current_pos[2],
                    "step_count": step_count,
                    "distance_to_goal": np.linalg.norm(current_pos - goal_pos)
                })

            episode_running_duration = time.time() - episode_start_running_time
            wandb.log({
                "episode": episode + 1,
                "eps_reach_target": reach_target_count,
                "eps_distance_to_goal": np.linalg.norm(current_pos - goal_pos),
                "eps_ave_length_per_step": total_path_length / step_count if step_count > 0 else 0,
                "episode_path_length": total_path_length,
                "episode_collisions": collisions,
                "episode_energy": energy,
                "episode_smoothness": smoothness,
                "episode_planning_duration": time.time() - episode_start_time,
                "episode_running_duration": episode_running_duration
            })
            if np.linalg.norm(current_pos - goal_pos) < 2:
                self.ave_path_length += total_path_length
                self.ave_excu_time += episode_running_duration
                self.ave_plan_time += (time.time() - episode_start_time)
                self.ave_smoothness += smoothness
                self.ave_energy += energy

            logging.info(
                f"Episode {episode + 1} completed - Path Length: {total_path_length}, Steps: {step_count}, Collisions: {collisions}")
            episode += 1

            if reach_target_count > 9  or episode >= num_episodes:
                wandb.log({
                    "ave_path_length": self.ave_path_length / reach_target_count,
                    "ave_excu_time": self.ave_excu_time / reach_target_count,
                    "ave_plan_time": self.ave_plan_time / reach_target_count,
                    "ave_smoothness": self.ave_smoothness / reach_target_count,
                    "ave_energy": self.ave_energy / reach_target_count
                })
                ave_path_length = self.ave_path_length / reach_target_count
                ave_excu_time = self.ave_excu_time / reach_target_count
                ave_plan_time = self.ave_plan_time / reach_target_count
                ave_smoothness = self.ave_smoothness / reach_target_count
                ave_energy = self.ave_energy / reach_target_count
                print(f"ave_path_length: {ave_path_length}")
                print(f"ave_excu_time: {ave_excu_time}")
                print(f"ave_plan_time: {ave_plan_time}")
                print(f"ave_smoothness: {ave_smoothness}")
                print(f"ave_energy: {ave_energy}")
                successrate = reach_target_count/num_episodes
                return successrate,ave_path_length, ave_excu_time, ave_plan_time, ave_smoothness, ave_energy
            env.set_current_target(env.choose_next_target())


        logging.info("PFM Planning finished training.")
        return
