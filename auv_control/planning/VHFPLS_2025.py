import numpy as np
import math
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner


def normalize_angle(angle):
    """将角度归一化到 [-pi, pi] 范围内"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VFHPlusPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.1, max_steps=2000,
                 max_lin_accel=20, collision_threshold=5.0, ticks_per_sec=100,
                 desired_speed=3.0, hist_resolution_deg=5, free_threshold=0.5,
                 alpha=1.0, beta=0.5):
        """
        参数说明：
          - grid_resolution: 用于平滑路径时（以及障碍检测）的离散化分辨率（单位与环境一致）
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率
          - desired_speed: 期望运动速度 [m/s]
          - hist_resolution_deg: 极坐标直方图分辨率（角度分辨率，单位：度）
          - free_threshold: 直方图中小于该值的扇区认为是空闲区域
          - alpha: 候选方向与目标方向差异的权重
          - beta: 候选方向与上一次选择方向差异的权重

        同时记录 wandb 的评价指标：
          - ave_path_length, ave_excu_time, ave_smoothness, ave_energy, ave_plan_time

        规划区域假定为 x:0~100, y:0~100, z:-100~0
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

        self.desired_speed = desired_speed

        # 极坐标直方图参数
        self.hist_resolution = np.deg2rad(hist_resolution_deg)  # 转换为弧度
        self.free_threshold = free_threshold
        self.alpha = alpha
        self.beta = beta

        # 规划区域设置
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0

        # 离线设计 LQR 控制器（更新 R 矩阵以降低控制输入激进程度）
        A = np.block([
            [np.eye(3), self.ts * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        B = np.block([
            [0.5 * (self.ts ** 2) * np.eye(3)],
            [self.ts * np.eye(3)]
        ])
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        # 将 R 的值增大，使控制器输出更平滑
        R = np.diag([10, 10, 10])
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # 用于记录上一次选择的转向角（相对于机器人前方）
        self.prev_steering = 0.0

        # 障碍地图：字典，key 为网格索引 tuple，value 为占据状态（1:障碍，0:自由）
        self.obstacle_map = {}

        super().__init__()

    # --- 坐标转换 ---
    def world_to_index(self, pos):
        # 仅用于障碍更新时将连续位置离散化
        ix = int((pos[0] - self.x_min) / self.grid_resolution)
        iy = int((pos[1] - self.y_min) / self.grid_resolution)
        iz = int((pos[2] - self.z_min) / self.grid_resolution)
        # 限制在范围内
        ix = min(max(ix, 0), int((self.x_max - self.x_min) / self.grid_resolution) - 1)
        iy = min(max(iy, 0), int((self.y_max - self.y_min) / self.grid_resolution) - 1)
        iz = min(max(iz, 0), int((self.z_max - self.z_min) / self.grid_resolution) - 1)
        return (ix, iy, iz)

    def index_to_world(self, idx):
        x = self.x_min + idx[0] * self.grid_resolution + self.grid_resolution / 2.0
        y = self.y_min + idx[1] * self.grid_resolution + self.grid_resolution / 2.0
        z = self.z_min + idx[2] * self.grid_resolution + self.grid_resolution / 2.0
        return np.array([x, y, z])

    # --- 障碍地图更新 ---
    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        sensor_readings: 长度为14的列表或数组；若读数 < 10，则视为障碍
        根据预设传感器方向（假设 agent 姿态为零）进行更新：
          取前 8 个传感器作为水平方向，其余可以忽略（或用于其他用途）
        """
        # 对于水平平面，假设 8 个激光的方向为 0,45,...,315 度
        for i in range(8):
            angle = math.radians(i * 45)
            r = sensor_readings[i]
            if r < 10.0:
                obs_pos = current_pos + r * np.array([math.cos(angle), math.sin(angle), 0])
                cell = self.world_to_index(obs_pos)
                self.obstacle_map[cell] = 1
            # 同时可沿射线采样，将近处区域标记为障碍
            num_samples = int(r / self.grid_resolution)
            for s in range(num_samples):
                sample_distance = s * self.grid_resolution
                sample_pos = current_pos + sample_distance * np.array([math.cos(angle), math.sin(angle), 0])
                cell = self.world_to_index(sample_pos)
                # 若未标记为障碍，则置为0（空闲）
                if self.obstacle_map.get(cell, 0) != 1:
                    self.obstacle_map[cell] = 0
        # 对于其他传感器（上下及倾斜），这里可选扩展处理，但本算法主要关注水平导航
        return

    # --- VFH+ 核心：计算期望转向角 ---
    def compute_vfh_direction(self, current_pos, goal_pos, sensor_readings, current_velocity):
        """
        计算 VFH+ 算法所得到的期望转向角（相对于机器人当前前方）。
        输入：
          - current_pos, goal_pos: 3D 坐标（我们主要关注水平平面 x,y）
          - sensor_readings: 长度为14的数组，取前8个作为水平激光数据
          - current_velocity: 当前速度向量（用于估计当前朝向），当速度较小时使用上一次转向
        返回：
          - desired steering angle（弧度，正值表示左转，负值表示右转），以及
          - 期望速度向量（在全局坐标系下，水平平面，保持 desired_speed）
        """
        # 估计当前朝向：如果速度足够大，则由速度决定，否则使用上一次转向
        if np.linalg.norm(current_velocity) > 0.1:
            heading = math.atan2(current_velocity[1], current_velocity[0])
        else:
            heading = self.prev_steering

        # 计算目标相对于全局的方位角
        delta = goal_pos - current_pos
        target_angle_global = math.atan2(delta[1], delta[0])
        # 目标在机器人坐标系下的期望角度
        target_angle = normalize_angle(target_angle_global - heading)

        # 构建直方图：分辨率为 hist_resolution 弧度，覆盖 [-pi, pi]
        num_bins = int(2 * np.pi / self.hist_resolution)
        histogram = np.zeros(num_bins)
        sensor_max_range = 10.0
        # 对于每个水平传感器（前8个），更新直方图
        for i in range(8):
            angle_i = math.radians(i * 45)  # 传感器安装角度，相对于机器人前方
            r = sensor_readings[i]
            if r < sensor_max_range:
                weight = (sensor_max_range - r) / sensor_max_range
                bin_index = int(normalize_angle(angle_i) / self.hist_resolution) % num_bins
                histogram[bin_index] += weight

        # 平滑直方图：简单卷积
        window_size = 3
        kernel = np.ones(window_size) / window_size
        histogram_smoothed = np.convolve(histogram, kernel, mode='same')

        # 识别空闲扇区：直方图值小于 free_threshold 的 bin 连续区域
        free_bins = (histogram_smoothed < self.free_threshold).astype(int)
        candidate_angles = []
        i = 0
        while i < num_bins:
            if free_bins[i]:
                start_bin = i
                while i < num_bins and free_bins[i]:
                    i += 1
                end_bin = i - 1
                mid_bin = (start_bin + end_bin) / 2.0
                candidate_angle = mid_bin * self.hist_resolution
                candidate_angle = normalize_angle(candidate_angle)
                candidate_angles.append(candidate_angle)
            else:
                i += 1

        if not candidate_angles:
            chosen_angle = target_angle
        else:
            costs = []
            for angle in candidate_angles:
                cost = self.alpha * abs(normalize_angle(angle - target_angle)) + \
                       self.beta * abs(normalize_angle(angle - self.prev_steering))
                costs.append(cost)
            best_idx = np.argmin(costs)
            chosen_angle = candidate_angles[best_idx]

        # 对 chosen_angle 进行低通滤波平滑，防止震荡
        smoothing_factor = 0.2  # 可调参数，值越小平滑效果越明显
        self.prev_steering = (1 - smoothing_factor) * self.prev_steering + smoothing_factor * chosen_angle
        chosen_angle = self.prev_steering

        # 生成期望速度向量（在机器人坐标系下）
        v_des_robot = self.desired_speed * np.array([math.cos(chosen_angle), math.sin(chosen_angle), 0])
        # 转换到全局坐标系：旋转 by heading
        c = math.cos(heading)
        s = math.sin(heading)
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        v_des_global = R @ v_des_robot
        return chosen_angle, v_des_global

    # --- 路径平滑（带边界约束） ---
    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        if path is None or len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        # 对各坐标进行边界约束，防止超出预定区域
        smooth_path[:, 0] = np.clip(smooth_path[:, 0], self.x_min, self.x_max)
        smooth_path[:, 1] = np.clip(smooth_path[:, 1], self.y_min, self.y_max)
        smooth_path[:, 2] = np.clip(smooth_path[:, 2], self.z_min, self.z_max)
        return [pt for pt in smooth_path]

    # --- 主训练及规划循环 ---
    def train(self, env, num_episodes=10):
        wandb.init(project="auv_VFHPlus_planning", name="VFHPlus_run")
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
            "desired_speed": self.desired_speed,
            "hist_resolution_deg": np.rad2deg(self.hist_resolution),
            "free_threshold": self.free_threshold,
            "alpha": self.alpha,
            "beta": self.beta
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            logging.info(f"VFH+ Episode {episode + 1} starting")
            env.reset()

            # 初始状态
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

            while step_count < self.max_steps:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # 更新障碍地图（利用传感器数据）
                sensor_readings = env.lasers.copy()
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # 计算期望转向并获得目标速度
                v_current = env.velocity.copy()
                steering_angle, v_des = self.compute_vfh_direction(current_pos, goal_pos, sensor_readings, v_current)

                # 设定短期目标位置：当前位置 + v_des * ts
                pos_des = current_pos + v_des * 20

                # 构造状态：[位置, 速度]
                x_current = np.hstack([current_pos, v_current])
                x_des = np.hstack([pos_des, v_des])
                error_state = x_current - x_des
                # LQR 控制计算控制输入
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                print("action is:", u)
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

                # 简单碰撞检测
                for obs in env.obstacles:
                    if np.linalg.norm(new_pos - np.array(obs)) < self.collision_threshold:
                        collisions += 1
                        break
                env.env.draw_line(current_pos.tolist(), new_pos.tolist(), color=[30, 50, 0],
                                  thickness=5, lifetime=0)
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
            self.ave_path_length += total_path_length
            self.ave_excu_time += episode_running_duration
            self.ave_plan_time += (time.time() - episode_start_time)
            self.ave_smoothness += smoothness
            self.ave_energy += energy

            logging.info(
                f"Episode {episode + 1} completed - Path Length: {total_path_length}, Steps: {step_count}, Collisions: {collisions}")
            episode += 1

            if reach_target_count > 9 or episode == num_episodes:
                wandb.log({
                    "ave_path_length": self.ave_path_length / reach_target_count,
                    "ave_excu_time": self.ave_excu_time / reach_target_count,
                    "ave_plan_time": self.ave_plan_time / reach_target_count,
                    "ave_smoothness": self.ave_smoothness / reach_target_count,
                    "ave_energy": self.ave_energy / reach_target_count
                })
                print(f"ave_path_length: {self.ave_path_length / reach_target_count}")
                print(f"ave_excu_time: {self.ave_excu_time / reach_target_count}")
                print(f"ave_plan_time: {self.ave_plan_time / reach_target_count}")
                print(f"ave_smoothness: {self.ave_smoothness / reach_target_count}")
                print(f"ave_energy: {self.ave_energy / reach_target_count}")
                env.set_current_target(env.choose_next_target())

        logging.info("VFH+ Planning finished training.")
        return