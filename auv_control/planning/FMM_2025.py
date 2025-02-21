import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner

# 状态常量：Far=0, Narrow=1, Accepted=2
FAR = 0
NARROW = 1
ACCEPTED = 2


class FMMPlanner(BasePlanner):
    def __init__(self, grid_resolution=1.0, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 k_att=1.0, k_rep=100.0, d0=10.0, desired_speed=3.0):
        """
        参数说明：
          - grid_resolution: 网格分辨率（单位与环境一致）；建议较粗（如1.0米），以减少计算量
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率
          - k_att, k_rep, d0, desired_speed: （本 FMM 算法主要用 LQR 跟踪，以下参数可供参考）
            此处可保留（后续如与其他方法混合时可能使用）

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

        # 规划区域设置
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0
        self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
        self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
        self.nz = int((self.z_max - self.z_min) / self.grid_resolution)

        # 离线设计 LQR 控制器（与之前一致）
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

        # 潜在参数（本方法主要依赖 FMM 求 T 场）
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.desired_speed = desired_speed

        # 障碍地图：字典，key 为网格索引 tuple，value 为占据状态（1:障碍，0:自由）
        self.obstacle_map = {}

        super().__init__()

    # --- 网格索引与世界坐标转换 ---
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

    # --- 6邻域（正交邻域）函数 ---
    def get_6_neighbors(self, idx):
        neighbors = []
        ix, iy, iz = idx
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nx_ = ix + dx
            ny_ = iy + dy
            nz_ = iz + dz
            if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny and 0 <= nz_ < self.nz:
                neighbors.append((nx_, ny_, nz_))
        return neighbors

    # --- FMM 更新公式 ---
    def solve_eikonal(self, accepted_vals):
        """
        根据已接受邻居的 T 值计算当前格点的更新值，假设均匀网格，步长 h = grid_resolution。
        对于二维公式：如果只有1个值，则 T = a + h；若有2个值，则
           T = (a + b + sqrt(2h^2 - (b-a)^2))/2 ；对于3D采用类似扩展（简化实现）。
        """
        h = self.grid_resolution
        vals = sorted(accepted_vals)
        T_val = vals[0] + h
        if len(vals) >= 2:
            if T_val > vals[1]:
                disc = 2 * h * h - (vals[1] - vals[0]) ** 2
                if disc < 0:
                    disc = 0
                T_val = (vals[0] + vals[1] + math.sqrt(disc)) / 2.0
        if len(vals) >= 3:
            if T_val > vals[2]:
                disc = 3 * h * h - ((vals[2] - vals[0]) ** 2 + (vals[2] - vals[1]) ** 2 + (vals[1] - vals[0]) ** 2)
                if disc < 0:
                    disc = 0
                T_val = (vals[0] + vals[1] + vals[2] + math.sqrt(disc)) / 3.0
        return T_val

    # --- FMM 主过程 ---
    def run_fmm(self, start_idx, goal_idx):
        """
        利用 FMM 从目标处（goal_idx）反向计算“到达时间” T，
        直到起点（start_idx）被接受（或全部区域处理完）。
        返回 T 数组（形状为 (nx,ny,nz)）和状态数组（0:Far,1:Narrow,2:Accepted）。
        """
        # 初始化 T 与状态数组
        T = np.full((self.nx, self.ny, self.nz), np.inf)
        status = np.zeros((self.nx, self.ny, self.nz), dtype=np.int8)
        heap = []
        # 目标点
        gi, gj, gk = goal_idx
        T[gi, gj, gk] = 0.0
        status[gi, gj, gk] = ACCEPTED
        # 将目标的6邻域加入窄带
        for nb in self.get_6_neighbors(goal_idx):
            if self.obstacle_map.get(nb, 0) == 1:
                continue
            ni, nj, nk = nb
            # 初始估计：T = T(goal) + h
            T[ni, nj, nk] = self.grid_resolution
            status[ni, nj, nk] = NARROW
            heapq.heappush(heap, (T[ni, nj, nk], nb))
        # 运行 FMM：直到起点被接受或窄带为空
        si, sj, sk = start_idx
        while heap:
            t_val, cell = heapq.heappop(heap)
            ci, cj, ck = cell
            if status[ci, cj, ck] == ACCEPTED:
                continue
            status[ci, cj, ck] = ACCEPTED
            # 若已到达起点，则停止
            if cell == start_idx:
                break
            for nb in self.get_6_neighbors(cell):
                if self.obstacle_map.get(nb, 0) == 1:
                    continue
                ni, nj, nk = nb
                # 收集 nb 的所有已接受邻居 T 值
                accepted_vals = []
                for nnb in self.get_6_neighbors(nb):
                    ii, jj, kk = nnb
                    if status[ii, jj, kk] == ACCEPTED:
                        accepted_vals.append(T[ii, jj, kk])
                if not accepted_vals:
                    continue
                T_new = self.solve_eikonal(accepted_vals)
                if T_new < T[ni, nj, nk]:
                    T[ni, nj, nk] = T_new
                    heapq.heappush(heap, (T_new, nb))
                    if status[ni, nj, nk] == FAR:
                        status[ni, nj, nk] = NARROW
        return T, status

    # --- 提取路径 ---
    def extract_path(self, T, start_idx, goal_idx):
        """
        从起点出发，沿 T 值梯度下降（选取6邻域中 T 最小的）提取一条路径，
        直到达到目标（T接近0）或达到最大步数。
        返回路径（以网格索引列表表示），若未找到则返回 None。
        """
        path_idx = []
        current = start_idx
        path_idx.append(current)
        max_iter = 10000
        iter_cnt = 0
        while current != goal_idx and iter_cnt < max_iter:
            neighbors = self.get_6_neighbors(current)
            # 选取 T 最小的邻居
            min_T = T[current[0], current[1], current[2]]
            next_cell = current
            for nb in neighbors:
                if T[nb[0], nb[1], nb[2]] < min_T:
                    min_T = T[nb[0], nb[1], nb[2]]
                    next_cell = nb
            # 若无法下降，则跳出
            if next_cell == current:
                break
            current = next_cell
            path_idx.append(current)
            iter_cnt += 1
        if current != goal_idx:
            return None
        # 转换为连续坐标
        path = [self.index_to_world(idx) for idx in path_idx]
        return path

    # --- 路径平滑（并进行边界约束） ---
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

    # --- 利用传感器更新障碍地图 ---
    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        sensor_readings: 长度为14的列表或数组；若读数 < 10，则视为障碍
        根据预设传感器方向（假设 agent 姿态为零），将测量转换为世界坐标，再转换为网格索引进行标记；
        同时沿射线采样，将视野内其他单元标记为空闲（0）
        """
        directions = []
        # 8个水平激光（角度 0,45,...,315 度）
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
                obs_pos = current_pos + reading * direction
                cell = self.world_to_index(obs_pos)
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

    # --- 主训练与规划循环 ---
    def train(self, env, num_episodes=10):
        wandb.init(project="auv_FMM_planning", name="FMM_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max]
            }
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            logging.info(f"FMM Episode {episode + 1} starting")
            env.reset()

            # 用零动作获取初始状态
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

            # 在每个控制周期内：更新障碍地图、运行 FMM 并提取路径，然后利用 LQR 控制跟踪
            while step_count < self.max_steps:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # 更新障碍地图（利用 14 个 RangeFinderSensor）
                sensor_readings = env.lasers.copy()
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # 计算 FMM，从目标开始向外传播
                start_idx = self.world_to_index(current_pos)
                goal_idx = self.world_to_index(goal_pos)
                T, status = self.run_fmm(start_idx, goal_idx)
                path = self.extract_path(T, start_idx, goal_idx)
                if path is None:
                    logging.info("No valid path found, stopping episode.")
                    break

                # 对路径平滑处理，并进行边界约束
                path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)
                for i in range(len(path) - 1):
                    env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5,
                                      lifetime=0)

                # 将路径作为参考轨迹，取路径首段作为局部目标
                if len(path) < 2:
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[1]
                    direction = waypoint - current_pos
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-6:
                        direction /= norm_dir
                    v_des = self.desired_speed * direction

                # 构造当前状态和期望状态（状态包含位置和速度）
                x_current = np.hstack([current_pos, env.velocity.copy()])
                x_des = np.hstack([current_pos + v_des * self.ts, v_des])
                error_state = x_current - x_des

                # LQR 控制生成控制输入
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                action = np.concatenate([u, np.zeros(3)])
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()
                env.env.draw_line(current_pos.tolist(), new_pos.tolist(), color=[30,100,0], thickness=3, lifetime=0)

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

            if reach_target_count > 9 or episode >= num_episodes:
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

        logging.info("FMM Planning finished training.")
        return