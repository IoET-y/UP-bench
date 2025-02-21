import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner


class LPAStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.5, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100):
        """
        参数说明：
          - grid_resolution: 离散化网格分辨率（与环境单位一致，调大可降低计算量）
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率

        同时记录评价指标：
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

        # LPA* 相关数据结构
        # 采用正向搜索：以起点为源、目标为终点
        self.g = {}  # 字典，key 为网格索引 tuple，表示各节点实际代价
        self.rhs = {}  # one–step lookahead 值
        self.U = []  # 优先级队列，元素格式为 (key, node_index)
        # 注意：本算法采用懒删除，重复项不影响正确性

        # 障碍地图：字典，key 为网格索引，value 为占据状态（1:障碍, 0:自由）
        self.obstacle_map = {}

        # 预先计算26个邻域偏移量，避免重复三重循环
        self.neighbor_shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.neighbor_shifts.append((dx, dy, dz))

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

    # --- 邻域函数 ---
    def get_neighbors(self, idx):
        neighbors = []
        for shift in self.neighbor_shifts:
            n_idx = (idx[0] + shift[0], idx[1] + shift[1], idx[2] + shift[2])
            if 0 <= n_idx[0] < self.nx and 0 <= n_idx[1] < self.ny and 0 <= n_idx[2] < self.nz:
                neighbors.append(n_idx)
        return neighbors

    # --- 启发函数与代价 ---
    def heuristic(self, s, goal):
        return self.grid_resolution * math.sqrt((s[0] - goal[0]) ** 2 + (s[1] - goal[1]) ** 2 + (s[2] - goal[2]) ** 2)

    def cost(self, a, b):
        if self.obstacle_map.get(a, 0) == 1 or self.obstacle_map.get(b, 0) == 1:
            return float('inf')
        return self.grid_resolution * math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def calculate_key(self, s, goal):
        g_val = self.g.get(s, float('inf'))
        rhs_val = self.rhs.get(s, float('inf'))
        return (min(g_val, rhs_val) + self.heuristic(s, goal), min(g_val, rhs_val))

    # --- LPA* 更新函数 ---
    def update_vertex(self, s, start, goal):
        # 对于除起点以外的节点，更新 rhs(s)= min_{s' in pred(s)} (g(s') + cost(s', s))
        if s != start:
            min_val = float('inf')
            for s_pred in self.get_neighbors(s):
                # 因为图无向，对 s 的所有邻居都作为前驱
                min_val = min(min_val, self.g.get(s_pred, float('inf')) + self.cost(s_pred, s))
            self.rhs[s] = min_val
        # 如果 s 在 U 中，不必主动删除（采用懒删除）
        if abs(self.g.get(s, float('inf')) - self.rhs.get(s, float('inf'))) > 1e-5:
            heapq.heappush(self.U, (self.calculate_key(s, goal), s))
        # 否则保持 s 一致时不入队

    def compute_shortest_path(self, start, goal):
        """不断更新直到目标（goal）满足一致性条件"""
        while self.U:
            top_key, u = self.U[0]
            if top_key >= self.calculate_key(goal, goal) and self.g.get(goal, float('inf')) == self.rhs.get(goal,
                                                                                                            float('inf')):
                break
            heapq.heappop(self.U)
            current_key = self.calculate_key(u, goal)
            # 若队列中的key已过时，则重新入队
            if top_key < current_key:
                heapq.heappush(self.U, (current_key, u))
                continue
            g_u = self.g.get(u, float('inf'))
            rhs_u = self.rhs.get(u, float('inf'))
            if g_u > rhs_u:
                self.g[u] = rhs_u
                for s in self.get_neighbors(u):
                    self.update_vertex(s, start, goal)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u, start, goal)
                for s in self.get_neighbors(u):
                    self.update_vertex(s, start, goal)

    def get_path(self, start, goal):
        """
        从起点沿使 (cost + g) 最小的邻居构造路径，直至到达目标。
        若无路径，则返回 None。
        """
        path = [start]
        current = start
        while current != goal:
            min_val = float('inf')
            next_node = None
            for s in self.get_neighbors(current):
                val = self.cost(current, s) + self.g.get(s, float('inf'))
                if val < min_val:
                    min_val = val
                    next_node = s
            if next_node is None or next_node == current or min_val == float('inf'):
                return None
            current = next_node
            path.append(current)
        world_path = [self.index_to_world(idx) for idx in path]
        return world_path

    def plan_path(self, start_pos, goal_pos):
        """
        根据当前已知障碍地图，初始化 LPA* 数据结构并计算从 start 到 goal 的全局路径。
        """
        start = self.world_to_index(start_pos)
        goal = self.world_to_index(goal_pos)
        # 重置 g 与 rhs 表
        self.g = {}
        self.rhs = {}
        self.U = []
        # 对所有节点初始值为infty（采用 lazy 初始化方式）
        self.g[start] = float('inf')
        self.rhs[start] = 0.0
        heapq.heappush(self.U, (self.calculate_key(start, goal), start))
        self.compute_shortest_path(start, goal)
        return self.get_path(start, goal)

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        if path is None or len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        根据当前传感器读数更新障碍地图：
          - sensor_readings: 长度为14的列表或数组，若读数 < 10 则视为障碍
          - 预设传感器相对方向（假设agent姿态为零），将读数转换为世界坐标，再转换为网格索引进行标记
          同时沿射线采样，将视野内其他单元标记为自由（0）
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
        # 注意：本实现中，因采用懒更新，启发值将在下一次搜索中重新计算
        return updated_cells

    def train(self, env, num_episodes=10):
        """
        利用 LPA* 实时感知算法进行路径规划，并结合离线设计的 LQR 控制器跟踪规划路径。
        流程：
          1. 重置环境，获取起点 (env.location) 和目标 (env.get_current_target())。
          2. 初始全局规划：利用当前障碍地图（初始为空）计算全局路径。
          3. 在跟踪过程中，每个控制周期更新障碍地图、调用 LPA* 更新局部路径，
             再利用 LQR 控制器生成动作跟踪目标。
          4. 记录 evaluation metrics，并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_LPAStar_planning", name="LPAStar_run")
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
            print("Now episode:", episode + 1, "Reach target count:", reach_target_count)
            episode_start_time = time.time()
            logging.info(f"LPA* Episode {episode + 1} starting")
            env.reset()

            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  # 3D起点
            target = env.get_current_target()
            goal_pos = np.array(target)  # 3D目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 每个 episode 重置障碍地图与 LPA* 数据结构
            self.obstacle_map = {}
            self.g = {}
            self.rhs = {}
            self.U = {}
            # 重置网格尺寸参数
            self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
            self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
            self.nz = int((self.z_max - self.z_min) / self.grid_resolution)

            # 初始全局规划
            path = self.plan_path(start_pos, goal_pos)
            print("Got an initial path.")
            if path is None:
                logging.info("LPA* did not find an initial path.")
                episode += 1
                continue

            # 对规划路径进行平滑处理以便于显示
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)
            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            step_count = 0
            total_path_length = 0.0
            collisions = 0
            energy = 0.0
            smoothness = 0.0
            prev_u = None
            current_pos = start_pos.copy()
            path_idx = 0
            max_steps_episode = self.max_steps
            episode_planning_duration = time.time() - episode_start_time

            episode_start_running_time = time.time()
            print("Following the path with LQR control...")
            # 跟踪控制循环：每个控制周期更新障碍地图、重新规划局部路径，然后利用 LQR 跟踪
            while step_count < max_steps_episode:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                sensor_readings = env.lasers.copy()  # 14个传感器读数
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # LPA* 重规划：利用当前障碍地图更新数据结构
                current_idx = self.world_to_index(current_pos)
                goal_idx = self.world_to_index(goal_pos)
                # 为当前环境状态重新初始化 LPA*（这里直接以全局规划方式更新，可进一步增量更新）
                path_indices = self.plan_path(current_pos, goal_pos)
                if path_indices is not None:
                    new_path = path_indices
                    path = self.smooth_path(new_path, smoothing_factor=1.0, num_points=200)
                    for i in range(len(path) - 1):
                        env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5,
                                          lifetime=0)
                else:
                    logging.info("No valid path found, stopping episode.")
                    break

                if path_idx >= len(path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[path_idx]
                    if path_idx < len(path) - 1:
                        desired_speed = 3  # [m/s]
                        direction = path[path_idx + 1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < 1:
                        path_idx += 1
                        continue

                # 生成 LQR 控制输入
                x_current = np.hstack([current_pos, env.velocity.copy()])
                x_des = np.hstack([waypoint, v_des])
                error_state = x_current - x_des
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                action = np.concatenate([u, np.zeros(3)])
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()

                distance_moved = np.linalg.norm(new_pos - current_pos)
                total_path_length += distance_moved
                energy += (np.linalg.norm(u) ** 2) / 100
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

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
                    "distance_to_waypoint": np.linalg.norm(current_pos - waypoint),
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
                "episode_planning_duration": episode_planning_duration,
                "episode_running_duration": episode_running_duration
            })
            self.ave_path_length += total_path_length
            self.ave_excu_time += episode_planning_duration
            self.ave_plan_time += episode_running_duration
            self.ave_smoothness += smoothness
            self.ave_energy += energy

            logging.info(
                f"Episode {episode + 1} completed - Path Length: {total_path_length}, Steps: {step_count}, Collisions: {collisions}")
            episode += 1
            if reach_target_count >= 10 or episode >= num_episodes:
                wandb.log({
                    "ave_path_length": self.ave_path_length / reach_target_count,
                    "ave_excu_time": self.ave_excu_time / reach_target_count,
                    "ave_plan_time": self.ave_plan_time / reach_target_count,
                    "ave_smoothness": self.ave_smoothness / reach_target_count,
                    "ave_energy": self.ave_energy / reach_target_count
                })
                print(f'ave_path_length: {self.ave_path_length / reach_target_count}')
                print(f'ave_excu_time: {self.ave_excu_time / reach_target_count}')
                print(f'ave_plan_time: {self.ave_plan_time / reach_target_count}')
                print(f'ave_smoothness: {self.ave_smoothness / reach_target_count}')
                print(f'ave_energy: {self.ave_energy / reach_target_count}')
                return
            env.set_current_target(env.choose_next_target())

        logging.info("LPA* Planning finished training.")
        return