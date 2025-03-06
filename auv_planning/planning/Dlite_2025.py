import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner

class DStarLitePlanner(BasePlanner):
    def __init__(self, grid_resolution=0.5, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100):
        """
        参数说明：
          - grid_resolution: 离散化网格的分辨率（单位与环境一致，此处推荐调大以避免计算量过大）
          - max_steps: 每个episode允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率
        """
        # EVALUATION MATRICS
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
        self.ts = 1.0 / self.ticks_per_sec  # 离散时间步长
        self.current_time = 0.0

        # 规划区域设置：假定区域：x:0~100, y:0~100, z:-100~0
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0
        # 网格尺寸（用于判断索引是否在边界内）
        self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
        self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
        self.nz = int((self.z_max - self.z_min) / self.grid_resolution)

        # 离线设计LQR控制器（与A*代码一致）
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
        # -------------------------------

        # D* Lite 相关数据结构
        # g: 从目标反向规划时各节点的代价
        # rhs: one-step lookahead 值
        self.g = {}     # 字典，key 为网格索引（tuple），value 为代价
        self.rhs = {}   # 同上
        self.U = []     # 优先级队列，元素为 (key, cell_index)
        self.km = 0.0   # 用于处理start移动时的偏差
        self.last = None  # 记录上一次机器人的网格位置

        # 障碍地图：字典，key 为网格索引，value 为占据状态（1:障碍，0:自由）
        self.obstacle_map = {}

        # -------------------------------
        # 预先计算26个邻域偏移量，避免每次计算三重循环
        self.neighbor_shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.neighbor_shifts.append((dx, dy, dz))

        super().__init__()

    # --- 网格坐标转换函数 ---
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

    # --- D* Lite 核心函数 ---
    def get_neighbors(self, idx):
        """返回当前网格单元idx的所有26邻域（在边界内的）"""
        neighbors = []
        for shift in self.neighbor_shifts:
            n_idx = (idx[0] + shift[0], idx[1] + shift[1], idx[2] + shift[2])
            if 0 <= n_idx[0] < self.nx and 0 <= n_idx[1] < self.ny and 0 <= n_idx[2] < self.nz:
                neighbors.append(n_idx)
        return neighbors

    def heuristic(self, a, b):
        """欧式距离启发式，注意乘上分辨率换算到实际距离"""
        return self.grid_resolution * math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def cost(self, a, b):
        """
        返回从 a 到 b 的转移代价：
          - 如果 a 或 b 被标记为障碍，则代价为正无穷
          - 否则为欧式距离（乘以网格分辨率）
        """
        if self.obstacle_map.get(a, 0) == 1 or self.obstacle_map.get(b, 0) == 1:
            return float('inf')
        return self.grid_resolution * math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def calculate_key(self, s):
        g_s = self.g.get(s, float('inf'))
        rhs_s = self.rhs.get(s, float('inf'))
        key_first = min(g_s, rhs_s) + self.heuristic(self.start, s) + self.km
        key_second = min(g_s, rhs_s)
        return (key_first, key_second)

    def update_vertex(self, s):
        if s != self.goal:
            min_val = float('inf')
            # 缓存局部变量，减少函数调用开销
            get_nbr = self.get_neighbors
            get_g = self.g.get
            for s_prime in get_nbr(s):
                min_val = min(min_val, self.cost(s, s_prime) + get_g(s_prime, float('inf')))
            self.rhs[s] = min_val
        # 如果不一致则将 s 加入队列（懒删除：队列中可能有重复项）
        if abs(self.g.get(s, float('inf')) - self.rhs.get(s, float('inf'))) > 1e-5:
            heapq.heappush(self.U, (self.calculate_key(s), s))

    def compute_shortest_path(self):
        """
        根据 D* Lite 算法，不断更新直到起点的最优值收敛。
        为防止死循环，增加了最大迭代次数的限制。
        """
        max_iterations = 100000
        iterations = 0
        while self.U and (
            self.U[0][0] < self.calculate_key(self.start) or
            abs(self.rhs.get(self.start, float('inf')) - self.g.get(self.start, float('inf'))) > 1e-5
        ):
            iterations += 1
            if iterations % 1000 == 0:
                logging.info(f"compute_shortest_path iteration {iterations}, U size: {len(self.U)}")
            if iterations > max_iterations:
                logging.warning("compute_shortest_path reached maximum iterations, breaking out.")
                break

            k_old, u = heapq.heappop(self.U)
            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

    def get_path(self):
        """
        根据当前计算的 g 值，从 start 出发选择代价最小的邻居构造路径，直至到达 goal
        """
        path = []
        current = self.start
        path.append(current)
        # 限制最大步数，防止死循环
        max_steps_path = self.max_steps * 2
        steps = 0
        while current != self.goal and steps < max_steps_path:
            min_val = float('inf')
            next_node = None
            for s in self.get_neighbors(current):
                c = self.cost(current, s) + self.g.get(s, float('inf'))
                if c < min_val:
                    min_val = c
                    next_node = s
            if next_node is None or next_node == current:
                logging.warning("get_path: 无法继续寻找下一个节点，可能路径中断")
                return None
            current = next_node
            path.append(current)
            steps += 1
        if steps >= max_steps_path:
            logging.warning("get_path: 超过最大步数，路径可能不正确")
            return None
        # 将网格索引转换为连续世界坐标（取单元格中心）
        world_path = [self.index_to_world(idx) for idx in path]
        return world_path

    # --- 利用传感器更新障碍地图 ---
    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        根据当前的传感器读数更新障碍地图：
          - sensor_readings：长度为14的列表或数组，若读数 < 10，则认为该方向有障碍
          - 利用预设的传感器方向列表，将相对距离转换为世界坐标，再转换为网格索引进行标记
        同时沿射线方向采样，将视野内的其他单元格标记为自由（0）
        """
        # 定义传感器相对方向（假设 agent 姿态为零）
        directions = []
        for i in range(8):
            angle = math.radians(i * 45)
            directions.append(np.array([math.cos(angle), math.sin(angle), 0]))
        directions.append(np.array([0, 0, 1]))
        directions.append(np.array([0, 0, -1]))
        directions.append(np.array([math.cos(math.radians(45)), 0, math.sin(math.radians(45))]))
        directions.append(np.array([0, math.cos(math.radians(45)), math.sin(math.radians(45))]))
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
                # 标记为自由
                self.obstacle_map[sample_cell] = 0
        for cell in updated_cells:
            for neighbor in self.get_neighbors(cell):
                self.update_vertex(neighbor)
        return updated_cells

    def plan_path(self, start, goal):
        """
        根据当前已知障碍地图初始化 D* Lite 数据结构，并计算从 start 到 goal 的路径。
        注意：D* Lite 算法采用反向规划，即将 goal 作为源点。
        """
        self.start = self.world_to_index(start)
        self.goal = self.world_to_index(goal)
        self.last = self.start
        self.km = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        # 对 goal 初始化：g[goal] = inf, rhs[goal] = 0
        self.g[self.goal] = float('inf')
        self.rhs[self.goal] = 0.0
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))
        self.compute_shortest_path()
        path = self.get_path()
        return path

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
        对离散路径使用样条插值进行平滑处理
        """
        if path is None or len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    # --- 主训练及规划循环 ---
    def train(self, env, num_episodes=10):
        """
        利用 D* Lite 实时感知算法进行路径规划，并结合离线设计的 LQR 控制器跟踪规划路径。
        流程：
          1. 重置环境，获取起点（env.location）和目标（env.get_current_target()）。
          2. 利用当前已知障碍（初始为空）规划路径。
          3. 在跟踪过程中，每个周期更新传感器信息，更新障碍地图，并调用 compute_shortest_path 重新规划。
          4. 生成动作 u = -K (x - x_des) ，并限制在最大线性加速度内，同时角加速度置0。
          5. 记录evaluation metrics，并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_DStarLite_planning", name="DStarLite_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max],
            }
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            print("now is episode: ", episode+1, " and reach target count: ", reach_target_count)
            episode_start_time = time.time()
            logging.info(f"D* Lite Episode {episode+1} starting")
            env.reset()

            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()    # 3D起点
            target = env.get_current_target()
            goal_pos = np.array(target)          # 3D目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 每个episode重置障碍地图
            self.obstacle_map = {}

            # 重置网格尺寸参数（基于当前规划区域设置）
            self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
            self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
            self.nz = int((self.z_max - self.z_min) / self.grid_resolution)
            print("before get a path")

            # 初始规划：由于障碍地图为空，得到一条初始路径
            path = self.plan_path(start_pos, goal_pos)
            print("get a path")
            if path is None:
                logging.info("D* Lite did not find an initial path.")
                episode += 1
                continue

            # 对规划路径进行平滑处理以便于视觉展示
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)
            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[30,50,0], thickness=5, lifetime=0)

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
            print("follow the path with LQR ")
            # 跟踪控制循环：在每个控制周期内更新障碍地图并重新规划
            while step_count < max_steps_episode:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                sensor_readings = env.lasers.copy()  # 14个读数
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                self.km += self.heuristic(self.last, self.world_to_index(current_pos)) if self.last is not None else 0
                self.last = self.world_to_index(current_pos)
                self.start = self.world_to_index(current_pos)
                self.compute_shortest_path()
                new_path = self.get_path()
                if new_path is not None:
                    path = self.smooth_path(new_path, smoothing_factor=1.0, num_points=200)
                    for i in range(len(path) - 1):
                        env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[30,50,0], thickness=5, lifetime=0)
                else:
                    logging.info("No valid path found, stopping episode.")
                    break

                if path_idx >= len(path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[path_idx]
                    if path_idx < len(path) - 1:
                        desired_speed = 3
                        direction = path[path_idx+1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < 1:
                        path_idx += 1
                        continue

                print("follow done in the episode")
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
                energy += np.linalg.norm(u) ** 2/100
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
                    "distance_to_goal": np.linalg.norm(current_pos - goal_pos),
                })

            episode_running_duration = time.time() - episode_start_running_time
            wandb.log({
                "episode": episode+1,
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

            logging.info(f"Episode {episode+1} completed - Path Length: {total_path_length}, Steps: {step_count}, Collisions: {collisions}")
            episode += 1
            if reach_target_count >= 10 or episode >= num_episodes:
                wandb.log({
                    "ave_path_length": self.ave_path_length / reach_target_count,
                    "ave_excu_time": self.ave_excu_time / reach_target_count,
                    "ave_plan_time": self.ave_plan_time / reach_target_count,
                    "ave_smoothness": self.ave_smoothness / reach_target_count,
                    "ave_energy": self.ave_energy / reach_target_count
                })
                print(f'ave_path_length is: {self.ave_path_length / reach_target_count}')
                print(f'ave_excu_time is: {self.ave_excu_time / reach_target_count}')
                print(f'ave_plan_time is: {self.ave_plan_time / reach_target_count}')
                print(f'ave_smoothness is: {self.ave_smoothness / reach_target_count}')
                print(f'ave_energy is: {self.ave_energy / reach_target_count}')
                return
            env.set_current_target(env.choose_next_target())

        logging.info("D* Lite Planning finished training.")
        return