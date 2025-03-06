import numpy as np
import math
import random
import time
import heapq
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

# 假设 BasePlanner 在同目录下的 base 模块中定义
from .base import BasePlanner

class ACOPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0,
                 num_ants=50, iterations=100,
                 alpha=1.0, beta=4.0, evaporation_rate=0.1, Q=100.0,
                 max_path_steps=500):
        """
        参数说明：
          - grid_resolution: 离散化网格分辨率（与环境一致）
          - max_steps: 每个 episode 允许的最大控制步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - num_ants: 每次迭代中蚂蚁个数
          - iterations: ACO 迭代次数
          - alpha: 信息素权重指数
          - beta: 启发式权重指数（启发式通常选 1/distance_to_goal）
          - evaporation_rate: 每次迭代信息素蒸发比例
          - Q: 信息素沉积常数
          - max_path_steps: 每个蚂蚁路径允许的最大步数（防止死循环）
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

        # ACO参数
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.max_path_steps = max_path_steps

        self.ticks_per_sec = 100
        self.ts = 1.0 / self.ticks_per_sec  # 离散时间步长
        self.current_time = 0.0

        # 规划区域设置：例如 x:0~100, y:0~100, z:-100~0
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0

        # 障碍物安全半径（用于膨胀障碍物区域）
        self.obstacle_radius = 5

        # 定义26个邻域方向（3D全邻域）
        self.neighbor_shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.neighbor_shifts.append((dx, dy, dz))
        self.num_neighbors = len(self.neighbor_shifts)  # 26

        # -------------------------------
        # 离线设计LQR控制器
        # 假设agent的动力学为双积分模型：
        #   p_{k+1} = p_k + dt * v_k + 0.5 * dt^2 * u_k
        #   v_{k+1} = v_k + dt * u_k
        # 状态 x = [p; v] \in R^6, 控制 u \in R^3.
        I3 = np.eye(3)
        A = np.block([
            [np.eye(3), self.ts * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        B = np.block([
            [0.5 * (self.ts ** 2) * np.eye(3)],
            [self.ts * np.eye(3)]
        ])
        # 调整代价矩阵：提高速度误差权重，使控制更平滑，同时适当增加控制代价
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1, 0.1])
        # 求解离散Riccati方程
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        # -------------------------------

        super().__init__()

    def world_to_index(self, pos):
        """
        将连续世界坐标 pos = [x, y, z] 转换为网格索引 (ix, iy, iz)
        """
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
        """
        将网格索引 (ix, iy, iz) 转换为连续世界坐标（取单元格中心）
        """
        x = self.x_min + idx[0] * self.grid_resolution + self.grid_resolution / 2.0
        y = self.y_min + idx[1] * self.grid_resolution + self.grid_resolution / 2.0
        z = self.z_min + idx[2] * self.grid_resolution + self.grid_resolution / 2.0
        return np.array([x, y, z])

    def create_obstacle_grid(self, obstacles):
        """
        根据 obstacles 列表构建3D网格地图：0表示空闲，1表示障碍区域
        仅对在规划区域内的障碍物进行标记
        """
        nx = int((self.x_max - self.x_min) / self.grid_resolution)
        ny = int((self.y_max - self.y_min) / self.grid_resolution)
        nz = int((self.z_max - self.z_min) / self.grid_resolution)
        grid = np.zeros((nx, ny, nz), dtype=int)
        for obs in obstacles:
            if not (self.x_min <= obs[0] <= self.x_max and
                    self.y_min <= obs[1] <= self.y_max and
                    self.z_min <= obs[2] <= self.z_max):
                continue
            obs_idx = self.world_to_index(obs)
            # 将障碍物“膨胀”一定半径
            radius_in_cells = int(math.ceil(self.obstacle_radius / self.grid_resolution))
            for i in range(max(0, obs_idx[0]-radius_in_cells), min(nx, obs_idx[0]+radius_in_cells+1)):
                for j in range(max(0, obs_idx[1]-radius_in_cells), min(ny, obs_idx[1]+radius_in_cells+1)):
                    for k in range(max(0, obs_idx[2]-radius_in_cells), min(nz, obs_idx[2]+radius_in_cells+1)):
                        cell_center = self.index_to_world((i, j, k))
                        if np.linalg.norm(cell_center - np.array(obs)) <= self.obstacle_radius:
                            grid[i, j, k] = 1
        return grid

    def plan_path(self, start, goal, obstacles):
        """
        使用 ACO 算法规划从 start 到 goal 的路径
        参数：
          - start: 起点 [x, y, z]
          - goal: 目标点 [x, y, z]
          - obstacles: 障碍物列表，每个为 [x, y, z]
        返回：
          - path: 连续坐标点构成的路径列表（每个为 np.array([x, y, z])）；若规划失败返回 None
        """
        grid = self.create_obstacle_grid(obstacles)
        nx, ny, nz = grid.shape

        start_idx = self.world_to_index(start)
        goal_idx = self.world_to_index(goal)

        # 若起点或目标在障碍内，则直接返回 None
        if grid[start_idx] == 1 or grid[goal_idx] == 1:
            logging.info("start or end in obstacle area。")
            return None

        # 初始化信息素：为每个网格点的每个邻域方向赋予初始信息素值
        pheromone = np.ones((nx, ny, nz, self.num_neighbors), dtype=np.float64)

        best_path = None
        best_cost = np.inf

        # -------------------------------
        # 向量化计算启发式信息（欧氏距离）
        goal_center = self.index_to_world(goal_idx)
        x_coords = self.x_min + (np.arange(nx) + 0.5) * self.grid_resolution
        y_coords = self.y_min + (np.arange(ny) + 0.5) * self.grid_resolution
        z_coords = self.z_min + (np.arange(nz) + 0.5) * self.grid_resolution
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        heuristic = np.sqrt((X - goal_center[0])**2 + (Y - goal_center[1])**2 + (Z - goal_center[2])**2)
        heuristic[heuristic < 1e-6] = 1e-6
        # -------------------------------

        # 早停参数：如果连续若干次迭代无改善则提前终止
        no_improvement = 0
        early_stop_threshold = 10
        improvement_threshold = 1e-3

        # ACO 主循环
        for it in range(self.iterations):
            paths = []
            costs = []
            for ant in range(self.num_ants):
                current = start_idx
                path = [current]
                steps = 0
                reached = False
                while steps < self.max_path_steps:
                    i, j, k = current
                    # 如果到达目标，则结束
                    if current == goal_idx:
                        reached = True
                        break
                    # 收集所有可行的邻居
                    moves = []
                    probs = []
                    for n_idx, (dx, dy, dz) in enumerate(self.neighbor_shifts):
                        ni = i + dx
                        nj = j + dy
                        nk = k + dz
                        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                            continue
                        # 遇到障碍则不可走
                        if grid[ni, nj, nk] == 1:
                            continue
                        tau = pheromone[i, j, k, n_idx] ** self.alpha
                        eta = (1.0 / heuristic[ni, nj, nk]) ** self.beta
                        moves.append((ni, nj, nk))
                        probs.append(tau * eta)
                    if len(moves) == 0:
                        break  # 无可行走法
                    # 归一化概率
                    total = sum(probs)
                    probs = [p / total for p in probs]
                    # 随机选择下一个状态
                    r = random.random()
                    cumulative = 0.0
                    for move, p in zip(moves, probs):
                        cumulative += p
                        if r <= cumulative:
                            next_cell = move
                            break
                    path.append(next_cell)
                    current = next_cell
                    steps += 1
                # 如果蚂蚁到达目标，则记录路径
                if reached:
                    cost = 0.0
                    for idx in range(len(path)-1):
                        pt1 = self.index_to_world(path[idx])
                        pt2 = self.index_to_world(path[idx+1])
                        cost += np.linalg.norm(pt2 - pt1)
                    paths.append(path)
                    costs.append(cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path

            # 信息素蒸发
            pheromone *= (1 - self.evaporation_rate)
            # 信息素沉积：对每个成功蚂蚁路径，按照 Q/cost 进行沉积
            for path, cost in zip(paths, costs):
                deposit = self.Q / cost
                for idx in range(len(path)-1):
                    i, j, k = path[idx]
                    next_cell = path[idx+1]
                    dx = next_cell[0] - i
                    dy = next_cell[1] - j
                    dz = next_cell[2] - k
                    try:
                        n_idx = self.neighbor_shifts.index((dx, dy, dz))
                    except ValueError:
                        continue
                    pheromone[i, j, k, n_idx] += deposit

            # 早停判断：若本次迭代未有明显改善，则累计计数；连续若干次无改善则提前退出
            if best_path is not None and costs and (min(costs) + improvement_threshold) >= best_cost:
                no_improvement += 1
            else:
                no_improvement = 0
            if no_improvement >= early_stop_threshold:
                logging.info(f"Early stopping at iteration {it+1} due to no improvement.")
                break

        if best_path is None:
            logging.info("no path found by ACO。")
            return None

        # 将 best_path（由网格索引组成）转换为连续坐标
        continuous_path = []
        for idx in best_path:
            continuous_path.append(self.index_to_world(idx))
        return continuous_path

    def smooth_path(self, path, smoothing_factor=1.0, num_points=100):
        """
        对离散路径进行平滑处理，利用样条插值生成平滑曲线
        """
        if len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    def train(self, env, num_episodes=10):
        """
        使用 ACO 算法规划路径后，利用 LQR 控制器跟踪规划路径。
        过程：
          1. 重置环境，获取起点 (env.location) 和目标 (env.get_current_target())。
          2. 利用 ACO 算法规划路径（使用 env.obstacles）。
          3. 对规划路径进行样条平滑处理。
          4. 构造状态 x = [position, velocity] 与期望状态 x_des，
             其中期望位置由路径点给出，期望速度利用相邻路径点计算（设定目标速度）。
          5. 利用 LQR 控制器生成控制输入 u = -K (x - x_des)，并限制在最大加速度内（角加速度置0）。
          6. 统计指标，并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_ACO_3D_LQR_planning", name="ACO_3D_LQR_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "num_ants": self.num_ants,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "evaporation_rate": self.evaporation_rate,
            "Q": self.Q,
            "max_path_steps": self.max_path_steps,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max],
            }
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            episode_start_time = time.time()
            logging.info(f"ACO LQR Episode {episode+1} starting")
            env.reset()
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()    # 3D 起点
            target = env.get_current_target()
            goal_pos = np.array(target)          # 3D 目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            if path is None:
                logging.info("ACO found no path")
                episode += 1
                continue

            # 对规划路径进行平滑处理
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

            # 绘制规划路径（在环境中展示）
            for i in range(len(path)-1):
                env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[30,50,0], thickness=5, lifetime=0)

            # 跟踪控制
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

            while step_count < max_steps_episode:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                if path_idx >= len(path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[path_idx]
                    if path_idx < len(path)-1:
                        desired_speed = 3.0  # 单位 m/s
                        direction = path[path_idx+1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-6:
                            direction = direction / norm_dir
                        else:
                            direction = np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < 2:
                        path_idx += 1
                        continue

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
                energy += np.linalg.norm(u)**2
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
            if np.linalg.norm(current_pos - goal_pos) < 2:
                self.ave_path_length += total_path_length
                self.ave_excu_time += episode_running_duration
                self.ave_plan_time += episode_planning_duration
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

        logging.info("ACO + LQR Planning finished training.")
        return