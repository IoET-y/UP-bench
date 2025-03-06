#auv_planning/planning/ASTAR_2025.py

import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

# 假设 BasePlanner 在同目录下的 base 模块中定义
from .base import BasePlanner

class AStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0):
        """
        参数说明：
          - grid_resolution: 离散化网格的分辨率（单位与环境一致）
          - max_steps: 每个episode允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
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

        self.ticks_per_sec = 100
        self.ts = 1.0 / self.ticks_per_sec  # 离散时间步长
        self.current_time = 0.0

        # 规划区域设置：参考 custom_env 中 draw_box 参数
        # 这里假定区域：x:0~100, y:0~100, z:-100~0
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0

        # 障碍物安全半径，用于在网格上“膨胀”障碍区域
        self.obstacle_radius = 5

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

    class Node:
        def __init__(self, x, y, z, g=0, h=0, parent=None):
            self.x = x
            self.y = y
            self.z = z
            self.g = g      # 从起点到当前节点的代价
            self.h = h      # 启发式代价（欧氏距离）
            self.f = g + h  # 总代价
            self.parent = parent

        def __lt__(self, other):
            return self.f < other.f

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
            radius_in_cells = int(math.ceil(self.obstacle_radius / self.grid_resolution))
            for i in range(max(0, obs_idx[0] - radius_in_cells), min(nx, obs_idx[0] + radius_in_cells + 1)):
                for j in range(max(0, obs_idx[1] - radius_in_cells), min(ny, obs_idx[1] + radius_in_cells + 1)):
                    for k in range(max(0, obs_idx[2] - radius_in_cells), min(nz, obs_idx[2] + radius_in_cells + 1)):
                        cell_center = self.index_to_world((i, j, k))
                        if np.linalg.norm(cell_center - np.array(obs)) <= self.obstacle_radius:
                            grid[i, j, k] = 1
        return grid

    def plan_path(self, start, goal, obstacles):
        """
        使用3D A*算法规划从 start 到 goal 的路径
        参数：
          - start: 起点 [x, y, z]
          - goal: 目标点 [x, y, z]
          - obstacles: 障碍物列表（每个为 [x, y, z]）
        返回：
          - path: 由连续坐标点构成的路径列表（每个元素为 np.array([x,y,z])）；若规划失败返回 None
        """
        grid = self.create_obstacle_grid(obstacles)
        nx, ny, nz = grid.shape

        start_idx = self.world_to_index(start)
        goal_idx = self.world_to_index(goal)

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
            # 遍历26邻域（包含对角线方向）
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

        # 从目标节点反向回溯构造路径
        path_indices = []
        node = goal_node
        while node is not None:
            path_indices.append((node.x, node.y, node.z))
            node = node.parent
        path_indices.reverse()

        # 将网格索引转换为连续坐标（取单元格中心）
        path = []
        for idx in path_indices:
            pos = self.index_to_world(idx)
            path.append(pos)
        return path

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
        对 A* 得到的离散路径进行平滑处理，利用样条插值生成平滑曲线
        参数：
          - path: 原始路径列表（每个元素为 np.array([x, y, z])）
          - smoothing_factor: 样条平滑参数（s=0时为精确插值，增大则更平滑）
          - num_points: 平滑路径采样点数
        返回：
          - smoothed_path: 平滑后的路径列表，每个元素为 np.array([x, y, z])
        """
        if len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        # 生成样条参数
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    def train(self, env, num_episodes=10):
        """
        使用3D A*规划路径后，利用LQR控制器跟踪规划路径。
        过程：
          1. 重置环境，获取起点（env.location）和目标（env.get_current_target()）。
          2. 使用3D A*算法规划路径（利用 env.obstacles 作为障碍物信息）。
          3. 控制器利用LQR追踪：构造状态 x = [position, velocity] 和期望状态 x_des，
             其中期望状态的“位置”为当前路径点，
             “速度”通过设定一个合理的目标速度并利用路径方向获得。
          4. 生成动作 u = -K (x - x_des)，并限制在最大线性加速度内，同时角加速度置0。
          5. 统计evaluation metrics，并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_AStar_3D_LQR_planning", name="AStar_3D_LQR_run")
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
            episode_start_time = time.time()
            logging.info(f"3D LQR Episode {episode+1} starting")
            env.reset()

            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()    # 3D起点
            target = env.get_current_target()
            goal_pos = np.array(target)          # 3D目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 规划路径
            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            if path is None:
                logging.info("3D A* did not find a path.")
                episode += 1
                continue

            # 对规划路径进行平滑处理
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

            # 绘制规划路径
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

            # 跟踪控制循环
            while step_count < max_steps_episode:
                # 判断是否到达目标（以规划网格分辨率为阈值）
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # 选择当前路径点作为期望位置
                if path_idx >= len(path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[path_idx]
                    # 如果存在下一个点，计算期望速度
                    if path_idx < len(path) - 1:
                        desired_speed = 3  # 根据任务需求设定目标速度 [m/s]
                        direction = path[path_idx+1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-6:
                            direction = direction / norm_dir
                        else:
                            direction = np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    # 当agent足够接近当前路径点时，切换到下一个点
                    if np.linalg.norm(current_pos - waypoint) < 1:
                        path_idx += 1
                        continue

                # 构造当前状态和期望状态
                # 当前状态：位置和env中提供的速度（3维）
                x_current = np.hstack([current_pos, env.velocity.copy()])
                # 期望状态：当前位置为 waypoint，期望速度为 v_des
                x_des = np.hstack([waypoint, v_des])
                error_state = x_current - x_des

                # LQR控制： u = -K * (x - x_des)
                u = -self.K.dot(error_state)
                # 限制控制输入幅值
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                # 构造完整action：前三个为线性加速度，后三个角加速度置0
                action = np.concatenate([u, np.zeros(3)])
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()

                # 更新统计指标
                distance_moved = np.linalg.norm(new_pos - current_pos)
                total_path_length += distance_moved
                energy += np.linalg.norm(u) ** 2
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

                # 碰撞检测：若距离任一障碍物小于 collision_threshold，则计为碰撞
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


        logging.info("3D A* + LQR Planning finished training.")
        return
