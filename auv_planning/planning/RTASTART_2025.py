#auv_planning/planning/RTASTART_2025.py

import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner

class RTAAStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.5, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 lookahead=50):
        """
        参数说明：
          - grid_resolution: 离散化网格的分辨率（单位与环境一致，建议适当调大以降低计算量）
          - max_steps: 每个episode允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率
          - lookahead: 每次RTAA*搜索的最大节点扩展数
        同时记录evaluation metrics：
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

        # 离线设计LQR控制器（与D* Lite版本一致）
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

        # RTAA*相关参数
        self.lookahead = lookahead
        self.H = {}  # 用于存储“学习”更新的启发值，key为网格索引

        # 障碍地图：字典，key为网格索引，value为占据状态（1:障碍，0:自由）
        self.obstacle_map = {}

        # 预先计算26邻域偏移量，避免每次三重循环
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
    def heuristic(self, a, b):
        return self.grid_resolution * math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def cost(self, a, b):
        if self.obstacle_map.get(a, 0) == 1 or self.obstacle_map.get(b, 0) == 1:
            return float('inf')
        return self.grid_resolution * math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    # --- 内部搜索节点 ---
    class Node:
        def __init__(self, idx, g, h, parent):
            self.idx = idx      # 网格索引
            self.g = g          # 从起点累计代价
            self.h = h          # 启发值（若存在更新则使用更新值，否则采用标准启发）
            self.f = g + h      # f = g + h
            self.parent = parent
        def __lt__(self, other):
            return self.f < other.f

    # --- RTAA*有限扩展A*搜索 ---
    def rtaa_search(self, start, goal, lookahead):
        open_list = []
        closed = {}
        start_h = self.H.get(start, self.heuristic(start, goal))
        start_node = self.Node(start, 0, start_h, None)
        heapq.heappush(open_list, start_node)
        closed[start] = start_node
        expansions = 0
        goal_reached = False
        goal_node = None

        while open_list and expansions < lookahead:
            current = heapq.heappop(open_list)
            expansions += 1
            if current.idx == goal:
                goal_reached = True
                goal_node = current
                break
            for neighbor in self.get_neighbors(current.idx):
                # 跳过障碍单元
                if self.obstacle_map.get(neighbor, 0) == 1:
                    continue
                tentative_g = current.g + self.cost(current.idx, neighbor)
                if neighbor in closed:
                    if tentative_g < closed[neighbor].g:
                        closed[neighbor].g = tentative_g
                        closed[neighbor].parent = current
                        closed[neighbor].f = tentative_g + self.H.get(neighbor, self.heuristic(neighbor, goal))
                        heapq.heappush(open_list, closed[neighbor])
                else:
                    h_val = self.H.get(neighbor, self.heuristic(neighbor, goal))
                    neighbor_node = self.Node(neighbor, tentative_g, h_val, current)
                    closed[neighbor] = neighbor_node
                    heapq.heappush(open_list, neighbor_node)
        if goal_reached:
            # 重构从start到goal的完整路径
            path = []
            node = goal_node
            while node is not None:
                path.append(node.idx)
                node = node.parent
            path.reverse()
            return path
        if open_list:
            best_node = min(open_list, key=lambda n: n.f)
            # 学习更新：更新起点启发值
            self.H[start] = best_node.f
            # 从起点选择一个后继动作：在所有邻居中选择 (cost + H) 最小者
            best_successor = None
            best_val = float('inf')
            for neighbor in self.get_neighbors(start):
                if self.obstacle_map.get(neighbor, 0) == 1:
                    continue
                val = self.cost(start, neighbor) + self.H.get(neighbor, self.heuristic(neighbor, goal))
                if val < best_val:
                    best_val = val
                    best_successor = neighbor
            if best_successor is None:
                return None
            return [start, best_successor]
        return None

    # --- 初始全局规划 ---
    def plan_path(self, start, goal):
        self.start = self.world_to_index(start)
        self.goal = self.world_to_index(goal)
        self.H = {}  # 重置已学习的启发值
        # 初始规划时允许更大扩展数
        path_indices = self.rtaa_search(self.start, self.goal, self.lookahead * 10)
        if path_indices is None:
            return None
        world_path = [self.index_to_world(idx) for idx in path_indices]
        return world_path

    # --- 路径平滑 ---
    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        if path is None or len(path) < 4:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    # --- 利用传感器更新障碍地图 ---
    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        sensor_readings: 长度为14的列表或数组；若读数 < 10，则视为障碍
        根据预设的传感器相对方向（假设agent姿态为零），将测量转换为世界坐标，
        再转换为网格索引进行标记；同时沿射线采样，将视野内其他单元标记为自由（0）
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
        # 可选：对于更新过的单元，其邻域的启发值可能需要更新（本实现中在下一次搜索时重新计算）
        return updated_cells

    # --- 主训练及规划循环 ---
    def train(self, env, num_episodes=10):
        wandb.init(project="auv_RTAAStar_planning", name="RTAAStar_run")
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
            "lookahead": self.lookahead
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            print("Now episode:", episode+1, "Reach target count:", reach_target_count)
            episode_start_time = time.time()
            logging.info(f"RTAA* Episode {episode+1} starting")
            env.reset()

            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  # 3D起点
            target = env.get_current_target()
            goal_pos = np.array(target)        # 3D目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 每个episode重置障碍地图及启发值表
            self.obstacle_map = {}
            self.H = {}
            # 重新计算网格尺寸参数（基于规划区域设置）
            self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
            self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
            self.nz = int((self.z_max - self.z_min) / self.grid_resolution)
            # 初始规划：由于障碍地图为空
            path = self.plan_path(start_pos, goal_pos)
            print("Got an initial path.")
            if path is None:
                logging.info("RTAA* did not find an initial path.")
                episode += 1
                continue

            # 对规划路径进行平滑处理便于显示
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
            print("Following the path with LQR control...")
            # 跟踪控制循环：每个控制周期内更新障碍地图、重新规划局部路径，然后利用LQR跟踪
            while step_count < max_steps_episode:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                sensor_readings = env.lasers.copy()  # 14个传感器读数
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # 利用RTAA*重新规划：从当前状态开始，采用有限lookahead搜索
                current_idx = self.world_to_index(current_pos)
                goal_idx = self.world_to_index(goal_pos)
                path_indices = self.rtaa_search(current_idx, goal_idx, self.lookahead)
                if path_indices is not None:
                    new_path = [self.index_to_world(idx) for idx in path_indices]
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
                        desired_speed = 3  # [m/s]
                        direction = path[path_idx+1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < 1:
                        path_idx += 1
                        continue

                # 利用LQR控制器生成控制输入
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
                energy += np.linalg.norm(u)**2 / 100
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

        logging.info("RTAA* Planning finished training.")
        return
