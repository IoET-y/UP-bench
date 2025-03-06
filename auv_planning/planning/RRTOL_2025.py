#auv_planning/planning/RRTOL_2025.py

import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # 用于路径平滑

from .base import BasePlanner


class OnlineRRTStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.5, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 step_size=2.0, neighbor_radius=5.0, init_iter=200, iter_per_cycle=50,
                 local_radius=15.0, valid_thresh=1.0):
        """
        参数说明：
          - grid_resolution: 用于障碍检测时的离散化分辨率（与环境单位一致）
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - ticks_per_sec: 模拟时间步频率
          - step_size: RRT 扩展时每次前进的步长（单位：米）
          - neighbor_radius: 重连（rewire）时搜索邻域的半径
          - init_iter: 每个 episode 初始规划时的迭代次数
          - iter_per_cycle: 每个控制周期额外扩展的迭代次数
          - local_radius: 局部采样区域半径（以当前 AUV 为中心采样）
          - valid_thresh: 路径有效性检测阈值（例如1m以内视为通过）

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

        # 用于障碍检测时将连续点转换为离散网格
        self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
        self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
        self.nz = int((self.z_max - self.z_min) / self.grid_resolution)

        # RRT* 参数
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.init_iter = init_iter
        self.iter_per_cycle = iter_per_cycle
        self.local_radius = local_radius  # 局部采样区域
        self.valid_thresh = valid_thresh  # 路径有效性检测距离阈值

        # 初始化树（每个节点：position, parent, cost）
        self.tree = []

        # 障碍地图：字典，key 为网格索引，value 为占据状态（1:障碍，0:自由）
        self.obstacle_map = {}

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

        super().__init__()

    # --------------- RRT* 核心函数 ---------------
    class Node:
        def __init__(self, pos, parent=None, cost=0.0):
            self.pos = pos  # 位置，np.array([x,y,z])
            self.parent = parent  # 父节点（或 None）
            self.cost = cost  # 从起点到该节点累计代价

    def sample_local(self, center):
        """在以 center 为中心、local_radius 范围内采样一个点（保证落在全局区域内）"""
        sample = center + np.random.uniform(-self.local_radius, self.local_radius, size=3)
        sample[0] = np.clip(sample[0], self.x_min, self.x_max)
        sample[1] = np.clip(sample[1], self.y_min, self.y_max)
        sample[2] = np.clip(sample[2], self.z_min, self.z_max)
        return sample

    def nearest_node(self, q_rand):
        """在树中找到与 q_rand 距离最近的节点"""
        min_dist = float('inf')
        nearest = None
        for node in self.tree:
            d = np.linalg.norm(node.pos - q_rand)
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest

    def steer(self, q_near, q_rand):
        """从 q_near 朝 q_rand 方向移动，步长不超过 step_size"""
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_rand
        else:
            return q_near + (direction / dist) * self.step_size

    def near_nodes(self, q_new):
        """返回树中与 q_new 距离小于 neighbor_radius 的节点"""
        neighbors = []
        for node in self.tree:
            if np.linalg.norm(node.pos - q_new) <= self.neighbor_radius:
                neighbors.append(node)
        return neighbors

    def collision_free(self, p1, p2):
        """
        检查从 p1 到 p2 的直线路径是否无碰撞，
        采用分步采样，将连续点转换为离散网格进行检测
        """
        dist = np.linalg.norm(p2 - p1)
        steps = int(dist / (self.grid_resolution / 2)) + 1
        for i in range(steps + 1):
            t = i / steps
            p = p1 * (1 - t) + p2 * t
            cell = self.world_to_index(p)
            if self.obstacle_map.get(cell, 0) == 1:
                return False
        return True

    def world_to_index(self, pos):
        """将连续位置转换为离散网格索引，用于障碍检测"""
        ix = int((pos[0] - self.x_min) / self.grid_resolution)
        iy = int((pos[1] - self.y_min) / self.grid_resolution)
        iz = int((pos[2] - self.z_min) / self.grid_resolution)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        iz = min(max(iz, 0), self.nz - 1)
        return (ix, iy, iz)

    def grow_tree(self, iterations, sample_center):
        """
        在当前树上扩展指定次数的 RRT* 迭代，
        采样只在以 sample_center 为中心、local_radius 范围内进行
        """
        for _ in range(iterations):
            q_rand = self.sample_local(sample_center)
            q_near = self.nearest_node(q_rand)
            q_new_pos = self.steer(q_near.pos, q_rand)
            if not self.collision_free(q_near.pos, q_new_pos):
                continue
            new_cost = q_near.cost + np.linalg.norm(q_new_pos - q_near.pos)
            q_new = self.Node(q_new_pos, parent=q_near, cost=new_cost)
            neighbors = self.near_nodes(q_new_pos)
            for neighbor in neighbors:
                if self.collision_free(neighbor.pos, q_new_pos):
                    temp_cost = neighbor.cost + np.linalg.norm(q_new_pos - neighbor.pos)
                    if temp_cost < q_new.cost:
                        q_new.parent = neighbor
                        q_new.cost = temp_cost
            self.tree.append(q_new)
            for neighbor in neighbors:
                if neighbor == q_new:
                    continue
                if self.collision_free(q_new.pos, neighbor.pos):
                    temp_cost = q_new.cost + np.linalg.norm(neighbor.pos - q_new.pos)
                    if temp_cost < neighbor.cost:
                        neighbor.parent = q_new
                        neighbor.cost = temp_cost

    def prune_tree(self, current_pos):
        """
        剪枝：移除树中距离 current_pos 超出 local_radius*2 的节点，
        使树主要集中在局部区域
        """
        self.tree = [node for node in self.tree if np.linalg.norm(node.pos - current_pos) < self.local_radius * 2]

    def get_best_path(self, start, goal):
        """
        从树中寻找离 goal 最近的节点，并构造从 start 到该节点的路径；
        如果该节点与 goal 之间 collision free，则返回完整路径
        """
        best_node = None
        best_dist = float('inf')
        for node in self.tree:
            d = np.linalg.norm(node.pos - goal)
            if d < best_dist:
                best_dist = d
                best_node = node
        if best_node is None:
            return None
        if self.collision_free(best_node.pos, goal):
            goal_node = self.Node(goal, parent=best_node, cost=best_node.cost + np.linalg.norm(goal - best_node.pos))
            best_node = goal_node
        path = []
        curr = best_node
        while curr is not None:
            path.append(curr.pos)
            curr = curr.parent
        path.reverse()
        return path

    # --------------- 障碍地图与传感器更新 ---------------
    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
        根据传感器读数更新障碍地图：
          - sensor_readings: 长度为14的数组；若读数 < 10，则视为障碍
          - 利用预设传感器方向（假设 agent 姿态为零），将测量转换为世界坐标，再转换为网格索引进行标记；
            同时沿射线采样，将视野内其他单元标记为空闲（0）
        """
        directions = []
        # 8个水平激光：角度0,45,...,315度
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

    # --------------- 全局路径有效性检查与局部重规划 ---------------
    def is_path_valid(self, path, current_pos):
        """
        检查路径从当前状态开始是否有效：即检查从 current_pos 到路径中第一个点是否 collision free，
        以及路径各段是否依然无障碍
        """
        if path is None or len(path) == 0:
            return False
        # 若 current_pos 与路径首个点距离较近，则认为已通过
        if np.linalg.norm(current_pos - path[0]) > self.valid_thresh:
            if not self.collision_free(current_pos, path[0]):
                return False
        for i in range(len(path) - 1):
            if not self.collision_free(path[i], path[i + 1]):
                return False
        return True


    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
        对离散路径使用样条插值进行平滑处理，并对结果进行边界约束，防止超出预定区域。
        """
        if path is None or len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points)

        # 对每个维度进行约束，防止超出预定区域
        smooth_path[0] = np.clip(smooth_path[0], self.x_min, self.x_max)
        smooth_path[1] = np.clip(smooth_path[1], self.y_min, self.y_max)
        smooth_path[2] = np.clip(smooth_path[2], self.z_min, self.z_max)

        return [pt for pt in smooth_path.T]
    # --------------- 主训练与规划循环 ---------------
    def train(self, env, num_episodes=10):
        wandb.init(project="auv_RRTStarOnline_planning", name="RRTStarOnline_run")
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
            "step_size": self.step_size,
            "neighbor_radius": self.neighbor_radius,
            "init_iter": self.init_iter,
            "iter_per_cycle": self.iter_per_cycle,
            "local_radius": self.local_radius
        })

        episode = 0
        reach_target_count = 0
        global_path = None

        while reach_target_count < 10 and episode < num_episodes:
            print("Now episode:", episode + 1, "Reach target count:", reach_target_count)
            episode_start_time = time.time()
            logging.info(f"Online RRT* Episode {episode + 1} starting")
            env.reset()

            # 获取初始状态与目标
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  # 3D起点
            target = env.get_current_target()
            goal_pos = np.array(target)  # 3D目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 重置障碍地图与 RRT 树
            self.obstacle_map = {}
            self.tree = []
            # 将起点作为树根加入
            root = self.Node(start_pos, parent=None, cost=0.0)
            self.tree.append(root)

            # 初始全局规划：扩展较多次
            self.grow_tree(self.init_iter, start_pos)
            global_path = self.get_best_path(start_pos, goal_pos)
            if global_path is None:
                logging.info("RRT* did not find an initial path.")
                episode += 1
                continue

            # 对规划路径进行平滑处理以便于显示
            global_path = self.smooth_path(global_path, smoothing_factor=1.0, num_points=200)
            for i in range(len(global_path) - 1):
                env.env.draw_line(global_path[i].tolist(), global_path[i + 1].tolist(), color=[30, 50, 0], thickness=5,
                                  lifetime=0)

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
            # 跟踪控制循环
            while step_count < max_steps_episode:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # 更新障碍地图（利用 14 个 RangeFinderSensor）
                sensor_readings = env.lasers.copy()
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)
                self.prune_tree(current_pos)

                # 对已有 global_path 进行剪枝：去掉已通过的路点
                while global_path and np.linalg.norm(current_pos - global_path[0]) < self.valid_thresh:
                    global_path.pop(0)

                # 如果全局路径存在且有效，则直接使用，否则仅局部重规划
                if global_path is not None and self.is_path_valid(global_path, current_pos):
                    updated_path = global_path
                else:
                    # 局部重规划：目标选取全局路径首个点（若无，则取全局目标）
                    local_target = global_path[0] if global_path and len(global_path) > 0 else goal_pos
                    local_segment = self.get_best_path(current_pos, local_target)
                    if local_segment is not None:
                        local_segment = self.smooth_path(local_segment, smoothing_factor=1.0, num_points=200)
                        # 将局部段拼接到后续 global_path 上（若存在）
                        if global_path is not None:
                            updated_path = local_segment[:-1] + global_path
                        else:
                            updated_path = local_segment
                        global_path = updated_path
                    else:
                        logging.info("No valid local path found, stopping episode.")
                        break

                # 选择当前目标点：若 updated_path 仍存在，则按序跟踪
                if path_idx >= len(updated_path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = updated_path[path_idx]
                    if path_idx < len(updated_path) - 1:
                        desired_speed = 3  # [m/s]
                        direction = updated_path[path_idx + 1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < self.valid_thresh:
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
            if np.linalg.norm(current_pos - goal_pos) < 2:
                self.ave_path_length += total_path_length
                self.ave_excu_time += episode_running_duration
                self.ave_plan_time += episode_planning_duration
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

        logging.info("Online RRT* Planning finished training.")
        return
