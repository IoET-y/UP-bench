# rrt_planner_3d_lqr.py

import numpy as np
import math
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev
import random

# 假设 BasePlanner 在同目录下的 base 模块中定义
from .base import BasePlanner

class RRTPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0,
                 max_iterations=5000, step_size=2.0, goal_bias=0.05):
        """
        参数说明：
          - grid_resolution: 空间中用于碰撞检测的采样分辨率（单位与环境一致）
          - max_steps: 每个episode允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - max_iterations: RRT 最大采样次数
          - step_size: RRT 扩展时步长
          - goal_bias: 采样时直接选择目标点的概率
        """
        self.grid_resolution = grid_resolution
        self.max_steps = max_steps
        self.max_lin_accel = max_lin_accel
        self.collision_threshold = collision_threshold
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias

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

        # 障碍物安全半径（用于碰撞检测时“膨胀”障碍）
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
        # 调整代价矩阵：位置权重较高，适当提高速度误差权重，使控制更平滑
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1, 0.1])
        # 求解离散Riccati方程，计算 LQR 增益矩阵
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        # -------------------------------

        super().__init__()

    class Node:
        def __init__(self, pos, parent=None):
            """
            pos: 3D 坐标，np.array([x, y, z])
            parent: 父节点（用于回溯路径）
            """
            self.pos = pos
            self.parent = parent

    def get_random_point(self, goal):
        """
        根据目标偏向采样策略采样一个随机点
        """
        if random.random() < self.goal_bias:
            return goal.copy()
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        return np.array([x, y, z])

    def get_nearest_node(self, tree, point):
        """
        在 tree 中寻找距离 point 最近的节点
        """
        dists = [np.linalg.norm(node.pos - point) for node in tree]
        idx = np.argmin(dists)
        return tree[idx]

    def is_collision_free(self, p1, p2, obstacles):
        """
        检查从 p1 到 p2 的直线路径是否与任一障碍物碰撞
        采用线段采样法，每隔一定步长采样一个点
        """
        dist = np.linalg.norm(p2 - p1)
        num_samples = max(int(dist / (self.grid_resolution / 2)), 2)
        for i in range(num_samples):
            t = i / (num_samples - 1)
            pt = p1 + t * (p2 - p1)
            # 检查该采样点与所有障碍物的距离
            for obs in obstacles:
                if np.linalg.norm(pt - np.array(obs)) < self.obstacle_radius:
                    return False
        return True

    def plan_path(self, start, goal, obstacles):
        """
        使用 RRT 算法规划从 start 到 goal 的路径
        参数：
          - start: 起点 np.array([x, y, z])
          - goal: 目标 np.array([x, y, z])
          - obstacles: 障碍物列表（每个为 [x, y, z]）
        返回：
          - path: 由连续坐标点构成的路径列表（每个元素为 np.array([x,y,z])）；若规划失败返回 None
        """
        tree = []
        start_node = self.Node(np.array(start))
        tree.append(start_node)
        found = False
        goal_node = None

        for i in range(self.max_iterations):
            rnd_point = self.get_random_point(np.array(goal))
            nearest_node = self.get_nearest_node(tree, rnd_point)
            direction = rnd_point - nearest_node.pos
            if np.linalg.norm(direction) == 0:
                continue
            direction = direction / np.linalg.norm(direction)
            new_pos = nearest_node.pos + self.step_size * direction

            # 保证 new_pos 在规划区域内
            new_pos[0] = np.clip(new_pos[0], self.x_min, self.x_max)
            new_pos[1] = np.clip(new_pos[1], self.y_min, self.y_max)
            new_pos[2] = np.clip(new_pos[2], self.z_min, self.z_max)

            if not self.is_collision_free(nearest_node.pos, new_pos, obstacles):
                continue

            new_node = self.Node(new_pos, parent=nearest_node)
            tree.append(new_node)

            # 若新节点足够接近目标，则认为规划成功
            if np.linalg.norm(new_pos - goal) < self.step_size:
                # 在目标与新节点之间再检查一次碰撞
                if self.is_collision_free(new_pos, goal, obstacles):
                    goal_node = self.Node(np.array(goal), parent=new_node)
                    tree.append(goal_node)
                    found = True
                    break

        if not found:
            logging.info("RRT: 未能在最大迭代次数内找到路径。")
            return None

        # 从 goal_node 反向回溯得到路径
        path = []
        node = goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()
        return path

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
        对 RRT 得到的离散路径进行平滑处理，利用样条插值生成平滑曲线
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
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    def train(self, env, num_episodes=10):
        """
        使用 RRT 算法规划路径后，利用 LQR 控制器跟踪规划路径。
        过程：
          1. 重置环境，获取起点（env.location）和目标（env.get_current_target()）。
          2. 使用 RRT 算法规划路径（利用 env.obstacles 作为障碍物信息）。
          3. 对规划路径进行样条平滑处理。
          4. 构造状态 x = [position, velocity] 与期望状态 x_des（期望位置为路径点，期望速度通过路径方向设定）。
          5. 利用 LQR 控制器生成动作 u = -K (x - x_des)，并限制在最大线性加速度内（角加速度置0）。
          6. 统计指标并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_RRT_3D_LQR_planning", name="RRT_3D_LQR_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "max_iterations": self.max_iterations,
            "step_size": self.step_size,
            "goal_bias": self.goal_bias,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max],
            }
        })

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 :
            episode_start_time = time.time()
            logging.info(f"RRT LQR Episode {episode+1} starting")
            env.reset()
            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()    # 3D起点
            target = env.get_current_target()
            goal_pos = np.array(target)          # 3D目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 规划路径（RRT 算法）
            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            if path is None:
                logging.info("RRT 未能找到路径。")
                episode += 1
                continue

            # 对规划路径进行平滑处理
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

            # 绘制规划路径（环境中展示）
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
                # 判断是否到达目标（以规划区域尺度为阈值）
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # 选择当前路径点作为期望目标
                if path_idx >= len(path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[path_idx]
                    # 若存在下一个点，计算期望速度（设定目标速度并根据相邻路径点计算方向）
                    if path_idx < len(path) - 1:
                        desired_speed = 3.0  # [m/s] 目标速度
                        direction = path[path_idx+1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-6:
                            direction = direction / norm_dir
                        else:
                            direction = np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    # 当 agent 足够接近当前路径点时，切换到下一个点
                    if np.linalg.norm(current_pos - waypoint) < 1:
                        path_idx += 1
                        continue

                # 构造当前状态与期望状态
                # 当前状态：位置与 env 中提供的速度（3维）
                x_current = np.hstack([current_pos, env.velocity.copy()])
                # 期望状态：位置为 waypoint，速度为 v_des
                x_des = np.hstack([waypoint, v_des])
                error_state = x_current - x_des

                # LQR 控制律： u = -K (x - x_des)
                u = -self.K.dot(error_state)
                # 限制控制输入幅值
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                # 构造完整 action：前三个为线性加速度，后三个角加速度置0
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

            logging.info(f"Episode {episode+1} completed - Path Length: {total_path_length}, Steps: {step_count}, Collisions: {collisions}")

            if reach_target_count >= 10:
                return
            episode += 1
            env.set_current_target(env.choose_next_target())


        logging.info("RRT + LQR Planning finished training.")
        return