import numpy as np
import math
import random
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

# 假设 BasePlanner 在同目录下的 base 模块中定义
from .base import BasePlanner


class FireflyPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=10000,
                 max_lin_accel=10, collision_threshold=5.0,
                 population_size=40, iterations=100, num_intermediate=10,
                 alpha_firefly=0.2, beta0=1.0, gamma=1.0):
        """
        参数说明：
          - grid_resolution: 用于碰撞检测的分辨率
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - population_size: 萤火虫（候选路径）数量
          - iterations: 萤火虫算法迭代次数
          - num_intermediate: 候选路径中除起点和目标外的中间点数量
          - alpha_firefly: 随机扰动因子
          - beta0: 初始吸引力系数
          - gamma: 吸收系数（控制距离衰减）
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

        # Firefly 算法参数
        self.population_size = population_size
        self.iterations = iterations
        self.num_intermediate = num_intermediate
        self.alpha_firefly = alpha_firefly
        self.beta0 = beta0
        self.gamma = gamma

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

        # 障碍物安全半径（用于碰撞检测时“膨胀”障碍）
        self.obstacle_radius = 5

        # -------------------------------
        # 离线设计 LQR 控制器
        # 假设 agent 的动力学为双积分模型：
        #   p_{k+1} = p_k + dt*v_k + 0.5*dt^2*u_k
        #   v_{k+1} = v_k + dt*u_k
        # 状态 x = [p; v] ∈ R^6, 控制 u ∈ R^3.
        I3 = np.eye(3)
        A = np.block([
            [np.eye(3), self.ts * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        B = np.block([
            [0.5 * (self.ts ** 2) * np.eye(3)],
            [self.ts * np.eye(3)]
        ])
        # 设置较高的位置权重，同时适当增加速度权重，使得控制更平滑
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1, 0.1])
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        # -------------------------------

        super().__init__()

    # -------------------------
    # Firefly 算法相关方法
    # -------------------------
    def generate_candidate(self, start, goal):
        """
        生成一个候选路径，返回形状为 (num_intermediate+2, 3) 的 NumPy 数组，
        其中第一行为起点，最后一行为目标，中间行为随机中间点。
        """
        num_points = self.num_intermediate + 2
        candidate = np.empty((num_points, 3), dtype=np.float64)
        candidate[0] = np.array(start)
        candidate[-1] = np.array(goal)
        for i in range(1, num_points - 1):
            candidate[i, 0] = random.uniform(self.x_min, self.x_max)
            candidate[i, 1] = random.uniform(self.y_min, self.y_max)
            candidate[i, 2] = random.uniform(self.z_min, self.z_max)
        return candidate

    def initialize_population(self, start, goal):
        """
        初始化种群，每个个体为一条候选路径（NumPy 数组）
        """
        return [self.generate_candidate(start, goal) for _ in range(self.population_size)]

    def is_collision_free(self, p1, p2, obstacles):
        """
        检查从 p1 到 p2 的直线路径是否与任一障碍物碰撞，
        采用线段采样检测，若任一点距离障碍物小于 obstacle_radius 则认为碰撞。
        """
        dist = np.linalg.norm(p2 - p1)
        num_samples = max(int(dist / (self.grid_resolution / 2)), 2)
        for i in range(num_samples):
            t = i / (num_samples - 1)
            pt = p1 + t * (p2 - p1)
            for obs in obstacles:
                if np.linalg.norm(pt - np.array(obs)) < self.obstacle_radius:
                    return False
        return True

    def path_collision_penalty(self, candidate, obstacles):
        """
        计算候选路径中所有相邻点之间的碰撞惩罚，若发生碰撞则累加惩罚值。
        """
        penalty = 0.0
        for i in range(candidate.shape[0] - 1):
            if not self.is_collision_free(candidate[i], candidate[i + 1], obstacles):
                penalty += 1000  # 惩罚值，可根据需要调整
        return penalty

    def path_length(self, candidate):
        """
        利用向量化计算候选路径总长度（欧氏距离累计）
        """
        diffs = np.diff(candidate, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def fitness(self, candidate, obstacles):
        """
        候选路径的适应度：路径长度加上碰撞惩罚（越小越好）
        """
        return self.path_length(candidate) + self.path_collision_penalty(candidate, obstacles)

    def run_firefly_algorithm(self, start, goal, obstacles):
        """
        执行萤火虫算法规划路径，返回适应度最优的候选路径（NumPy 数组形式）
        """
        population = self.initialize_population(start, goal)
        best_candidate = None
        best_fitness = np.inf

        # 早停参数：若连续 early_stop_threshold 次迭代无改进，则提前终止
        no_improvement = 0
        early_stop_threshold = 10
        improvement_threshold = 1e-3

        for it in range(self.iterations):
            fitnesses = [self.fitness(candidate, obstacles) for candidate in population]
            # 更新全局最优解
            current_best = min(fitnesses)
            current_best_idx = fitnesses.index(current_best)
            if current_best < best_fitness - improvement_threshold:
                best_fitness = current_best
                best_candidate = population[current_best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            # 提前退出判断
            if no_improvement >= early_stop_threshold:
                logging.info(f"Early stopping at iteration {it+1} due to no improvement.")
                break
            # 对每对火萤进行比较与吸引
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitnesses[j] < fitnesses[i]:
                        # 仅更新中间点（不改变起点与目标）
                        diff = (population[i][1:-1] - population[j][1:-1])
                        r = np.linalg.norm(diff)
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        random_component = self.alpha_firefly * (np.random.rand(*population[i][1:-1].shape) - 0.5)
                        population[i][1:-1] = population[i][1:-1] + beta * (population[j][1:-1] - population[i][1:-1]) + random_component
                        # 限制更新后中间点在规划区域内
                        population[i][1:-1, 0] = np.clip(population[i][1:-1, 0], self.x_min, self.x_max)
                        population[i][1:-1, 1] = np.clip(population[i][1:-1, 1], self.y_min, self.y_max)
                        population[i][1:-1, 2] = np.clip(population[i][1:-1, 2], self.z_min, self.z_max)
        return best_candidate

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
        对候选路径进行样条插值平滑处理，返回平滑后的路径列表（每个元素为 np.array([x,y,z])）
        """
        if path.shape[0] < 3:
            return path
        path_array = path.T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        return [pt for pt in smooth_path]

    # -------------------------
    # 训练（规划与跟踪）过程
    # -------------------------
    def train(self, env, num_episodes=10):
        """
        使用萤火虫算法规划路径后，利用 LQR 控制器跟踪规划路径。
        过程：
          1. 重置环境，获取起点 (env.location) 和目标 (env.get_current_target())。
          2. 利用萤火虫算法规划路径（采用环境中的障碍物信息）。
          3. 对规划路径进行样条平滑处理。
          4. 构造状态 x = [position, velocity] 与期望状态 x_des，
             其中期望位置由路径点给出，期望速度依据相邻路径点计算（设定目标速度）。
          5. 利用 LQR 控制器生成控制输入 u = -K (x - x_des)，限制在最大加速度内，角加速度置 0。
          6. 统计指标，并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_Firefly_3D_LQR_planning", name="Firefly_3D_LQR_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "population_size": self.population_size,
            "iterations": self.iterations,
            "num_intermediate": self.num_intermediate,
            "alpha_firefly": self.alpha_firefly,
            "beta0": self.beta0,
            "gamma": self.gamma,
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
            logging.info(f"Firefly LQR Episode {episode + 1} starting")
            env.reset()
            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  # 3D 起点
            target = env.get_current_target()
            goal_pos = np.array(target)  # 3D 目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 利用萤火虫算法规划路径
            candidate_path = self.run_firefly_algorithm(start_pos, goal_pos, env.obstacles)
            if candidate_path is None:
                logging.info("萤火虫算法未能找到路径。")
                episode += 1
                continue

            # 对规划路径进行平滑处理
            path = self.smooth_path(candidate_path, smoothing_factor=1.0, num_points=200)

            # 绘制规划路径（在环境中展示）
            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            # 跟踪控制参数
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
                    if path_idx < len(path) - 1:
                        desired_speed = 3.0  # 单位 [m/s]
                        direction = path[path_idx + 1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-6:
                            direction = direction / norm_dir
                        else:
                            direction = np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < 1:
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
                energy += np.linalg.norm(u) ** 2
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

        logging.info("Firefly + LQR Planning finished training.")
        return