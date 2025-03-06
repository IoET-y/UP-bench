# auv_planning/planning/GA_2025.py

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


class GeneticPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0,
                 population_size=50, generations=100, num_intermediate=10,
                 mutation_rate=0.2, crossover_rate=0.7):
        """
        参数说明：
          - grid_resolution: 用于碰撞检测的分辨率
          - max_steps: 每个 episode 允许的最大步数
          - max_lin_accel: 最大线性加速度（控制指令上限）
          - collision_threshold: 碰撞检测阈值
          - population_size: 遗传算法种群数量
          - generations: 遗传算法迭代代数
          - num_intermediate: 候选路径中除起点和目标外的中间点数量
          - mutation_rate: 变异概率
          - crossover_rate: 交叉概率
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

        # 遗传算法参数
        self.population_size = population_size
        self.generations = generations
        self.num_intermediate = num_intermediate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

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
        #   p_{k+1} = p_k + dt * v_k + 0.5 * dt^2 * u_k
        #   v_{k+1} = v_k + dt * u_k
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
    # 遗传算法相关方法
    # -------------------------

    def generate_candidate(self, start, goal):
        """
        生成一个候选路径，包含 start、goal 和 num_intermediate 个随机中间点
        """
        # 起点和目标固定，中间点随机生成于规划区域内
        candidate = [np.array(start)]
        for _ in range(self.num_intermediate):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            z = random.uniform(self.z_min, self.z_max)
            candidate.append(np.array([x, y, z]))
        candidate.append(np.array(goal))
        return candidate

    def initialize_population(self, start, goal):
        """
        初始化种群，每个个体为一条候选路径
        """
        return [self.generate_candidate(start, goal) for _ in range(self.population_size)]

    def is_collision_free(self, p1, p2, obstacles):
        """
        检查从 p1 到 p2 的直线路径是否与任一障碍物碰撞
        采用线段采样，每隔一定距离采样一个点进行检测
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
        计算候选路径中所有相邻点之间是否有碰撞，
        若有碰撞则返回惩罚值，否则返回 0
        """
        penalty = 0.0
        for i in range(len(candidate) - 1):
            if not self.is_collision_free(candidate[i], candidate[i + 1], obstacles):
                penalty += 1000  # 碰撞惩罚，可根据需要调整
        return penalty

    def path_length(self, candidate):
        """
        计算候选路径总长度（欧氏距离累计）
        """
        length = 0.0
        for i in range(len(candidate) - 1):
            length += np.linalg.norm(candidate[i + 1] - candidate[i])
        return length

    def fitness(self, candidate, obstacles):
        """
        候选路径的适应度：路径长度加上碰撞惩罚
        注意：这里越小越好
        """
        return self.path_length(candidate) + self.path_collision_penalty(candidate, obstacles)

    def tournament_selection(self, population, fitnesses, k=3):
        """
        锦标赛选择，从种群中随机选 k 个候选，返回适应度最好的那一个
        """
        selected = random.sample(list(zip(population, fitnesses)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def crossover(self, parent1, parent2):
        """
        对两个父代个体进行交叉。由于起点和目标固定，
        仅对中间点进行单点交叉
        """
        child1 = [parent1[0]]  # 起点
        child2 = [parent2[0]]
        # 仅对中间点进行交叉
        if self.num_intermediate > 0:
            crossover_point = random.randint(1, self.num_intermediate)
            # 子代1：父1前 crossover_point 个中间点 + 父2剩余部分
            child1 += parent1[1:crossover_point + 1] + parent2[crossover_point + 1:-1]
            # 子代2：父2前 crossover_point 个中间点 + 父1剩余部分
            child2 += parent2[1:crossover_point + 1] + parent1[crossover_point + 1:-1]
        child1.append(parent1[-1])  # 目标
        child2.append(parent1[-1])
        return child1, child2

    def mutate(self, candidate):
        """
        对候选路径进行变异，随机扰动中间点的位置
        """
        new_candidate = [candidate[0]]  # 起点不变
        for pt in candidate[1:-1]:
            if random.random() < self.mutation_rate:
                # 对当前点加上随机扰动，扰动幅度可设为区域尺寸的 5%
                dx = random.uniform(-0.05 * (self.x_max - self.x_min), 0.05 * (self.x_max - self.x_min))
                dy = random.uniform(-0.05 * (self.y_max - self.y_min), 0.05 * (self.y_max - self.y_min))
                dz = random.uniform(-0.05 * (abs(self.z_max - self.z_min)), 0.05 * (abs(self.z_max - self.z_min)))
                new_pt = pt + np.array([dx, dy, dz])
                # 限制在规划区域内
                new_pt[0] = np.clip(new_pt[0], self.x_min, self.x_max)
                new_pt[1] = np.clip(new_pt[1], self.y_min, self.y_max)
                new_pt[2] = np.clip(new_pt[2], self.z_min, self.z_max)
                new_candidate.append(new_pt)
            else:
                new_candidate.append(pt)
        new_candidate.append(candidate[-1])
        return new_candidate

    def run_genetic_algorithm(self, start, goal, obstacles):
        """
        执行遗传算法规划路径，返回最优候选路径（列表形式）
        """
        population = self.initialize_population(start, goal)
        best_candidate = None
        best_fitness = float('inf')

        for gen in range(self.generations):
            fitnesses = [self.fitness(candidate, obstacles) for candidate in population]
            # 更新最优解
            for candidate, fit in zip(population, fitnesses):
                if fit < best_fitness:
                    best_fitness = fit
                    best_candidate = candidate
            new_population = []
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self.tournament_selection(population, fitnesses, k=3)
                parent2 = self.tournament_selection(population, fitnesses, k=3)
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:self.population_size]
            # 可选：输出当前代数及最优适应度
            # logging.info(f"GA Generation {gen+1}, best fitness: {best_fitness}")
        return best_candidate

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
        对遗传算法得到的离散路径进行平滑处理，利用样条插值生成平滑曲线
        """
        if len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
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
        使用遗传算法规划路径后，利用 LQR 控制器跟踪规划路径。
        过程：
          1. 重置环境，获取起点（env.location）和目标（env.get_current_target()）。
          2. 利用遗传算法规划路径（采用环境中的障碍物信息）。
          3. 对规划路径进行样条平滑处理。
          4. 构造状态 x = [position, velocity] 与期望状态 x_des，
             其中期望位置来自路径点，期望速度依据相邻路径点计算（设定目标速度）。
          5. 利用 LQR 控制器生成动作 u = -K (x - x_des)，限制在最大加速度内，角加速度置 0。
          6. 统计指标，并通过 wandb.log 记录日志。
        """
        wandb.init(project="auv_Genetic_3D_LQR_planning", name="Genetic_3D_LQR_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "population_size": self.population_size,
            "generations": self.generations,
            "num_intermediate": self.num_intermediate,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
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
            logging.info(f"Genetic GA LQR Episode {episode + 1} starting")
            env.reset()
            # 用零动作获取初始状态
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  # 3D 起点
            target = env.get_current_target()
            goal_pos = np.array(target)  # 3D 目标
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # 利用遗传算法规划路径
            candidate_path = self.run_genetic_algorithm(start_pos, goal_pos, env.obstacles)
            if candidate_path is None:
                logging.info("遗传算法未能找到路径。")
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
                    # 若存在下一个点，则计算期望速度（设定目标速度，并根据相邻路径点确定方向）
                    if path_idx < len(path) - 1:
                        desired_speed = 1.0  # 单位 [m/s]
                        direction = path[path_idx + 1] - waypoint
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

                # 构造当前状态与期望状态（状态由位置与 env.velocity 提供）
                x_current = np.hstack([current_pos, env.velocity.copy()])
                x_des = np.hstack([waypoint, v_des])
                error_state = x_current - x_des

                # LQR 控制律： u = -K (x - x_des)
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                action = np.concatenate([u, np.zeros(3)])  # 角加速度置 0
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()

                distance_moved = np.linalg.norm(new_pos - current_pos)
                total_path_length += distance_moved
                energy += np.linalg.norm(u) ** 2
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

                # 碰撞检测：若与任一障碍物距离小于 collision_threshold，则计为碰撞
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

        logging.info("Genetic GA + LQR Planning finished training.")
        return
