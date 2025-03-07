import numpy as np
import math
import random
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

from .base import BasePlanner


class FireflyPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=10000,
                 max_lin_accel=10, collision_threshold=5.0,
                 population_size=40, iterations=100, num_intermediate=10,
                 alpha_firefly=0.2, beta0=1.0, gamma=1.0):

        # Firefly parameter
        self.population_size = population_size
        self.iterations = iterations
        self.num_intermediate = num_intermediate
        self.alpha_firefly = alpha_firefly
        self.beta0 = beta0
        self.gamma = gamma

        self.obstacle_radius = 5

        super().__init__()
    
    # -------------------------
    # Firefly algorithm  methods
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
        Initialize the population, each individual is a candidate path (NumPy array)
        """
        return [self.generate_candidate(start, goal) for _ in range(self.population_size)]

    def is_collision_free(self, p1, p2, obstacles):
        """
        Check whether the straight line path from p1 to p2 collides with any obstacle.
        Use line segment sampling detection. If the distance from any point to the obstacle is less than obstacle_radius, it is considered a collision.
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
        Calculate the collision penalty between all adjacent points in the candidate path, and accumulate the penalty value if a collision occurs.
        """
        penalty = 0.0
        for i in range(candidate.shape[0] - 1):
            if not self.is_collision_free(candidate[i], candidate[i + 1], obstacles):
                penalty += 1000  # 惩罚值，可根据需要调整
        return penalty

    def path_length(self, candidate):
        """
        Use vectorization to calculate the total length of candidate paths (Euclidean distance accumulation)
        """
        diffs = np.diff(candidate, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def fitness(self, candidate, obstacles):
        """
        Fitness of candidate path: path length plus collision penalty (the smaller the better)
        """
        return self.path_length(candidate) + self.path_collision_penalty(candidate, obstacles)

    def run_firefly_algorithm(self, start, goal, obstacles):
        """
        Execute the firefly algorithm to plan the path and return the candidate path with the best fitness (in NumPy array format)
        """
        population = self.initialize_population(start, goal)
        best_candidate = None
        best_fitness = np.inf

        no_improvement = 0
        early_stop_threshold = 10
        improvement_threshold = 1e-3

        for it in range(self.iterations):
            fitnesses = [self.fitness(candidate, obstacles) for candidate in population]
            current_best = min(fitnesses)
            current_best_idx = fitnesses.index(current_best)
            if current_best < best_fitness - improvement_threshold:
                best_fitness = current_best
                best_candidate = population[current_best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= early_stop_threshold:
                logging.info(f"Early stopping at iteration {it+1} due to no improvement.")
                break

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
    # Planning and Tracking Process
    # -------------------------
    def train(self, env, num_episodes=10):
        """
        After planning the path using the firefly algorithm, use the LQR controller to track the planned path.
        Process:
        1. Reset the environment and get the starting point (env.location) and target (env.get_current_target()).
        2. Use the firefly algorithm to plan the path (using the obstacle information in the environment).
        3. Perform spline smoothing on the planned path.
        4. Construct the state x = [position, velocity] and the expected state x_des,
        where the expected position is given by the path point, and the expected velocity is calculated based on the adjacent path points (set the target velocity).
        5. Use the LQR controller to generate the control input u = -K (x - x_des), limit it to the maximum acceleration, and set the angular acceleration to 0.
        6. Statistical indicators and log them through wandb.log.
        """
        wandb.init(project="auv_Firefly_3D_LQR_planning", name="Firefly_3D_LQR_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "population_size": self.population_size,
            "iterations": self.iterations,
            "num_episodes": num_episodes,
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

            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()  
            target = env.get_current_target()
            goal_pos = np.array(target) 
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            candidate_path = self.run_firefly_algorithm(start_pos, goal_pos, env.obstacles)
            if candidate_path is None:
                logging.info("萤火虫算法未能找到路径。")
                episode += 1
                continue

            path = self.smooth_path(candidate_path, smoothing_factor=1.0, num_points=200)

            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            # tracking parameter
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
            # tracking loop
            while step_count < max_steps_episode:
                # # Determine whether the target has been reached (with the planning area scale as the threshold)
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # Select the current path point as the desired target
                if path_idx >= len(path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = path[path_idx]
                    if path_idx < len(path) - 1:
                        desired_speed = 3.0  #  [m/s]
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
