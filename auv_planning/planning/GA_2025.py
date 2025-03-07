import numpy as np
import math
import random
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

from .base import BasePlanner

class GeneticPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0,
                 population_size=50, generations=100, num_intermediate=10,
                 mutation_rate=0.2, crossover_rate=0.7):


        # GA parameter
        self.population_size = population_size
        self.generations = generations
        self.num_intermediate = num_intermediate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.obstacle_radius = 5

        super().__init__()

    # -------------------------
    # Genetic Algorithm  Methods
    # -------------------------
    def generate_candidate(self, start, goal):
        """
        Generates a candidate path, returning a list containing the start, target, and num_intermediate random intermediate points
        """
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
        Initialize the population, each individual is a candidate path (list form)
        """
        return [self.generate_candidate(start, goal) for _ in range(self.population_size)]

    def is_collision_free(self, p1, p2, obstacles):

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
        Calculate the collision penalty between adjacent points in the candidate path, and accumulate the penalty value if a collision occurs.
        """
        penalty = 0.0
        for i in range(len(candidate) - 1):
            if not self.is_collision_free(candidate[i], candidate[i + 1], obstacles):
                penalty += 1000  # 惩罚值，可调整
        return penalty

    def path_length(self, candidate):
    """
    Calculate the total length of candidate paths (Euclidean distance accumulation)
    """
        length = 0.0
        for i in range(len(candidate) - 1):
            length += np.linalg.norm(candidate[i + 1] - candidate[i])
        return length

    def fitness(self, candidate, obstacles):
         """
        The fitness of the candidate path: path length plus collision penalty, the smaller the value, the better
        """
        return self.path_length(candidate) + self.path_collision_penalty(candidate, obstacles)

    def tournament_selection(self, population, fitnesses, k=3):
        """
        锦标赛选择，从种群中随机选取 k 个候选，返回适应度最好的个体
        """
        selected = random.sample(list(zip(population, fitnesses)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def crossover(self, parent1, parent2):
        """
        Perform a single-point crossover on two parents (cross only the middle point, keeping the starting point and target unchanged)
        """
        child1 = [parent1[0]]
        child2 = [parent2[0]]
        if self.num_intermediate > 0:
            crossover_point = random.randint(1, self.num_intermediate)
            child1 += parent1[1:crossover_point + 1] + parent2[crossover_point + 1:-1]
            child2 += parent2[1:crossover_point + 1] + parent1[crossover_point + 1:-1]
        child1.append(parent1[-1])
        child2.append(parent1[-1])
        return child1, child2

    def mutate(self, candidate):
        """
        Mutate the candidate path, randomly perturb the position of the intermediate point, and keep it within the planning area
        """
        new_candidate = [candidate[0]]
        for pt in candidate[1:-1]:
            if random.random() < self.mutation_rate:
                dx = random.uniform(-0.05 * (self.x_max - self.x_min), 0.05 * (self.x_max - self.x_min))
                dy = random.uniform(-0.05 * (self.y_max - self.y_min), 0.05 * (self.y_max - self.y_min))
                dz = random.uniform(-0.05 * abs(self.z_max - self.z_min), 0.05 * abs(self.z_max - self.z_min))
                new_pt = pt + np.array([dx, dy, dz])
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
        Execute the genetic algorithm to plan the path and return the optimal candidate path (in list form)
        """
        population = self.initialize_population(start, goal)
        best_candidate = None
        best_fitness = float('inf')
        for gen in range(self.generations):
            fitnesses = [self.fitness(candidate, obstacles) for candidate in population]
            for candidate, fit in zip(population, fitnesses):
                if fit < best_fitness:
                    best_fitness = fit
                    best_candidate = candidate
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses, k=3)
                parent2 = self.tournament_selection(population, fitnesses, k=3)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:self.population_size]
        return best_candidate



    def train(self, env, num_episodes=10):
        """
        After using the genetic algorithm to plan the path, use the LQR controller to track the path.
        Process:
        1. Reuse common_train_setup() to reset the environment, get the starting point and target.
        2. Call run_genetic_algorithm() for path planning and record the planning time.
        3. Perform spline smoothing on the candidate path.
        4. Call control_loop() for path tracking and record the execution time.
        5. Reuse common_train_cleanup() to record indicators, update cumulative statistics and return results.
        """
        wandb.init(project="auv_Genetic_3D_LQR_planning", name="Genetic_3D_LQR_run")
        wandb.config.update({
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "population_size": self.population_size,
            "generations": self.generations,
            "num_episodes": num_episodes,
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
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_Genetic_3D_LQR_planning", "Genetic_3D_LQR_run", self.config
            )

            plan_start_time = time.time()
            candidate_path = self.run_genetic_algorithm(start_pos, goal_pos, env.obstacles)
            plan_end_time = time.time()
            planning_duration = plan_end_time - plan_start_time

            if candidate_path is None:
                logging.info("遗传算法未能找到路径。")
                episode += 1
                continue

            path = self.smooth_path(candidate_path, smoothing_factor=1.0, num_points=200)

            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            exec_start_time = time.time()
            reached_goal, step_count, total_path_length, collisions, energy, smoothness = self.control_loop(
                env, start_pos, goal_pos, path, desired_speed=1.0
            )
            exec_end_time = time.time()
            execution_duration = exec_end_time - exec_start_time

            if reached_goal:
                reach_target_count += 1

            result = self.common_train_cleanup(
                env, episode, reach_target_count, env.location, goal_pos,
                episode_start_time, step_count, total_path_length, collisions, energy, smoothness,
                planning_duration, execution_duration
            )
            episode += 1
            if result is not None:
                return result

        logging.info("Genetic GA + LQR Planning finished training.")
        return 0, 0, 0, 0, 0, 0
