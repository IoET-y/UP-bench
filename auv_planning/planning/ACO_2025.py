import numpy as np
import math
import random
import time
import heapq
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

from .base import BasePlanner

class ACOPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0,
                 num_ants=50, iterations=100,
                 alpha=1.0, beta=4.0, evaporation_rate=0.1, Q=100.0,
                 max_path_steps=500):


        # ACO parameter
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.max_path_steps = max_path_steps

        # Obstacle safety radius (used to expand the obstacle area)
        self.obstacle_radius = 5

        # Define 26 neighborhood directions (full 3D neighborhood, excluding the origin)
        self.neighbor_shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.neighbor_shifts.append((dx, dy, dz))
        self.num_neighbors = len(self.neighbor_shifts)


        super().__init__()

    def plan_path(self, start, goal, obstacles):
    """
    Use the ACO algorithm to plan a path from start to goal
    Parameters:
    - start: starting point [x, y, z]
    - goal: target point [x, y, z]
    - obstacles: list of obstacles, each is [x, y, z]
    Returns:
    - path: list of paths consisting of consecutive coordinate points; returns None if planning fails
    """
    # Use create_obstacle_grid() in BasePlanner to build a grid
        grid = self.create_obstacle_grid(obstacles)
        nx, ny, nz = grid.shape

        start_idx = self.world_to_index(start)
        goal_idx = self.world_to_index(goal)

        # If the starting point or target is within the obstacle, return None directly
        if grid[start_idx] == 1 or grid[goal_idx] == 1:
            logging.info("start or end in obstacle area.")
            return None

        # Initialize the pheromone matrix, shape is (nx, ny, nz, num_neighbors)
        pheromone = np.ones((nx, ny, nz, self.num_neighbors), dtype=np.float64)

        # Precompute heuristic information (target distance, avoid division by 0)
        goal_center = self.index_to_world(goal_idx)
        x_coords = self.x_min + (np.arange(nx) + 0.5) * self.grid_resolution
        y_coords = self.y_min + (np.arange(ny) + 0.5) * self.grid_resolution
        z_coords = self.z_min + (np.arange(nz) + 0.5) * self.grid_resolution
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        heuristic = np.sqrt((X - goal_center[0])**2 + (Y - goal_center[1])**2 + (Z - goal_center[2])**2)
        heuristic[heuristic < 1e-6] = 1e-6

        best_path = None
        best_cost = np.inf

        # early stop parameter
        no_improvement = 0
        early_stop_threshold = 10
        improvement_threshold = 1e-3

        # ACO main loop
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
                    if current == goal_idx:
                        reached = True
                        break
                    moves = []
                    probs = []
                    for n_idx, (dx, dy, dz) in enumerate(self.neighbor_shifts):
                        ni = i + dx
                        nj = j + dy
                        nk = k + dz
                        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                            continue
                        if grid[ni, nj, nk] == 1:
                            continue
                        tau = pheromone[i, j, k, n_idx] ** self.alpha
                        eta = (1.0 / heuristic[ni, nj, nk]) ** self.beta
                        moves.append((ni, nj, nk))
                        probs.append(tau * eta)
                    if not moves:
                        break
                    total = sum(probs)
                    probs = [p / total for p in probs]
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
            # Pheromone Evaporation
            pheromone *= (1 - self.evaporation_rate)
            # Pheromone deposition
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
            if best_path is not None and costs and (min(costs) + improvement_threshold) >= best_cost:
                no_improvement += 1
            else:
                no_improvement = 0
            if no_improvement >= early_stop_threshold:
                logging.info(f"Early stopping at iteration {it+1} due to no improvement.")
                break

        if best_path is None:
            logging.info("No path found by ACO.")
            return None

        continuous_path = [self.index_to_world(idx) for idx in best_path]
        return continuous_path


    # -------------------------
    def train(self, env, num_episodes=10):
"""
After planning the path using the ACO algorithm, use the LQR controller to track the planned path.
Process:
1. Call common_train_setup() to reset the environment, get the starting point and target.
2. Record the duration of the planning phase and call plan_path() to get the path.
3. Perform spline smoothing on the planned path (reuse smooth_path() in BasePlanner).
4. Record the duration of the execution phase and call control_loop() for path tracking.
5. Call common_train_cleanup() to record indicators, update cumulative statistics and return results.
"""
        self.config = {
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "num_episodes": num_episodes,
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
                "z": [self.z_min, self.z_max]
            }
        }
        self.initialize_wandb("auv_ACO_3D_LQR_planning", "ACO_3D_LQR_run", self.config)

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_ACO_3D_LQR_planning", "ACO_3D_LQR_run", self.config
            )

            plan_start_time = time.time()
            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            plan_end_time = time.time()
            planning_duration = plan_end_time - plan_start_time

            if path is None:
                logging.info("ACO found no path.")
                episode += 1
                continue

            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

            for i in range(len(path)-1):
                env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[30,50,0], thickness=5, lifetime=0)

            ## Record the execution phase time (call control_loop for path tracking)
            exec_start_time = time.time()
            reached_goal, step_count, total_path_length, collisions, energy, smoothness = self.control_loop(
                env, start_pos, goal_pos, path, desired_speed=3.0
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
            env.set_current_target(env.choose_next_target())

        logging.info("ACO + LQR Planning finished training.")
        return 0, 0, 0, 0, 0, 0
