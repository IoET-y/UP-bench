# auv_planning/planning/ASTAR_2025.py

import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

from .base import BasePlanner

class AStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0):
        super().__init__(grid_resolution, max_steps, max_lin_accel, collision_threshold)

    def plan_path(self, start, goal, obstacles):
        """
        Use 3D A* algorithm to plan a path from start to goal
        Parameters:
        - start: starting point [x, y, z]
        - goal: target point [x, y, z]
        - obstacles: list of obstacles (each is [x, y, z])
        Return:
        - path: list of paths consisting of consecutive coordinate points (each element is np.array([x,y,z])); if planning fails, returns None
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

        path_indices = []
        node = goal_node
        while node is not None:
            path_indices.append((node.x, node.y, node.z))
            node = node.parent
        path_indices.reverse()

        path = []
        for idx in path_indices:
            pos = self.index_to_world(idx)
            path.append(pos)
        return path

    def train(self, env, num_episodes=10):
        self.config = {
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "num_episodes": num_episodes,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max]
            }
        }
        self.initialize_wandb("auv_AStar_3D_LQR_planning", "AStar_3D_LQR_run", self.config)

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_AStar_3D_LQR_planning", "AStar_3D_LQR_run", self.config
            )

            plan_start_time = time.time()
            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            plan_end_time = time.time()
            planning_duration = plan_end_time - plan_start_time

            if path is None:
                logging.info("3D A* did not find a path.")
                episode += 1
                continue

            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

            
            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            # Record the execution time (control loop) separately
            exec_start_time = time.time()
            reached_goal, step_count, total_path_length, collisions, energy, smoothness = self.control_loop(
                env, start_pos, goal_pos, path, desired_speed=3.0
            )
            exec_end_time = time.time()
            execution_duration = exec_end_time - exec_start_time

            if reached_goal:
                reach_target_count += 1

            result = self.common_train_cleanup(
                env,
                episode, reach_target_count,
                env.location, goal_pos,
                episode_start_time, step_count, total_path_length,
                collisions, energy, smoothness,
                planning_duration, execution_duration
            )
            episode += 1
            if result is not None:
                return result

        logging.info(f"{self.__class__.__name__} Planning finished training.")
        return 0, 0, 0, 0, 0, 0  
