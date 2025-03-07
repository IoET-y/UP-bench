# dijkstra_planner_3d_lqr.py

import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg

from .base import BasePlanner

class DijkstraPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0):
        super().__init__(grid_resolution, max_steps, max_lin_accel, collision_threshold, ticks_per_sec=100)
        self.obstacle_radius = 5


    class Node:
        def __init__(self, x, y, z, g=0, parent=None):
           """
            Parameters:
            - x, y, z: grid index
            - g: cumulative cost from the starting point (f = g in Dijkstra algorithm)
            - parent: parent node, used for path backtracking
            """
            self.x = x
            self.y = y
            self.z = z
            self.g = g
            self.parent = parent

        def __lt__(self, other):
            return self.g < other.g

    def plan_path(self, start, goal, obstacles):
        """
        Use Dijkstra algorithm to plan a path from start to goal
        Parameters:
        - start: starting point [x, y, z]
        - goal: target point [x, y, z]
        - obstacles: list of obstacles (each is [x, y, z])
        Returns:
        - path: list of paths consisting of consecutive coordinate points (each element is np.array([x, y, z])); if planning fails, returns None
        """
        grid = self.create_obstacle_grid(obstacles)
        nx, ny, nz = grid.shape

        start_idx = self.world_to_index(start)
        goal_idx = self.world_to_index(goal)

        open_list = []
        visited = set()
        start_node = self.Node(start_idx[0], start_idx[1], start_idx[2], g=0, parent=None)
        heapq.heappush(open_list, start_node)
        node_map = {(start_idx[0], start_idx[1], start_idx[2]): start_node}
        found = False
        goal_node = None

        # 26 Neighborhood (including diagonal direction)
        neighbor_shifts = [(dx, dy, dz) for dx in [-1, 0, 1]
                                        for dy in [-1, 0, 1]
                                        for dz in [-1, 0, 1]
                                        if not (dx == 0 and dy == 0 and dz == 0)]

        while open_list:
            current = heapq.heappop(open_list)
            curr_idx = (current.x, current.y, current.z)
            if curr_idx in visited:
                continue
            visited.add(curr_idx)
            if curr_idx == goal_idx:
                found = True
                goal_node = current
                break

            for dx, dy, dz in neighbor_shifts:
                nx_idx = current.x + dx
                ny_idx = current.y + dy
                nz_idx = current.z + dz
                if nx_idx < 0 or nx_idx >= nx or ny_idx < 0 or ny_idx >= ny or nz_idx < 0 or nz_idx >= nz:
                    continue
                if grid[nx_idx, ny_idx, nz_idx] == 1:
                    continue
                neighbor_idx = (nx_idx, ny_idx, nz_idx)
                move_cost = math.sqrt(dx**2 + dy**2 + dz**2)
                new_cost = current.g + move_cost
                if neighbor_idx in node_map:
                    if new_cost < node_map[neighbor_idx].g:
                        node_map[neighbor_idx].g = new_cost
                        node_map[neighbor_idx].parent = current
                else:
                    neighbor_node = self.Node(nx_idx, ny_idx, nz_idx, g=new_cost, parent=current)
                    node_map[neighbor_idx] = neighbor_node
                    heapq.heappush(open_list, neighbor_node)

        if not found:
            logging.info("Dijkstra: 未能找到路径。")
            return None

        # Construct the path by backtracking from the target node
        path_indices = []
        node = goal_node
        while node is not None:
            path_indices.append((node.x, node.y, node.z))
            node = node.parent
        path_indices.reverse()

        # Convert grid index to continuous coordinates (take cell center)
        path = [self.index_to_world(idx) for idx in path_indices]
        return path

    def train(self, env, num_episodes=10):
        """
        Use Dijkstra algorithm to plan the path, and then use LQR controller to track the path.
        Adopt the modular design in BasePlanner, reuse common_train_setup, smooth_path, control_loop, common_train_cleanup and other methods,
        and record the time of path planning and control execution respectively.
        """
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
        self.initialize_wandb("auv_Dijkstra_3D_LQR_planning", "Dijkstra_3D_LQR_run", self.config)

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
           # Public training settings: reset the environment, get the starting point and target
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_Dijkstra_3D_LQR_planning", "Dijkstra_3D_LQR_run", self.config
            )

            # Record planning time
            plan_start_time = time.time()
            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            plan_end_time = time.time()
            planning_duration = plan_end_time - plan_start_time

            if path is None:
                logging.info("Dijkstra 未能找到路径。")
                episode += 1
                continue

            #Smooth the planned path (reuse smooth_path in BasePlanner)
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

           
            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i+1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            exec_start_time = time.time()
            reached_goal, step_count, total_path_length, collisions, energy, smoothness = self.control_loop(
                env, start_pos, goal_pos, path, desired_speed=3.0
            )
            exec_end_time = time.time()
            execution_duration = exec_end_time - exec_start_time

            if reached_goal:
                reach_target_count += 1

            # training cleanup: record metrics and update cumulative statistics
            result = self.common_train_cleanup(
                env, episode, reach_target_count, env.location, goal_pos,
                episode_start_time, step_count, total_path_length, collisions, energy, smoothness,
                planning_duration, execution_duration
            )
            episode += 1
            if result is not None:
                return result

        logging.info("Dijkstra + LQR Planning finished training.")
        return 0, 0, 0, 0, 0, 0
