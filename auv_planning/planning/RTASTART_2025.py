import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # for path smoothing

from .base import BasePlanner


class RTAAStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.5, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 lookahead=50, valid_thresh = 1):
        """
        Parameters:
          - lookahead: maximum node expansions per RTAA* search
        """
        self.lookahead = lookahead
        self.H = {}  # learned heuristic values (key: grid index)
        self.obstacle_map = {}  # key: grid index, value: occupancy (1: obstacle, 0: free)
        self.valid_thresh = valid_thresh

        # Precompute 26 neighbor shifts (3D full neighborhood)
        self.neighbor_shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.neighbor_shifts.append((dx, dy, dz))
        super().__init__(grid_resolution, max_steps, max_lin_accel, collision_threshold, ticks_per_sec)

    # --- RTAA* Node class ---
    class Node:
        def __init__(self, idx, g, h, parent):
            self.idx = idx  # grid index
            self.g = g  # cost from start
            self.h = h  # heuristic value
            self.f = g + h  # total cost
            self.parent = parent

        def __lt__(self, other):
            return self.f < other.f

    # --- Get neighboring grid indices ---
    def get_neighbors(self, idx):
        neighbors = []
        for shift in self.neighbor_shifts:
            n_idx = (idx[0] + shift[0], idx[1] + shift[1], idx[2] + shift[2])
            if 0 <= n_idx[0] < self.nx and 0 <= n_idx[1] < self.ny and 0 <= n_idx[2] < self.nz:
                neighbors.append(n_idx)
        return neighbors

    # --- Heuristic and cost functions ---
    def heuristic(self, a, b):
        return self.grid_resolution * math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def cost(self, a, b):
        # If either cell is an obstacle, cost is infinite
        if self.obstacle_map.get(a, 0) == 1 or self.obstacle_map.get(b, 0) == 1:
            return float('inf')
        return self.heuristic(a, b)

    # --- RTAA* limited expansion search ---
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
            path = []
            node = goal_node
            while node is not None:
                path.append(node.idx)
                node = node.parent
            path.reverse()
            return path
        if open_list:
            best_node = min(open_list, key=lambda n: n.f)
            self.H[start] = best_node.f  # update learned heuristic at start
            # Choose the best successor from start's neighbors
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

    # --- Global path planning ---
    def plan_path(self, start, goal):
        self.start = self.world_to_index(start)
        self.goal = self.world_to_index(goal)
        self.H = {}  # reset learned heuristics
        path_indices = self.rtaa_search(self.start, self.goal, self.lookahead * 10)
        if path_indices is None:
            return None
        return [self.index_to_world(idx) for idx in path_indices]

    # --- Training loop ---
    def train(self, env, num_episodes=10):
        """
        Online RTAA* planning with LQR tracking.
        Uses common_train_setup and common_train_cleanup from BasePlanner.
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
            },
            "lookahead": self.lookahead
        }
        self.initialize_wandb("auv_RTAAStar_planning", "RTAAStar_run", self.config)

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            logging.info(f"RTAA* Episode {episode + 1} starting")
            # Common training setup: reset env and get start/goal state.
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_RTAAStar_planning", "RTAAStar_run", self.config
            )

            # Global planning (using RTAA* search)
            path = self.plan_path(start_pos, goal_pos)
            if path is None:
                logging.info("RTAA* did not find an initial path.")
                episode += 1
                continue

            # Smooth path using BasePlanner's smooth_path method
            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)
            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

            step_count = 0
            total_path_length = 0.0
            collisions = 0
            energy = 0.0
            smoothness = 0.0
            prev_u = None
            current_pos = start_pos.copy()
            path_idx = 0
            episode_planning_duration = time.time() - episode_start_time
            episode_start_running_time = time.time()

            # Tracking control loop with online replanning
            while step_count < self.max_steps:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # Update obstacle map using BasePlanner's method
                sensor_readings = env.lasers.copy()  # assume 14 sensor readings
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # Remove passed waypoints
                while path and np.linalg.norm(current_pos - path[0]) < self.valid_thresh:
                    path.pop(0)

                # Use global path if valid, else replan locally
                if path is not None and self.is_path_valid(path, current_pos):
                    updated_path = path
                else:
                    local_target = path[0] if path and len(path) > 0 else goal_pos
                    local_segment = self.rtaa_search(self.world_to_index(current_pos),
                                                     self.world_to_index(local_target), self.lookahead)
                    if local_segment is not None:
                        local_segment = [self.index_to_world(idx) for idx in local_segment]
                        updated_path = self.smooth_path(local_segment, smoothing_factor=1.0, num_points=200)
                        path = updated_path
                    else:
                        logging.info("No valid local path found, stopping episode.")
                        break

                if path_idx >= len(updated_path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = updated_path[path_idx]
                    if path_idx < len(updated_path) - 1:
                        desired_speed = 3.0  # m/s
                        direction = updated_path[path_idx + 1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < 1:
                        path_idx += 1
                        continue

                # LQR control computation
                x_current = np.hstack([current_pos, env.velocity.copy()])
                x_des = np.hstack([waypoint, v_des])
                error_state = x_current - x_des
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                action = np.concatenate([u, np.zeros(3)])
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()

                dist_moved = np.linalg.norm(new_pos - current_pos)
                total_path_length += dist_moved
                energy += (np.linalg.norm(u) ** 2) / 100
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

                for obs in env.obstacles:
                    if np.linalg.norm(new_pos - np.array(obs)) < self.collision_threshold:
                        collisions += 1
                        break

                env.env.draw_line(current_pos.tolist(), new_pos.tolist(), color=[0, 100, 0], thickness=3, lifetime=0)
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
            result = self.common_train_cleanup(
                env, episode, reach_target_count, env.location, goal_pos,
                episode_start_time, step_count, total_path_length, collisions, energy, smoothness,
                episode_planning_duration, episode_running_duration
            )
            episode += 1
            if result is not None:
                return result
            env.set_current_target(env.choose_next_target())

        logging.info("RTAA* Planning finished training.")
        return 0, 0, 0, 0, 0, 0
