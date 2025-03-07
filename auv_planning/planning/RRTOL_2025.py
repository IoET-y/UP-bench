import numpy as np
import math
import heapq
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev  # for path smoothing

from .base import BasePlanner

class OnlineRRTStarPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.5, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 step_size=2.0, neighbor_radius=5.0, init_iter=200, iter_per_cycle=50,
                 local_radius=15.0, valid_thresh=1.0):


        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.init_iter = init_iter
        self.iter_per_cycle = iter_per_cycle
        self.local_radius = local_radius
        self.valid_thresh = valid_thresh



        # RRT* parameters and tree initialization
        self.tree = []  # list of Node objects
        self.obstacle_map = {}  # reused by BasePlanner methods


        super().__init__(grid_resolution, max_steps, max_lin_accel, collision_threshold, ticks_per_sec)

        # Define 26 neighbor shifts (3D full neighborhood excluding zero)
        self.neighbor_shifts = [(dx, dy, dz) for dx in [-1, 0, 1]
                                          for dy in [-1, 0, 1]
                                          for dz in [-1, 0, 1]
                                          if not (dx == 0 and dy == 0 and dz == 0)]
        self.num_neighbors = len(self.neighbor_shifts)

    class Node:
        def __init__(self, pos, parent=None, cost=0.0):
            self.pos = pos  # np.array([x, y, z])
            self.parent = parent
            self.cost = cost

    def sample_local(self, center):
        """Sample a point uniformly within local_radius of center (clipped to planning region)."""
        sample = center + np.random.uniform(-self.local_radius, self.local_radius, size=3)
        sample[0] = np.clip(sample[0], self.x_min, self.x_max)
        sample[1] = np.clip(sample[1], self.y_min, self.y_max)
        sample[2] = np.clip(sample[2], self.z_min, self.z_max)
        return sample

    def nearest_node(self, q_rand):
        """Return the node in the tree closest to q_rand."""
        return min(self.tree, key=lambda node: np.linalg.norm(node.pos - q_rand))

    def steer(self, q_near, q_rand):
        """Return a point in the direction from q_near to q_rand, not exceeding step_size."""
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_rand
        return q_near + (direction/dist) * self.step_size

    def near_nodes(self, q_new):
        """Return nodes in the tree within neighbor_radius of q_new."""
        return [node for node in self.tree if np.linalg.norm(node.pos - q_new) <= self.neighbor_radius]



    def grow_tree(self, iterations, sample_center):
        """
        Grow the RRT* tree for a given number of iterations using local sampling around sample_center.
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
            # Rewire neighbors if a lower cost path exists through q_new
            for neighbor in neighbors:
                if neighbor == q_new:
                    continue
                if self.collision_free(q_new.pos, neighbor.pos):
                    temp_cost = q_new.cost + np.linalg.norm(neighbor.pos - q_new.pos)
                    if temp_cost < neighbor.cost:
                        neighbor.parent = q_new
                        neighbor.cost = temp_cost

    def prune_tree(self, current_pos):
        """Prune nodes that are farther than local_radius*2 from current_pos."""
        self.tree = [node for node in self.tree if np.linalg.norm(node.pos - current_pos) < self.local_radius * 2]

    def get_best_path(self, start, goal):
        """
        Find the node closest to goal and return the path from start to that node.
        If the segment from the best node to goal is collision-free, append goal.
        """
        best_node = min(self.tree, key=lambda node: np.linalg.norm(node.pos - goal), default=None)
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
        return path[::-1]

    def train(self, env, num_episodes=10):
        """
        Online RRT* with LQR tracking.
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
            "step_size": self.step_size,
            "neighbor_radius": self.neighbor_radius,
            "init_iter": self.init_iter,
            "iter_per_cycle": self.iter_per_cycle,
            "local_radius": self.local_radius,
            "valid_thresh": self.valid_thresh
        }
        self.initialize_wandb("auv_RRTStarOnline_planning", "RRTStarOnline_run", self.config)

        episode = 0
        reach_target_count = 0
        global_path = None

        while reach_target_count < 10 and episode < num_episodes:
            logging.info(f"Episode {episode+1} starting")
            # Common training setup: reset env and get start/goal state.
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_RRTStarOnline_planning", "RRTStarOnline_run", self.config
            )

            # Reset tree and obstacle map
            self.obstacle_map = {}
            self.tree = []
            root = self.Node(start_pos, parent=None, cost=0.0)
            self.tree.append(root)

            # Initial global planning
            self.grow_tree(self.init_iter, start_pos)
            global_path = self.get_best_path(start_pos, goal_pos)
            if global_path is None:
                logging.info("Initial global path not found.")
                episode += 1
                continue

            # Smooth global path using BasePlanner's smooth_path
            global_path = self.smooth_path(global_path, smoothing_factor=1.0, num_points=200)
            for i in range(len(global_path)-1):
                env.env.draw_line(global_path[i].tolist(), global_path[i+1].tolist(), color=[30,50,0], thickness=5, lifetime=0)

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

                # Update obstacle map using BasePlanner method
                sensor_readings = env.lasers.copy()
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)
                self.prune_tree(current_pos)

                # Remove waypoints already passed
                while global_path and np.linalg.norm(current_pos - global_path[0]) < self.valid_thresh:
                    global_path.pop(0)

                # Use global path if valid, else replan locally
                if global_path is not None and self.is_path_valid(global_path, current_pos):
                    updated_path = global_path
                else:
                    local_target = global_path[0] if global_path and len(global_path) > 0 else goal_pos
                    local_segment = self.get_best_path(current_pos, local_target)
                    if local_segment is not None:
                        local_segment = self.smooth_path(local_segment, smoothing_factor=1.0, num_points=200)
                        updated_path = local_segment[:-1] + global_path if global_path is not None else local_segment
                        global_path = updated_path
                    else:
                        logging.info("No valid local path found.")
                        break

                # Select current waypoint
                if path_idx >= len(updated_path):
                    waypoint = goal_pos
                    v_des = np.zeros(3)
                else:
                    waypoint = updated_path[path_idx]
                    if path_idx < len(updated_path)-1:
                        desired_speed = 3.0  # m/s
                        direction = updated_path[path_idx+1] - waypoint
                        norm_dir = np.linalg.norm(direction)
                        direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
                        v_des = desired_speed * direction
                    else:
                        v_des = np.zeros(3)
                    if np.linalg.norm(current_pos - waypoint) < self.valid_thresh:
                        path_idx += 1
                        continue

                # LQR control
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
                energy += (np.linalg.norm(u)**2) / 100
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

                for obs in env.obstacles:
                    if np.linalg.norm(new_pos - np.array(obs)) < self.collision_threshold:
                        collisions += 1
                        break

                env.env.draw_line(current_pos.tolist(), new_pos.tolist(), color=[0,100,0], thickness=3, lifetime=0)
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

        logging.info("Online RRT* Planning finished training.")
        return 0, 0, 0, 0, 0, 0
