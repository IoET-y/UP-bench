# rrt_planner_3d_lqr.py

import numpy as np
import math
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev
import random

from .base import BasePlanner


class RRTPlanner(BasePlanner):
    def __init__(self, grid_resolution=1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0,
                 max_iterations=5000, step_size=2.0, goal_bias=0.05):
        """
        Parameter description:
        - grid_resolution: Sampling resolution for collision detection in space (unit consistent with the environment)
        - max_steps: Maximum number of steps allowed per episode
        - max_lin_accel: Maximum linear acceleration (upper limit of control instructions)
        - collision_threshold: Collision detection threshold
        - max_iterations: Maximum number of RRT sampling times
        - step_size: Step size when RRT is expanded
        - goal_bias: Probability of directly selecting the target point when sampling
        """
        super().__init__(grid_resolution, max_steps, max_lin_accel, collision_threshold, ticks_per_sec=100)

        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.obstacle_radius = 5 


    class Node:
        def __init__(self, pos, parent=None):
        """
        Parameters:
        - pos: 3D coordinate, np.array([x, y, z])
        - parent: parent node (for backtracking path)
        """
            self.pos = pos
            self.parent = parent

    def get_random_point(self, goal):
        """
        Sample a random point according to the target biased sampling strategy
        """
        if random.random() < self.goal_bias:
            return goal.copy()
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        return np.array([x, y, z])

    def get_nearest_node(self, tree, point):
        """
        Find the node closest to point in the tree
        """
        dists = [np.linalg.norm(node.pos - point) for node in tree]
        idx = np.argmin(dists)
        return tree[idx]

    def is_collision_free(self, p1, p2, obstacles):
         """
        Check whether the straight line path from p1 to p2 collides with any obstacle
        Use line segment sampling method to sample a point at a certain step length
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

    def plan_path(self, start, goal, obstacles):
        """
        Use the RRT algorithm to plan a path from start to goal
        Parameters:
        - start: starting point np.array([x, y, z])
        - goal: target np.array([x, y, z])
        - obstacles: list of obstacles (each is [x, y, z])
        Returns:
        - path: list of paths consisting of consecutive coordinate points; returns None if planning fails
        """
        tree = []
        start_node = self.Node(np.array(start))
        tree.append(start_node)
        found = False
        goal_node = None

        for _ in range(self.max_iterations):
            rnd_point = self.get_random_point(np.array(goal))
            nearest_node = self.get_nearest_node(tree, rnd_point)
            direction = rnd_point - nearest_node.pos
            if np.linalg.norm(direction) == 0:
                continue
            direction = direction / np.linalg.norm(direction)
            new_pos = nearest_node.pos + self.step_size * direction

            new_pos[0] = np.clip(new_pos[0], self.x_min, self.x_max)
            new_pos[1] = np.clip(new_pos[1], self.y_min, self.y_max)
            new_pos[2] = np.clip(new_pos[2], self.z_min, self.z_max)

            if not self.is_collision_free(nearest_node.pos, new_pos, obstacles):
                continue

            new_node = self.Node(new_pos, parent=nearest_node)
            tree.append(new_node)

            if np.linalg.norm(new_pos - goal) < self.step_size:
                if self.is_collision_free(new_pos, goal, obstacles):
                    goal_node = self.Node(np.array(goal), parent=new_node)
                    tree.append(goal_node)
                    found = True
                    break

        if not found:
            logging.info("RRT: Failed to find path within maximum number of iterations.")
            return None

        # 从 goal_node 反向回溯构造路径
        path = []
        node = goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()
        return path

    def train(self, env, num_episodes=10):
        """
        After planning the path using the RRT algorithm, use the LQR controller to track the path.
        Reuse the common_train_setup, smooth_path, control_loop, and common_train_cleanup methods in BasePlanner,
        and record the planning time and execution time respectively.
        """
        self.config = {
            "grid_resolution": self.grid_resolution,
            "max_steps": self.max_steps,
            "max_lin_accel": self.max_lin_accel,
            "collision_threshold": self.collision_threshold,
            "max_iterations": self.max_iterations,
            "num_episodes": num_episodes,
            "step_size": self.step_size,
            "goal_bias": self.goal_bias,
            "planning_region": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max]
            }
        }
        self.initialize_wandb("auv_RRT_3D_LQR_planning", "RRT_3D_LQR_run", self.config)

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_RRT_3D_LQR_planning", "RRT_3D_LQR_run", self.config
            )

            plan_start_time = time.time()
            path = self.plan_path(start_pos, goal_pos, env.obstacles)
            plan_end_time = time.time()
            planning_duration = plan_end_time - plan_start_time

            if path is None:
                logging.info("RRT 未能找到路径。")
                episode += 1
                continue

            path = self.smooth_path(path, smoothing_factor=1.0, num_points=200)

            for i in range(len(path) - 1):
                env.env.draw_line(path[i].tolist(), path[i + 1].tolist(), color=[30, 50, 0], thickness=5, lifetime=0)

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

        logging.info("RRT + LQR Planning finished training.")
        return 0, 0, 0, 0, 0, 0
