import numpy as np
import math
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

from .base import BasePlanner

class PFMPlanner(BasePlanner):
    def __init__(self, grid_resolution=0.1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100,
                 k_att=1.0, k_rep=100.0, d0=10.0, desired_speed=3.0):
        """
        Parameters:
          - grid_resolution: resolution for obstacle detection
          - max_steps: max steps per episode
          - max_lin_accel: maximum linear acceleration (control limit)
          - collision_threshold: threshold distance for collisions
          - ticks_per_sec: simulation frequency
          - k_att: attractive coefficient
          - k_rep: repulsive coefficient
          - d0: influence distance for repulsion
          - desired_speed: desired speed for control
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.desired_speed = desired_speed

        # PFM-specific obstacle map (key: grid index, value: occupancy)
        self.obstacle_map = {}

        # Call BasePlanner initializer to setup region, LQR, etc.
        super().__init__(grid_resolution, max_steps, max_lin_accel, collision_threshold, ticks_per_sec)

    def compute_potential_force(self, current_pos, goal_pos):
        """
        Compute total potential force at current_pos.
        Attractive force: F_attr = k_att * (goal - current)
        Repulsive force: for each obstacle cell within d0, add repulsion.
        """
        F_attr = self.k_att * (goal_pos - current_pos)
        F_rep = np.zeros(3)
        for cell, occ in self.obstacle_map.items():
            if occ == 1:
                obs_pos = self.index_to_world(cell)
                diff = current_pos - obs_pos
                d = np.linalg.norm(diff)
                if d < self.d0 and d > 1e-3:
                    rep_force = self.k_rep * (1.0/d - 1.0/self.d0) / (d**2) * (diff / d)
                    F_rep += rep_force
        return F_attr + F_rep

    def train(self, env, num_episodes=10):
        """
        Train the planner using potential fields with LQR control.
        Reuses common_train_setup and common_train_cleanup for modularity.
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
            "k_att": self.k_att,
            "k_rep": self.k_rep,
            "d0": self.d0,
            "desired_speed": self.desired_speed
        }
        self.initialize_wandb("auv_PFM_planning", "PFM_run", self.config)

        episode = 0
        reach_target_count = 0

        while reach_target_count < 10 and episode < num_episodes:
            # Common training setup: reset env, get start and goal, and start time.
            episode, reach_target_count, start_pos, goal_pos, episode_start_time = self.common_train_setup(
                env, episode, reach_target_count, "auv_PFM_planning", "PFM_run", self.config
            )

            # For potential field, planning stage is minimal.
            planning_duration = 0.0

            exec_start_time = time.time()

            step_count = 0
            total_path_length = 0.0
            collisions = 0
            energy = 0.0
            smoothness = 0.0
            prev_u = None
            current_pos = start_pos.copy()

            # Control loop using potential field force
            while step_count < self.max_steps:
                if np.linalg.norm(current_pos - goal_pos) < 2:
                    logging.info("Reached goal.")
                    reach_target_count += 1
                    break

                # Update obstacle map using sensor readings (from BasePlanner)
                sensor_readings = env.lasers.copy()  # expects 14 sensor readings
                self.update_obstacle_map_from_sensors(current_pos, sensor_readings)

                # Compute potential field force
                F_total = self.compute_potential_force(current_pos, goal_pos)
                norm_F = np.linalg.norm(F_total)
                direction = F_total / norm_F if norm_F > 1e-6 else np.zeros(3)
                v_des = self.desired_speed * direction
                pos_des = current_pos + v_des * self.ts

                # Construct current and desired state vectors
                x_current = np.hstack([current_pos, env.velocity.copy()])
                x_des = np.hstack([pos_des, v_des])
                error_state = x_current - x_des

                # LQR control input
                u = -self.K.dot(error_state)
                u = np.clip(u, -self.max_lin_accel, self.max_lin_accel)
                action = np.concatenate([u, np.zeros(3)])
                sensors = env.tick(action)
                env.update_state(sensors)
                new_pos = env.location.copy()

                distance_moved = np.linalg.norm(new_pos - current_pos)
                total_path_length += distance_moved
                energy += np.linalg.norm(u)**2
                if prev_u is not None:
                    smoothness += np.linalg.norm(u - prev_u)
                prev_u = u

                # Check for collisions with obstacles
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
                    "distance_to_goal": np.linalg.norm(current_pos - goal_pos)
                })

            exec_end_time = time.time()
            execution_duration = exec_end_time - exec_start_time

            result = self.common_train_cleanup(
                env, episode, reach_target_count, env.location, goal_pos,
                episode_start_time, step_count, total_path_length, collisions, energy, smoothness,
                planning_duration, execution_duration
            )
            episode += 1
            if result is not None:
                return result
            env.set_current_target(env.choose_next_target())

        logging.info("PFM Planning finished training.")
        return 0, 0, 0, 0, 0, 0
