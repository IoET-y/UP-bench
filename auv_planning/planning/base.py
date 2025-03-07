
# auv_planning/planning/base.py

import numpy as np
import math
import time
import logging
import wandb
import scipy.linalg
from scipy.interpolate import splprep, splev

class BasePlanner:
    def __init__(self, grid_resolution=0.1, max_steps=2000,
                 max_lin_accel=10, collision_threshold=5.0, ticks_per_sec=100):
        """
        Parameters:
          - grid_resolution: resolution for discretizing space (increase to reduce computation)
          - max_steps: maximum steps per episode
          - max_lin_accel: control limit (max linear acceleration)
          - collision_threshold: collision distance threshold
          - ticks_per_sec: simulation frequency
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
        self.ticks_per_sec = ticks_per_sec
        self.ts = 1.0 / self.ticks_per_sec
        self.current_time = 0.0

        # area define x:0~100, y:0~100, z:-100~0）
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        self.z_min = -100
        self.z_max = 0
        self.nx = int((self.x_max - self.x_min) / self.grid_resolution)
        self.ny = int((self.y_max - self.y_min) / self.grid_resolution)
        self.nz = int((self.z_max - self.z_min) / self.grid_resolution)

        # Offline LQR design (same as in BasePlanner)
        A = np.block([
            [np.eye(3), self.ts * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        B = np.block([
            [0.5 * (self.ts ** 2) * np.eye(3)],
            [self.ts * np.eye(3)]
        ])
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1, 0.1])
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # obs map，key grid index，value 1: obs，0: obs free
        self.obstacle_map = {}

    class Node:
        def __init__(self, x, y, z, g=0, h=0, parent=None):
            self.x = x
            self.y = y
            self.z = z
            self.g = g      # cost from starting node to current node
            self.h = h      # heuristic cost
            self.f = g + h  # total cost
            self.parent = parent

        def __lt__(self, other):
            return self.f < other.f

    def create_obstacle_grid(self, obstacles):
        """
        Build a 3D grid map from the obstacles list: 0 for free, 1 for obstacles
        Only mark obstacles within the planned area
        """
        nx = int((self.x_max - self.x_min) / self.grid_resolution)
        ny = int((self.y_max - self.y_min) / self.grid_resolution)
        nz = int((self.z_max - self.z_min) / self.grid_resolution)
        grid = np.zeros((nx, ny, nz), dtype=int)

        for obs in obstacles:
            if not (self.x_min <= obs[0] <= self.x_max and
                    self.y_min <= obs[1] <= self.y_max and
                    self.z_min <= obs[2] <= self.z_max):
                continue
            obs_idx = self.world_to_index(obs)
            radius_in_cells = int(math.ceil(self.obstacle_radius / self.grid_resolution))
            for i in range(max(0, obs_idx[0] - radius_in_cells), min(nx, obs_idx[0] + radius_in_cells + 1)):
                for j in range(max(0, obs_idx[1] - radius_in_cells), min(ny, obs_idx[1] + radius_in_cells + 1)):
                    for k in range(max(0, obs_idx[2] - radius_in_cells), min(nz, obs_idx[2] + radius_in_cells + 1)):
                        cell_center = self.index_to_world((i, j, k))
                        if np.linalg.norm(cell_center - np.array(obs)) <= self.obstacle_radius:
                            grid[i, j, k] = 1
        return grid
    def world_to_index(self, pos):
        """
        Convert continuous world coordinates pos = [x, y, z] to grid indices (ix, iy, iz)
        """
        ix = int((pos[0] - self.x_min) / self.grid_resolution)
        iy = int((pos[1] - self.y_min) / self.grid_resolution)
        iz = int((pos[2] - self.z_min) / self.grid_resolution)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        iz = min(max(iz, 0), self.nz - 1)
        return (ix, iy, iz)

    def index_to_world(self, idx):
        """
Convert grid index (ix, iy, iz) to continuous world coordinates (take cell center)        """
        x = self.x_min + idx[0] * self.grid_resolution + self.grid_resolution / 2.0
        y = self.y_min + idx[1] * self.grid_resolution + self.grid_resolution / 2.0
        z = self.z_min + idx[2] * self.grid_resolution + self.grid_resolution / 2.0
        return np.array([x, y, z])

    def update_obstacle_map_from_sensors(self, current_pos, sensor_readings):
        """
       Update obstacle map based on sensor readings:
        - sensor_readings: list or array of length 14; if reading < 10, it is considered an obstacle
        - According to the preset sensor direction (assuming the agent posture is zero), convert the measurement to world coordinates,
        and then convert to grid index for marking; at the same time, sample along the ray and mark other cells in the field of view as free (0)
        """
        directions = []
        # 8个水平激光（角度0,45,...,315度）
        for i in range(8):
            angle = math.radians(i * 45)
            directions.append(np.array([math.cos(angle), math.sin(angle), 0]))
        # UpRangeSensor
        directions.append(np.array([0, 0, 1]))
        # DownRangeSensor
        directions.append(np.array([0, 0, -1]))
        # UpInclinedRangeSensor：
        directions.append(np.array([math.cos(math.radians(45)), 0, math.sin(math.radians(45))]))
        directions.append(np.array([0, math.cos(math.radians(45)), math.sin(math.radians(45))]))
        # DownInclinedRangeSensor：
        directions.append(np.array([math.cos(math.radians(45)), 0, -math.sin(math.radians(45))]))
        directions.append(np.array([0, math.cos(math.radians(45)), -math.sin(math.radians(45))]))

        updated_cells = []
        max_range = 10.0
        for reading, direction in zip(sensor_readings, directions):
            if reading < max_range:
                obstacle_pos = current_pos + reading * direction
                cell = self.world_to_index(obstacle_pos)
                if self.obstacle_map.get(cell, 0) != 1:
                    self.obstacle_map[cell] = 1
                    updated_cells.append(cell)
            num_samples = int(reading / self.grid_resolution)
            for s in range(num_samples):
                sample_distance = s * self.grid_resolution
                sample_pos = current_pos + sample_distance * direction
                sample_cell = self.world_to_index(sample_pos)
                if self.obstacle_map.get(sample_cell, 0) != 0:
                    self.obstacle_map[sample_cell] = 0
        return updated_cells

    def smooth_path(self, path, smoothing_factor=1.0, num_points=200):
        """
            The discrete path is smoothed using spline interpolation, and the result is bounded to prevent it from exceeding the predetermined area.        """
        if path is None or len(path) < 3:
            return path
        path_array = np.array(path).T  # shape: (3, n)
        tck, u = splprep(path_array, s=smoothing_factor)
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        smooth_path = np.vstack(smooth_points).T
        smooth_path[:, 0] = np.clip(smooth_path[:, 0], self.x_min, self.x_max)
        smooth_path[:, 1] = np.clip(smooth_path[:, 1], self.y_min, self.y_max)
        smooth_path[:, 2] = np.clip(smooth_path[:, 2], self.z_min, self.z_max)
        return [pt for pt in smooth_path]

    def is_path_valid(self, path, current_pos):
        """
        if current_pos to first path point is collision free，
        or obstacle free
        """
        if path is None or len(path) == 0:
            return False
        # 若 current_pos 与路径首个点距离较近，则认为已通过
        if np.linalg.norm(current_pos - path[0]) > self.valid_thresh:
            if not self.collision_free(current_pos, path[0]):
                return False
        for i in range(len(path) - 1):
            if not self.collision_free(path[i], path[i + 1]):
                return False
        return True

    def collision_free(self, p1, p2):
        """
        Check if the straight line between p1 and p2 is collision-free using step sampling.
        Uses BasePlanner's world_to_index.
        """
        dist = np.linalg.norm(p2 - p1)
        steps = int(dist / (self.grid_resolution / 2)) + 1
        for i in range(steps + 1):
            t = i / steps
            p = p1 * (1 - t) + p2 * t
            cell = self.world_to_index(p)
            if self.obstacle_map.get(cell, 0) == 1:
                return False
        return True
    def log_final_metrics(self, reach_target_count, num_episodes):
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
        successrate = reach_target_count / num_episodes
        return successrate, ave_path_length, ave_excu_time, ave_plan_time, ave_smoothness, ave_energy

    def initialize_wandb(self, project_name, run_name, config):
        wandb.init(project=project_name, name=run_name)
        wandb.config.update(config)

    def control_loop(self, env, start_pos, goal_pos, path, desired_speed):
        step_count = 0
        total_path_length = 0.0
        collisions = 0
        energy = 0.0
        smoothness = 0.0
        prev_u = None
        current_pos = start_pos.copy()
        path_idx = 0
        max_steps_episode = self.max_steps

        episode_start_running_time = time.time()

        while step_count < max_steps_episode:
            if np.linalg.norm(current_pos - goal_pos) < 2:
                logging.info("Reached goal.")
                return True, step_count, total_path_length, collisions, energy, smoothness

            if path_idx >= len(path):
                waypoint = goal_pos
                v_des = np.zeros(3)
            else:
                waypoint = path[path_idx]
                if path_idx < len(path) - 1:
                    direction = path[path_idx + 1] - waypoint
                    norm_dir = np.linalg.norm(direction)
                    direction = direction / norm_dir if norm_dir > 1e-6 else np.zeros(3)
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

        return False, step_count, total_path_length, collisions, energy, smoothness

    def common_train_setup(self, env,episode,reach_target_count, project_name, run_name, config):
        self.initialize_wandb(project_name, run_name, config)


        while reach_target_count < 10 and episode < config["num_episodes"]:
            logging.info(f"{run_name} Episode {episode + 1} starting")
            env.reset()

            # 
            init_action = np.zeros(6)
            sensors = env.tick(init_action)
            env.update_state(sensors)
            start_pos = env.location.copy()
            target = env.get_current_target()
            goal_pos = np.array(target)
            logging.info(f"Start: {start_pos}, Goal: {goal_pos}")

            # reset obstacle 
            self.obstacle_map = {}

            episode_start_time = time.time()

            return episode, reach_target_count, start_pos, goal_pos, episode_start_time

    def log_metrics(self, episode, reach_target_count, current_pos, goal_pos, total_path_length, step_count, collisions, energy, smoothness, planning_duration, execution_duration):
        wandb.log({
            "episode": episode + 1,
            "eps_reach_target": reach_target_count,
            "eps_distance_to_goal": np.linalg.norm(current_pos - goal_pos),
            "eps_ave_length_per_step": total_path_length / step_count if step_count > 0 else 0,
            "episode_path_length": total_path_length,
            "episode_collisions": collisions,
            "episode_energy": energy,
            "episode_smoothness": smoothness,
            "episode_planning_duration": planning_duration,
            "episode_execution_duration": execution_duration
        })

    def common_train_cleanup(self, env, episode, reach_target_count, current_pos, goal_pos, episode_start_time, step_count, total_path_length, collisions, energy, smoothness, planning_duration, execution_duration):
        # 
        self.log_metrics(episode, reach_target_count, current_pos, goal_pos, total_path_length, step_count, collisions, energy, smoothness, planning_duration, execution_duration)
        if np.linalg.norm(current_pos - goal_pos) < 2:
            self.ave_path_length += total_path_length
            self.ave_excu_time += execution_duration
            self.ave_plan_time += planning_duration
            self.ave_smoothness += smoothness/100
            self.ave_energy += energy/100

        logging.info(
            f"Episode {episode + 1} completed - Path Length: {total_path_length}, Steps: {step_count}, Collisions: {collisions}")
        episode += 1

        if reach_target_count >= 10 or episode >= self.config["num_episodes"]:
            return self.log_final_metrics(reach_target_count, self.config["num_episodes"])
        env.set_current_target(env.choose_next_target())
        return None  # WHILE NOT FINISH JUST REUTRN None
