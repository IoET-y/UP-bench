import numpy as np
from auv_control import State
from .base import BasePlanner
import heapq
import random

class ACO_AStar(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=30, start=None, end=None, speed=None, num_ants=50, num_iterations=10):
        # Setup goal
        self.start = np.array([0, 0, -5]) if start is None else start
        self.end = np.array([50, 40, -20]) if end is None else end
        self.num_seconds = num_seconds
        self.speed = speed
        self.num_ants = num_ants
        self.num_iterations = num_iterations

        # Setup environment
        self.size = np.array([50, 50, 25])
        self.bottom_corner = np.array([-5, -5, -25])
        # self.top_corner = self.bottom_corner + self.size  # Added definition for top_corner

        # Setup obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_size = np.random.uniform(2, 5, self.num_obstacles)
        self.obstacle_loc = np.random.beta(1.5, 1.5, (num_obstacles, 3)) * self.size + self.bottom_corner
        # Ensure no obstacles too close to start or end
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or \
                    np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner

        # Initialize path and trajectory functions
        self.path = None
        self.pos_func = lambda t: self.start
        self.rot_func = lambda t: np.zeros(3)

        # Run A* to get an initial path
        self.step_size = 1
        self._run_astar()

        # If A* finds a path, use ACO to optimize it
        if self.path is not None:
            self._run_aco()

    def _run_astar(self):
        open_set = []
        heapq.heappush(open_set, (0, tuple(self.start)))
        came_from = {}
        g_score = {tuple(self.start): 0}
        f_score = {tuple(self.start): self._heuristic(self.start, self.end)}

        while open_set:
            _, current = heapq.heappop(open_set)
            current = np.array(current)

            if np.linalg.norm(current - self.end) < self.step_size:
                self._reconstruct_path(came_from, current)
                return

            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[tuple(current)] + np.linalg.norm(neighbor - current)

                if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                    came_from[tuple(neighbor)] = current
                    g_score[tuple(neighbor)] = tentative_g_score
                    f_score[tuple(neighbor)] = tentative_g_score + self._heuristic(neighbor, self.end)
                    heapq.heappush(open_set, (f_score[tuple(neighbor)], tuple(neighbor)))
    def _reconstruct_path(self, came_from, current):
        # Reconstruct the path from the end to the start
        path = [current]
        while tuple(current) in came_from:
            current = came_from[tuple(current)]
            path.append(current)

        self.path = np.array(path[::-1])
        self.path = np.vstack((self.path, self.end))
        # After reconstructing, set the trajectory functions
        self._set_trajectory_functions()
        
    def _run_aco(self):
        # Initialize pheromones with multiple initial paths
        pheromones = {tuple(pos): 1.0 for pos in self.path}
        
        best_path = self.path
        best_path_length = self._calculate_path_length(self.path)

        # Run ACO for the specified number of iterations
        for iteration in range(self.num_iterations):
            all_paths = []
            all_lengths = []

            # Each ant generates a path
            for ant in range(self.num_ants):
                path = self._generate_ant_path(pheromones)
                if path is None:
                    continue  # Skip if path could not reach the goal
                path_length = self._calculate_path_length(path)

                all_paths.append(path)
                all_lengths.append(path_length)

                # Update best path if the new one is better
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            # Update pheromones
            pheromones = self._update_pheromones(pheromones, all_paths, all_lengths)

            # Log iteration info
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Path Length: {best_path_length}")

        self.path = best_path

        # Update pos_func and rot_func if a valid path is found
        self._set_trajectory_functions()

    def _generate_ant_path(self, pheromones):
        current = self.start
        path = [current]

        max_steps = 1000  # Prevent ants from looping indefinitely
        step_count = 0

        while np.linalg.norm(current - self.end) >= self.step_size and step_count < max_steps:
            neighbors = self._get_neighbors(current)
            if not neighbors:
                return None

            # Choose the next node based on pheromones and distance (Exploration vs Exploitation)
            probabilities = []
            for neighbor in neighbors:
                pheromone = pheromones.get(tuple(neighbor), 1.0)
                distance = np.linalg.norm(neighbor - current)
                probabilities.append((pheromone ** 1.0) / (distance ** 1.0 + 1e-5))

            probabilities = np.array(probabilities) / np.sum(probabilities)
            next_idx = np.random.choice(len(neighbors), p=probabilities)
            current = neighbors[next_idx]
            path.append(current)
            step_count += 1

        if np.linalg.norm(current - self.end) < self.step_size:
            path.append(self.end)
            return np.array(path)
        return None

    def _update_pheromones(self, pheromones, all_paths, all_lengths):
        evaporation_rate = 0.9  # Evaporation rate to control pheromone decay
        # Evaporate pheromones
        for key in pheromones:
            pheromones[key] *= evaporation_rate

        # Add new pheromones based on path quality
        for path, length in zip(all_paths, all_lengths):
            pheromone_deposit = 1.0 / length
            for point in path:
                key = tuple(point)
                pheromones[key] = pheromones.get(key, 0) + pheromone_deposit

        return pheromones

    def _calculate_path_length(self, path):
        return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    def _set_trajectory_functions(self):
        distance = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        if self.speed is None:
            self.speed = np.sum(distance) / (self.num_seconds - 3)
        times = np.cumsum(distance / self.speed)

        def rot(t):
            step = np.searchsorted(times, t)

            if step + 1 >= len(self.path):
                return np.zeros(3)
            else:
                t = t - times[step - 1] if step > 0 else t

                p_prev = self.path[step]
                p_next = self.path[step + 1]

                m = self.speed * (p_next - p_prev) / np.linalg.norm(p_next - p_prev)

                yaw = np.arctan2(m[1], m[0]) * 180 / np.pi
                pitch = -np.arctan2(m[2], np.sqrt(m[0] ** 2 + m[1] ** 2)) * 180 / np.pi
                pitch = np.clip(pitch, -15, 15)

                return np.array([0, pitch, yaw])

        def pos(t):
            step = np.searchsorted(times, t)

            if step + 1 >= len(self.path):
                return self.end
            else:
                t = t - times[step - 1] if step > 0 else t

                p_prev = self.path[step]
                p_next = self.path[step + 1]

                m = self.speed * (p_next - p_prev) / np.linalg.norm(p_next - p_prev)
                return m * t + p_prev

        # Update pos_func and rot_func
        self.pos_func = np.vectorize(pos, signature='()->(n)')
        self.rot_func = np.vectorize(rot, signature='()->(n)')

    def _get_neighbors(self, current):
        # Adding diagonal directions for more flexibility
        directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                      np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1]),
                      np.array([1, 1, 0]), np.array([-1, -1, 0]), np.array([1, 0, 1]),
                      np.array([0, 1, 1]), np.array([-1, 0, 1]), np.array([0, -1, 1])]

        neighbors = []
        for direction in directions:
            neighbor = current + direction * self.step_size
            if np.all(neighbor >= self.bottom_corner) and np.all(neighbor <= self.top_corner):
                if not self._collision(current, neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def _heuristic(self, start, end):
        # Euclidean distance as a heuristic
        return np.linalg.norm(start - end)

    def _collision(self, start, end):
        # Checking for collision by interpolating between start and end points
        vals = np.linspace(start, end, int(np.linalg.norm(end - start) / 0.5))
        for v in vals:
            dist = np.linalg.norm(self.obstacle_loc - v, axis=1)
            if np.any(dist < self.obstacle_size + 1):
                return True
        return False

    @property
    def center(self):
        return self.bottom_corner + self.size / 2

    @property
    def top_corner(self):
        return self.bottom_corner + self.size

    def draw_traj(self, env, t):
        """Override superclass to also make the environment appear"""
        if self.path is None:
            print("No valid path found.")
            return

        # Setup environment visualization
        env.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0)
        for i in range(self.num_obstacles):
            loc = self.obstacle_loc[i].tolist()
            loc[1] *= -1
            env.spawn_prop('sphere', loc, [0, 0, 0], self.obstacle_size[i], False, "white")

        # Draw the planned path
        for p in self.path:
            env.draw_point(p.tolist(), color=[255, 0, 0], thickness=20, lifetime=0)

        super().draw_traj(env, t)