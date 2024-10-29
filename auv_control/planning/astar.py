import numpy as np
from auv_control import State
from .base import BasePlanner
import heapq

class Astar(BasePlanner):
    def __init__(self, num_seconds, num_obstacles=30, start=None, end=None, speed=None):
        # Setup goal
        self.start = np.array([0, 0, 0]) if start is None else start
        self.end = np.array([50, 40, -20]) if end is None else end
        self.num_seconds = num_seconds
        self.speed = speed

        # Setup environment
        self.size = np.array([50, 50, 25])
        self.bottom_corner = np.array([-5, -5, -25])

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

        # Run A* to find path
        self.step_size = 50
        self._run_astar()

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
        path = [current]
        while tuple(current) in came_from:
            current = came_from[tuple(current)]
            path.append(current)

        self.path = np.array(path[::-1])
        self.path = np.vstack((self.path, self.end))

        # Smooth the waypoint path
        smooth = [0]
        while smooth[-1] < len(self.path) - 1:
            for i in range(smooth[-1] + 1, len(self.path)):
                if self._collision(self.path[smooth[-1]:smooth[-1] + 1], self.path[i:i + 1]):
                    smooth.append(i - 1)
                    break
                if i == len(self.path) - 1:
                    smooth.append(len(self.path) - 1)
                    break

        self.path = self.path[smooth]

        # Make rotation and position functions
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

        # Update pos_func and rot_func if a valid path is found
        self.pos_func = np.vectorize(pos, signature='()->(n)')
        self.rot_func = np.vectorize(rot, signature='()->(n)')

    def _get_neighbors(self, current):
        directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                      np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
        neighbors = []
        for direction in directions:
            neighbor = current + direction * self.step_size
            if np.all(neighbor >= self.bottom_corner) and np.all(neighbor <= self.top_corner):
                if not self._collision(current, neighbor):
                    neighbors.append(neighbor)
        return neighbors

    # def _heuristic(self, start, end):
    #     return np.linalg.norm(start - end)
    def _heuristic(self, start, end):
        euclidean_dist = np.linalg.norm(start - end)
        obstacle_penalty = 0
        for obstacle in self.obstacle_loc:
            dist_to_obstacle = np.linalg.norm(start - obstacle)
            if dist_to_obstacle < 5:  # Arbitrary threshold for proximity
                obstacle_penalty += (5 - dist_to_obstacle)  # Higher penalty the closer to an obstacle
        return euclidean_dist + obstacle_penalty

    def _collision(self, start, end):
        vals = np.linspace(start, end, 50)
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
        """Override super class to also make environment appear"""
        if self.path is None:
            print("No valid path found.")
            return

        # Setup environment
        env.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0)
        for i in range(self.num_obstacles):
            loc = self.obstacle_loc[i].tolist()
            loc[1] *= -1
            env.spawn_prop('sphere', loc, [0, 0, 0], self.obstacle_size[i], False, "white")

        for p in self.path:
            env.draw_point(p.tolist(), color=[255, 0, 0], thickness=20, lifetime=0)

        super().draw_traj(env, t)