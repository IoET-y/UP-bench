# custom_env.py
import holoocean

import numpy as np
from itertools import chain
import random
from .holoocean_config import scenario

class custom_environment:
    def __init__(self, scenario_cfg = scenario, n_targets=0, n_obstacles=0, show_viewport =False, verbose = False):
        """

        Initialize the custom environment.

        Parameters:
        - scenario: Configuration for the environment.
        - n_targets: Number of targets in the environment.
        - n_obstacles: Number of obstacles in the environment.
        """
        random.seed(42)

        # Initialize the environment using holoocean.
        self.env = holoocean.make(scenario_cfg=scenario_cfg,show_viewport=show_viewport, verbose=verbose)

        # Initialize state variables.
        self.pose = np.zeros((4, 4))
        self.prev_location = self.pose[0:3, 3]
        self.location = self.pose[0:3, 3]
        self.rotation = np.zeros((3,))
        self.velocity = np.zeros((3,))
        self.lasers = np.zeros((14,))
        self.observation_space = [self.pose, self.rotation, self.velocity, self.lasers]
        print(len(self.observation_space))
        self.observation_space = [item for sublist in self.observation_space for item in sublist.flatten()]
        # Generate random targets and obstacles.

        self.targets = [self.generate_random_target() for _ in range(n_targets)]
        self.choosen_targets = []
        self.obstacles = [self.generate_random_obstacle() for _ in range(n_obstacles)]

        # Choose the initial target.
        self.current_target = self.choose_next_target()

    def generate_random_target(self):
        """Generate a random target position."""
        return [random.randint(0, 100), random.randint(0, 100), random.randint(-100, 0)]

    def generate_random_obstacle(self):
        """Generate a random obstacle position."""
        return [random.randint(0, 100), random.randint(-100, -0), random.randint(-100, 0)]

    def choose_next_target(self):
        """
        Choose the next target randomly, ensuring it has not been chosen before.

        Returns:
        - target: The chosen target position.
        """
        while True:
            target = random.choice(self.targets)
            if target not in self.choosen_targets:
                self.choosen_targets.append(target)
                return target

    def draw_targets(self):
        """Draw targets in the environment."""
        for i in self.targets:
            if i == self.current_target:
                self.env.draw_point(i, color=[0, 255, 0], thickness=100, lifetime=0)
            else:
                self.env.draw_point(i, color=[255, 255, 0], thickness=100, lifetime=0)

    def draw_obstacles(self):
        """Draw obstacles in the environment."""
        for i in self.obstacles:
            self.env.spawn_prop(prop_type="sphere", location=i, scale=5, material="black")

    def reset(self):
        """Reset the environment."""
        self.env.reset()
        self.env.draw_box(center=[50, 50, -50], extent=[50, 50, 50], thickness=50, lifetime=0)
        self.draw_targets()
        self.draw_obstacles()

    def tick(self, action):
        """
        Perform a simulation step in the environment.

        Parameters:
        - action: Action to be taken in the environment.

        Returns:
        - tick_result: Result of the simulation step.
        """
        self.env.act("auv0", action)
        return self.env.tick()

    def update_state(self, states):
        """
        Update the internal state based on sensor readings.

        Parameters:
        - states: Dictionary containing sensor readings.
        """
        sensors = ["PoseSensor", "VelocitySensor", "RotationSensor", "HorizontalRangeSensor", "UpRangeSensor",
                   "DownRangeSensor", "UpInclinedRangeSensor", "DownInclinedRangeSensor"]

        if all(element in states for element in sensors):
            self.pose = states["PoseSensor"]
            self.rotation = states["RotationSensor"]
            self.velocity = states['VelocitySensor']
            self.lasers = list(chain.from_iterable([states[key] for key in ["HorizontalRangeSensor", "UpRangeSensor",
                                                                            "DownRangeSensor",
                                                                            "UpInclinedRangeSensor",
                                                                            "DownInclinedRangeSensor"]]))
            self.pose = np.array(self.pose)
            self.location = self.pose[0:3, 3]
            self.rotation = np.array(self.rotation) + 180
            self.velocity = np.array(self.velocity)
            self.lasers = np.array(self.lasers)
            self.observation_space = [self.pose, self.rotation, self.velocity, self.lasers]
            self.observation_space = [item for sublist in self.observation_space for item in sublist.flatten()]

    def get_current_target(self):
        """Get the current target position."""
        return self.current_target

    def set_current_target(self, target):
        """
        Set the current target position.

        Parameters:
        - target: New target position.
        """
        self.current_target = target