import numpy as np
from scipy.spatial.transform import Rotation

class SE3:
    """
    Custom SE3 representation for transformation matrices, velocities, and augmentations.
    """
    def __init__(self, mat=None, velocity=None, aug=None):
        self.mat = np.eye(4) if mat is None else mat  # Transformation matrix (4x4)
        self.velocity = np.zeros(3) if velocity is None else velocity  # Linear velocity
        self.aug = np.zeros(6) if aug is None else aug  # Augmented state (bias, angular velocity, etc.)

    def __getitem__(self, idx):
        """
        Indexing for SE3.
        - 0: Velocity
        - 1: Position
        """
        if idx == 0:
            return self.velocity
        elif idx == 1:
            return self.mat[:3, 3]  # Position vector (translation part)
        else:
            raise IndexError("Index out of range for SE3")

    def __repr__(self):
        return f"SE3(mat={self.mat}, velocity={self.velocity}, aug={self.aug})"


class State:
    """A uniform representation of our state from various sources.

    Can come from a dictionary (HoloOcean), SE[2,6] object (custom SE3), or from a simple
    numpy array.

    State saved consists of position, velocity, rotation, angular velocity, and IMU bias.
    """
    def __init__(self, state, last_meas_omega=None):
        self.vec = np.zeros(12)
        self.mat = np.eye(5)
        self.bias = np.zeros(6)
        self.add = np.zeros(3)

        # True State
        if isinstance(state, dict):
            self.vec[0:3] = state["PoseSensor"][:3, 3]
            self.vec[3:6] = state["VelocitySensor"]
            self.vec[6:9] = rot_to_rpy(state["PoseSensor"][:3, :3])
           # self.vec[9:12] = state["IMUSensorClean"][1]
           # self.add = state["IMUSensor"][0]
            #self.bias[0:3] = state["IMUSensor"][3]
          #  self.bias[3:6] = state["IMUSensor"][2]

            self.mat[:3, :3] = state["PoseSensor"][:3, :3]
            self.mat[:3, 3] = state["VelocitySensor"]
            self.mat[:3, 4] = state["PoseSensor"][:3, 3]

        # Estimated State
        if isinstance(state, SE3):
            self.vec[0:3] = state[1]  # Position
            self.vec[3:6] = state[0]  # Velocity
            self.vec[6:9] = rot_to_rpy(state.mat[:3, :3].copy())  # Rotation to RPY

            if last_meas_omega is None:
                raise ValueError("Need a measurement for angular velocity")
            self.vec[9:12] = last_meas_omega - state.aug[:3]

            self.bias = state.aug
            self.mat = state.mat

        # Commanded State
        if isinstance(state, np.ndarray):
            self.vec = state
            # TODO Matrix representation here too?

    def show(state):
        return state

    @property
    def data_plot(self):
        return np.append(self.vec[:9], self.bias)


def rot_to_rpy(mat):
    """
    Converts a rotation matrix to roll, pitch, yaw (in degrees).
    """
    return Rotation.from_matrix(mat).as_euler("xyz") * 180 / np.pi
