import numpy as np


class Camera:
    """Represents a 3D camera with position and Euler rotation."""

    def __init__(self, pos: np.ndarray, rot: np.ndarray, fov=90.0):
        """Initialize the camera position and rotation."""

        self.start_pos = np.array(pos, dtype=float)
        self.start_rot = np.array(rot, dtype=float)
        self.start_fov = fov
        self.reset()

    def reset(self):
        self.pos = self.start_pos.copy()
        self.rot = self.start_rot.copy()
        self.fov = self.start_fov
        