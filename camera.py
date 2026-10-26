from config import FOV_MIN, FOV_MAX

import numpy as np


class Camera:
    """Represents a 3D camera with position and Euler rotation."""

    def __init__(self, pos: np.ndarray, rot: np.ndarray, fov=90.0):
        """Initialize the camera position and rotation."""

        self.start_pos = np.array(pos, dtype=float)
        self.start_rot = np.array(rot, dtype=float)
        self.start_fov = fov
        self.reset()

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = max(FOV_MIN, min(FOV_MAX, value))

    def reset(self):
        self.pos = self.start_pos.copy()
        self.rot = self.start_rot.copy()
        self.fov = self.start_fov
        