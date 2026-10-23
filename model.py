from transform import get_world_matrix

import numpy as np


class Model:
    def __init__(self, coords: np.ndarray, polygons: list[int]) -> None:
        self.coords = coords
        self.polygons = polygons
        self.pos = np.array([0, 0, 0], dtype=float)
        self.rot = np.array([0, 0, 0], dtype=float)
        self.scale = np.array([1, 1, 1], dtype=float)

    def get_world_coords(self):
        world_matrix = get_world_matrix(self.pos, self.rot, self.scale)
        world_coords = self.coords @ world_matrix.T[:3, :3] + world_matrix.T[3, :3]
        return world_coords
