from typing import Self, overload

import numpy as np
from numpy.typing import ArrayLike

from virtual_camera.render.transform import get_model_matrix


class Vec3:
    def __init__(self, nonzero=False):
        self.nonzero = nonzero

    def __set_name__(self, owner, name):
        self.name = name
        self.storage = "_" + name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> Self: ...
    @overload
    def __get__(self, obj: object, objtype: type | None = None) -> np.ndarray: ...
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.storage)

    def __set__(self, obj: object, value: ArrayLike) -> None:
        arr = np.array(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"{self.name} must have 3 numbers, got {value!r}")
        if self.nonzero and np.any(arr == 0):
            raise ValueError(f"{self.name} cannot contain any 0, got {value!r}")
        setattr(obj, self.storage, arr)


class Model:
    pos: Vec3 = Vec3()
    rot: Vec3 = Vec3()
    scale: Vec3 = Vec3(nonzero=True)

    def __init__(self, coords: np.ndarray, polygons: list[list[int]]) -> None:
        self.coords = coords
        self.polygons = polygons
        self.pos = (0, 0, 0)
        self.rot = (0, 0, 0)
        self.scale = (1, 1, 1)

    def get_world_coords(self):
        world_matrix = get_model_matrix(self.pos, self.rot, self.scale)
        world_coords = self.coords @ world_matrix.T[:3, :3] + world_matrix.T[3, :3]
        return world_coords
