import numpy as np

import math


def get_scale_matrix(scale: np.ndarray) -> np.ndarray:

    s = np.array([
        [scale[0], 0, 0, 0],
        [0, scale[1], 0, 0],
        [0, 0, scale[2], 0],
        [0, 0, 0, 1]
    ])

    return s

def get_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    """Create the camera rotation matrix from Euler angles.

    Returns:
        A 4x4 numpy array that rotates points by the camera rotation.
    """
    rx = np.array([
        [1, 0, 0, 0],
        [0, math.cos(rot[0]), -math.sin(rot[0]), 0],
        [0, math.sin(rot[0]),  math.cos(rot[0]), 0],
        [0, 0, 0, 1]
    ])

    ry = np.array([
        [ math.cos(rot[1]), 0, math.sin(rot[1]), 0],
        [0, 1, 0, 0],
        [-math.sin(rot[1]), 0, math.cos(rot[1]), 0],
        [0, 0, 0, 1]
    ])

    rz = np.array([
        [math.cos(rot[2]), -math.sin(rot[2]), 0, 0],
        [math.sin(rot[2]),  math.cos(rot[2]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return rz @ ry @ rx
    
def get_translation_matrix(pos: np.ndarray) -> np.ndarray:
    """Create the translation matrix for the camera position.

    Returns:
        A 4x4 numpy array that translates points by the camera position.
    """
    t = np.array([
        [1, 0, 0, pos[0]],
        [0, 1, 0, pos[1]],
        [0, 0, 1, pos[2]],
        [0, 0, 0, 1]
    ])

    return t

def get_model_matrix(pos: np.ndarray, rot: np.ndarray, scale=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Compute the matrix placing an object (model or camera) in the world.

    Returns:
        A 4x4 numpy array that scales, then rotates, then translates.
    """
    r = get_rotation_matrix(rot)
    t = get_translation_matrix(pos)
    s = get_scale_matrix(scale)

    return t @ r @ s

def get_projection_matrix(fov: float, aspect_ratio: float) -> np.ndarray:
    """Create a perspective projection matrix.

    Args:
        fov: Field of view in degrees.

    Returns:
        A 4x4 numpy array that projects 3D points into clip space.
    """
    fov_rad = math.radians(fov)
    aspect_ratio_inv = 1.0 / aspect_ratio
    fx = aspect_ratio_inv / math.tan(fov_rad / 2.0)
    fy = 1.0 / math.tan(fov_rad / 2.0)

    proj = np.array([
        [fx, 0, 0, 0],
        [0, fy, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ])
    
    return proj
