import numpy as np
import math


class Camera:
    WIDTH = 1600
    HEIGHT = 1200
    FPS = 60

    """Represents a 3D camera with position and Euler rotation."""

    def __init__(self, pos: np.ndarray, rot: np.ndarray):
        """Initialize the camera position and rotation.

        Args:
            pos: A 3-element list of the camera position in world space.
            rot: A 3-element list of Euler angles (x, y, z) in radians.
        """
        self.pos = np.array(pos, dtype=float)
        self.rot = np.array(rot, dtype=float)
        self.fov = 90.0

    def get_rotation_matrix(self) -> np.ndarray:
        """Create the camera rotation matrix from Euler angles.

        Returns:
            A 4x4 numpy array that rotates points by the camera rotation.
        """
        rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(self.rot[0]), -math.sin(self.rot[0]), 0],
            [0, math.sin(self.rot[0]),  math.cos(self.rot[0]), 0],
            [0, 0, 0, 1]
        ])

        ry = np.array([
            [ math.cos(self.rot[1]), 0, math.sin(self.rot[1]), 0],
            [0, 1, 0, 0],
            [-math.sin(self.rot[1]), 0, math.cos(self.rot[1]), 0],
            [0, 0, 0, 1]
        ])

        rz = np.array([
            [math.cos(self.rot[2]), -math.sin(self.rot[2]), 0, 0],
            [math.sin(self.rot[2]),  math.cos(self.rot[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return rz @ ry @ rx
    
    def get_translation_matrix(self) -> np.ndarray:
        """Create the translation matrix for the camera position.

        Returns:
            A 4x4 numpy array that translates points by the camera position.
        """
        t = np.array([
            [1, 0, 0, self.pos[0]],
            [0, 1, 0, self.pos[1]],
            [0, 0, 1, self.pos[2]],
            [0, 0, 0, 1]
        ])

        return t
    
    def get_model_matrix(self) -> np.ndarray:
        """Compute the combined camera model matrix.

        Returns:
            A 4x4 numpy array representing translation followed by rotation.
        """
        r = self.get_rotation_matrix()
        t = self.get_translation_matrix()

        return t @ r

    def get_projection_matrix(self) -> np.ndarray:
        """Create a perspective projection matrix.

        Args:
            fov: Field of view in degrees.

        Returns:
            A 4x4 numpy array that projects 3D points into clip space.
        """
        fov_rad = math.radians(self.fov)
        aspect_ratio_x = self. HEIGHT / self.WIDTH
        fx = aspect_ratio_x / math.tan(fov_rad / 2.0)
        fy = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.array([
                    [fx, 0, 0, 0],
                    [0, fy, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0]
                ])
        
        return proj
