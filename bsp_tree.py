import numpy as np

from dataclasses import dataclass
from random import randint

EPSILON = 1e-5

@dataclass
class Plane:
    n: np.ndarray
    d: float

class Node:
    def __init__(self, plane) -> None:
        self.front: Node | None = None
        self.back: Node | None = None
        self._polygon_coords_list = [] # All the coplanar polygons
        self.plane: Plane = plane
        self.color: tuple[int, ...] = (randint(0, 255), randint(0, 255), randint(0, 255)) # TODO: Temporary showcase measure

    def add_polygon_coords(self, polygon_coords: np.ndarray):
        self._polygon_coords_list.append(polygon_coords)

    def get_polygon_coords(self):
        return self._polygon_coords_list

class BSP_tree:
    def __init__(self) -> None:
        self.root: Node | None = None

    @staticmethod
    def get_plane(polygon_coords: np.ndarray) -> Plane:
        """Implementation of Newell's algorithm. Calculates a plane description: 
        unit normal vector and distance from origin.

        Args:
            polygon_coords (np.ndarray): np.ndarray of shape (V, 3) containing polygon vertice coordinates.

        Raises:
            ValueError: _description_

        Returns:
            Plane: Plane object (unit normal vector and distance from origin).
        """
        vert_num = polygon_coords.shape[0]
        if vert_num <= 2:
            raise ValueError("Number of polygon vertices is too small!")

        coords_i = polygon_coords
        coords_j = np.roll(polygon_coords, shift=-1, axis=0)
        x_i, y_i, z_i = coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]
        x_j, y_j, z_j = coords_j[:, 0], coords_j[:, 1], coords_j[:, 2]

        nx = np.sum((y_i - y_j) * (z_i + z_j))
        ny = np.sum((z_i - z_j) * (x_i + x_j))
        nz = np.sum((x_i - x_j) * (y_i + y_j))

        # Centroid
        p_0 = np.mean(polygon_coords, axis=0)

        n_raw = np.array([nx, ny, nz])
        norm = np.linalg.norm(n_raw)

        if norm < EPSILON:
            raise ValueError("Polygon vertices are colinear!")
        
        n = n_raw / norm # Unit normal vector
        d = -np.dot(n, p_0) # Distance from origin
        return Plane(n, d)

    def build(self, coords: np.ndarray, polygons: list[list[int]]) -> Node | None:
        """Runs BSP tree generation. Converts input data beforehand.

        Args:
            coords (np.ndarray): np.ndarray of shape (V, 3) containing all vertice coordinates.
            polygons (list[list[int]]): List of polygons described using vertice indexes.

        Returns:
            Node | None: BSP tree root node.
        """
        if not polygons:
            return None

        # Combine polygon descriptions with vertice coordinates
        polygon_coords_list = [coords[polygon] for polygon in polygons]

        self.root = self._build_from_coords(polygon_coords_list)
        return self.root

    def _build_from_coords(self, polygon_coords_list: list[np.ndarray]) -> Node | None:
        if not polygon_coords_list:
            return None

        plane = None
        root_idx = 0
        # Pick another polygon if get_plane fails
        for idx, polygon_coords in enumerate(polygon_coords_list):
            try:
                plane = self.get_plane(polygon_coords)
            except ValueError as e:
                err = ValueError(
                    f"Error building a BSP node. Damaged polygon: {polygon_coords}"
                )
                err.__cause__ = e
                continue

            root_idx = idx
            break

        # If no plane was left
        if plane is None:
            return None
            
        node = Node(plane)
        node.add_polygon_coords(polygon_coords_list[root_idx])

        front_list = []
        back_list = []

        # Go through every polygon to determine it being in front or at the 
        # back of current node polygon/plane. May need to split it and classify
        # it accordingly.
        for polygon_coords in polygon_coords_list[root_idx+1:]:
            classification, poly_front, poly_back = self._split_polygon(plane, polygon_coords)

            if classification == "SPANNING":
                front_list.append(poly_front)
                back_list.append(poly_back)
            elif classification == "FRONT":
                front_list.append(poly_front)
            elif classification == "BACK":
                back_list.append(poly_back)
            elif classification == "COPLANAR":
                node.add_polygon_coords(polygon_coords_list[root_idx])

        node.front = self._build_from_coords(front_list)
        node.back = self._build_from_coords(back_list)
        return node

    def _split_polygon(self, plane: Plane, polygon_coords: np.ndarray
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        """Determines whether a polygon is in front of or behind the given
        plane and splits it into two if an intersection occurs. Does not return
        two-vertice shapes.

        Args:
            plane (Plane): Plane object (unit normal vector and distance from origin).
            polygon_coords (np.ndarray): np.ndarray of shape (V, 3) containing polygon vertice coordinates.

        Returns:
            tuple (string, list[np.ndarray] or None, list[np.ndarray] or None):
                - Split result
                - Vertice coordinates in front of
                - Vertice coordinates behind
        """
        n, d = plane.n, plane.d
        num_verts = len(polygon_coords)

        # Calculate distances from plane in advance.
        # Apply normal vector and add distance from origin d to every vertice.
        distances = polygon_coords @ n + d

        front_verts = []
        back_verts = []
        coplanar_count = 0

        for i in range(num_verts):
            v_a = polygon_coords[i]
            v_b = polygon_coords[(i + 1) % num_verts]
            d_a = distances[i]
            d_b = distances[(i + 1) % num_verts]

            # Check whether current vertice is in front, at the back or is 
            # coplanar. Compare the distance to a minimum tolerance. Any point 
            # closer than EPSILON is considered to be coplanar.
            if d_a > EPSILON:
                front_verts.append(v_a)
            elif d_a < -EPSILON:
                back_verts.append(v_a)
            else:
                front_verts.append(v_a)
                back_verts.append(v_a)
                coplanar_count += 1

            # Check whether points lie on opposite sites of plane. True if both 
            # distances from plane have different signs. If they are smaller
            # than EPISLON then they do not lie on any side (coplanar case).
            if (d_a > EPSILON and d_b < -EPSILON) or (
                d_a < -EPSILON and d_b > EPSILON
            ):
                t = d_a / (d_a - d_b)
                intersection = (1.0 - t) * v_a + t * v_b
                front_verts.append(intersection)
                back_verts.append(intersection)

        if coplanar_count >= 3 and coplanar_count == polygon_coords.shape[0]:
            return "COPLANAR", polygon_coords, polygon_coords
        elif len(front_verts) >= 3 and len(back_verts) >= 3:
            return "SPANNING", np.array(front_verts), np.array(back_verts)
        elif len(front_verts) >= 3:
            return "FRONT", polygon_coords, None
        elif len(back_verts) >= 3:
            return "BACK", None, polygon_coords
        else:
            return "NONE", None, None
