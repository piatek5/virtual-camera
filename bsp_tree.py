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
        self.polygons_coords = [] # All the coplanar polygons
        self.plane: Plane = plane
        self.color: tuple[int, ...] = (randint(0, 255), randint(0, 255), randint(0, 255)) # TODO: Temporary showcase measure

    def add_polygon_coords(self, polygon_coords: list[np.ndarray]):
        self.polygons_coords.append(polygon_coords)

class BSP_tree:
    def __init__(self) -> None:
        self.root: Node | None = None

    @staticmethod
    def get_plane(polygon_coords: list[np.ndarray]) -> Plane:
        """Implementation of Newell's algorithm. Calculates a plane description: 
        unit normal vector and distance from origin.

        Args:
            polygon_coords (list[np.ndarray]): List of vertice coordinates (x, y, z).

        Raises:
            ValueError: _description_

        Returns:
            tuple[np.ndarray, float]: Unit normal vector and distance from origin.
        """
        vert_num = len(polygon_coords)
        if vert_num <= 2:
            raise ValueError("Number of polygon vertices is too small!)")
        
        nx, ny, nz = 0, 0, 0
        for i in range(vert_num):
            j = (i + 1) % vert_num
            coord_i = polygon_coords[i]
            coord_j = polygon_coords[j]
            x_i, y_i, z_i = coord_i[0], coord_i[1], coord_i[2]
            x_j, y_j, z_j = coord_j[0], coord_j[1], coord_j[2]
            nx += (y_i - y_j) * (z_i + z_j)
            ny += (z_i - z_j) * (x_i + x_j)
            nz += (x_i - x_j) * (y_i + y_j)

        p_0 = np.sum(polygon_coords, axis=0) / vert_num # Centroid
        n_raw = np.array((nx, ny, nz))
        norm = np.linalg.norm(n_raw)
        if norm < EPSILON:
            raise ValueError("Polygon vertices are coplanar!")
        n = n_raw / norm # Unit normal vector
        d = -np.dot(n, p_0) # Distance from origin
        return Plane(n, d)

    def build(self, coords: list[np.ndarray], polygons: list[list[int]]) -> Node | None:
        """Runs BSP tree generation. Converts input data beforehand.

        Args:
            coords (list[np.ndarray]): List of a all vertice coordinates.
            polygons (list[list[int]]): List of polygons described using vertice indexes.

        Returns:
            Node | None: BSP tree root node.
        """
        if not polygons:
            return None

        # Combine polygon descriptions with vertice coordinates
        polygons_coords = [
            [coords[index] for index in polygon]
            for polygon in polygons
        ]
        self.root = self._build_from_coords(polygons_coords)
        return self.root

    def _build_from_coords(self, polygons_coords: list[list[np.ndarray]]) -> Node | None:
        if not polygons_coords:
            return None

        plane = None
        root_idx = 0
        # Pick another polygon if get_plane fails
        for idx, polygon_coords in enumerate(polygons_coords):
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
        node.add_polygon_coords(polygons_coords[root_idx])

        front_list = []
        back_list = []

        # Go through every polygon to determine it being in front or at the 
        # back of current node polygon/plane. May need to split it and classify
        # it accordingly.
        for polygon_coords in polygons_coords[root_idx+1:]:
            classification, poly_front, poly_back = self._split_polygon(plane, polygon_coords)

            if classification == "SPANNING":
                front_list.append(poly_front)
                back_list.append(poly_back)
            elif classification == "FRONT":
                front_list.append(poly_front)
            elif classification == "BACK":
                back_list.append(poly_back)
            elif classification == "COPLANAR":
                node.polygons_coords.append(np.array(polygon_coords))

        node.front = self._build_from_coords(front_list)
        node.back = self._build_from_coords(back_list)
        return node

    def _split_polygon(self, plane: Plane, polygon_coords: list[np.ndarray]
    ) -> tuple[str, list[np.ndarray] | None, list[np.ndarray] | None]:
        """Determines whether a polygon is in front or at the back of the given
        plane and splits it into two if an intersection occurs. Does not return
        two-vertice shapes.

        Args:
            plane (tuple[np.ndarray, float]): Plane described as a unit normal vector and distance from origin.
            polygon_coords (list[np.ndarray]): _description_

        Returns:
            tuple (string, list[np.ndarray] or None, list[np.ndarray] or None):
                - Split result
                - Vertice coordinates in front
                - Vertice coordinates in back
        """
        n, d = plane.n, plane.d
        num_verts = len(polygon_coords)

        # Calculate distances from plane in advance
        distances = []
        for vert in polygon_coords:
            distances.append(np.dot(n, vert) + d)

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

        if len(front_verts) >= 3 and len(back_verts) >= 3:
            return "SPANNING", front_verts, back_verts
        elif len(front_verts) >= 3:
            return "FRONT", polygon_coords, None
        elif len(back_verts) >= 3:
            return "BACK", None, polygon_coords
        elif coplanar_count >= 3 and coplanar_count == len(polygon_coords):
            return "COPLANAR", polygon_coords, polygon_coords
        else:
            return "NONE", None, None
