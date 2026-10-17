import numpy as np

from dataclasses import dataclass
import random

EPSILON = 1e-5
SLIVER_EPS = 1e-4


@dataclass
class Plane:
    """Plane described by the equation n · p + d = 0.

    Attributes:
        n (np.ndarray): Unit normal vector of shape (3,). Points to the front side.
        d (float): Plane offset. For any point p, n · p + d is its signed
            distance from the plane (positive in front, negative behind).
    """
    n: np.ndarray
    d: float

@dataclass
class Polygon:
    """Polygon together with the plane it lies in.

    Attributes:
        coords (np.ndarray): np.ndarray of shape (V, 3) containing polygon vertice coordinates.
        plane (Plane): Plane the polygon lies in. Split fragments keep their
            parent's plane, as cutting a flat polygon does not change its plane.
    """
    coords: np.ndarray
    plane: Plane


class Node:
    """Single BSP tree node. Holds the splitting plane, all polygons lying
    in it and the front/back subtrees.
    """
    def __init__(self, plane) -> None:
        self.front: Node | None = None
        self.back: Node | None = None
        self._polygon_coords_list: list[np.ndarray] = [] # All coplanar polygons
        self.plane: Plane = plane
        self.color: tuple[int, ...] = (
            random.randint(0, 255), 
            random.randint(0, 255), 
            random.randint(0, 255)
        ) # TODO: Temporary showcase measure

    def add_polygon_coords(self, polygon_coords: np.ndarray):
        self._polygon_coords_list.append(polygon_coords)

    def get_polygon_coords(self):
        return self._polygon_coords_list
    

class BSP_tree:
    """Binary space partitioning tree built from polygons.

    Attributes:
        root (Node | None): Root node, set by build().
        lam (float): Weight of the front/back imbalance in the splitting
            plane cost (cost = splits + lam * |front - back|).
        plane_check_num (int): Number of random candidate polygons evaluated
            when picking a splitting plane for a node.
    """
    def __init__(self) -> None:
        self.root: Node | None = None
        self.lam = 0.25
        self.plane_check_num = 10

    @staticmethod
    def get_plane(polygon_coords: np.ndarray) -> Plane:
        """Implementation of Newell's algorithm. Calculates the plane a polygon
        lies in: unit normal vector and plane offset. The normal direction
        follows the vertice order (counter-clockwise when viewed from the front).

        Args:
            polygon_coords (np.ndarray): np.ndarray of shape (V, 3) containing polygon vertice coordinates.

        Raises:
            ValueError: If number of polygon vertices is <= 2.
            ValueError: If the polygon is degenerate, i.e. the length of the
                Newell normal (twice the polygon area) is below EPSILON,
                e.g. colinear vertices.

        Returns:
            Plane: Plane object (unit normal vector and plane offset).
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
        """Runs BSP tree generation. Converts input data into Polygon objects
        and skips damaged polygons beforehand. Also stores the result in
        self.root.

        Args:
            coords (np.ndarray): np.ndarray of shape (V, 3) containing all vertice coordinates.
            polygons (list[list[int]]): List of polygons described using vertice indexes.

        Returns:
            Node | None: BSP tree root node or None if no valid polygons were given.
        """
        # Combine and filter polygon input.
        polygon_coords_list = self._transform_input(coords, polygons)

        self.root = self._build_from_coords(polygon_coords_list)

        return self.root

    def _transform_input(self, coords: np.ndarray, polygons: list[list[int]]
    ) -> list[Polygon]:
        """Combines input polygon description into a list of Polygon objects
        and filters out damaged/incorrect polygons (those get_plane rejects).

        Args:
            coords (np.ndarray): np.ndarray of shape (V, 3) containing all vertice coordinates.
            polygons (list[list[int]]): List of polygons described using vertice indexes.

        Returns:
            list[Polygon]: Valid polygons with their coordinates and planes.
        """
        polygon_coords_list = []
        for polygon_indexed in polygons:
            polygon_coords = coords[polygon_indexed]
            try:
                plane = self.get_plane(polygon_coords)
            except ValueError as e:
                err = ValueError(
                    f"Error building a BSP node. Damaged polygon: {polygon_coords}. Skipping."
                )
                err.__cause__ = e
                continue

            polygon_obj = Polygon(polygon_coords, plane)
            polygon_coords_list.append(polygon_obj)

        return polygon_coords_list

    def _build_from_coords(self, polygon_list: list[Polygon]) -> Node | None:
        """Recursively builds a BSP subtree. Picks a splitting polygon, stores
        it with all polygons coplanar to it in a new node and distributes the
        rest (splitting them when needed) into front and back subtrees.

        Args:
            polygon_list (list[Polygon]): Polygons to put into this subtree.

        Returns:
            Node | None: Subtree root node or None if polygon_list is empty.
        """
        if not polygon_list:
            return None

        root_polygon = self._pick_best_polygon(polygon_list)

        if root_polygon is None:
            return None
            
        node = Node(root_polygon.plane)
        node.add_polygon_coords(root_polygon.coords)

        front_list = []
        back_list = []

        # Go through every polygon to determine it being in front or at the 
        # back of current node polygon/plane. May need to split it and classify
        # it accordingly.
        for polygon in polygon_list:
            if polygon is root_polygon:
                continue

            classification, poly_front, poly_back = self._split_polygon(
                polygon, root_polygon.plane)

            if classification == "SPANNING":
                front_list.append(poly_front)
                back_list.append(poly_back)
            elif classification == "FRONT":
                front_list.append(poly_front)
            elif classification == "BACK":
                back_list.append(poly_back)
            elif classification == "COPLANAR":
                node.add_polygon_coords(polygon.coords)

        node.front = self._build_from_coords(front_list)
        node.back = self._build_from_coords(back_list)
        return node

    def _pick_best_polygon(self, polygon_list: list[Polygon]
    ) -> Polygon | None:
        """Picks the polygon whose plane splits the given polygons best.
        Evaluates up to plane_check_num random candidates and returns the one
        with the lowest cost = splits + lam * |front - back|, preferring few
        splits and a balanced tree.

        Args:
            polygon_list (list[Polygon]): Polygons to choose from.

        Returns:
            Polygon | None: Best candidate or None if polygon_list is empty.
        """
        best_polygon, best_cost = None, np.inf

        num_to_check = min(len(polygon_list), self.plane_check_num)
        random_polygons = random.sample(polygon_list, num_to_check)

        # Check some (as set in class attributes) random polygons and pick
        # the best fit for representing a current root
        for polygon in random_polygons:
            front, back, splits = self._classify_counts(polygon, polygon_list)
            cost = splits + self.lam * abs(front - back)
            
            if cost < best_cost:
                best_cost = cost
                best_polygon = polygon

        return best_polygon

    def _classify_counts(self, ref_polygon: Polygon, polygon_list: list[Polygon]):
        """Counts how the polygons would be distributed by ref_polygon's plane,
        without actually splitting them. Each whole polygon counts once.
        Coplanar polygons are not counted.

        Args:
            ref_polygon (Polygon): Candidate polygon whose plane is evaluated.
            polygon_list (list[Polygon]): Polygons to classify (ref_polygon is skipped).

        Returns:
            tuple (int, int, int):
                - Number of polygons in front of the plane
                - Number of polygons behind the plane
                - Number of polygons the plane would split
        """
        front, back, splits = 0, 0, 0
        for curr_polygon in polygon_list:
            if curr_polygon is ref_polygon:
                continue
            ref_plane = ref_polygon.plane
            dist = curr_polygon.coords @ ref_plane.n + ref_plane.d
            f, b = np.any(dist > EPSILON), np.any(dist < -EPSILON)
            if f and b: splits += 1
            elif f: front += 1
            elif b: back += 1
        return front, back, splits

    def _split_polygon(self, polygon: Polygon, plane: Plane
    ) -> tuple[str, Polygon | None, Polygon | None]:
        """Determines whether a polygon is in front of, behind or coplanar with
        the given plane and splits it into two if it crosses the plane. Vertices
        closer than EPSILON to the plane are treated as lying on it. A polygon
        that crosses the plane by less than SLIVER_EPS on one side is not split,
        but sent whole to the other side (or, if both sides are below
        SLIVER_EPS, to the side it reaches further into). Does not return
        two-vertice shapes.

        Args:
            polygon (Polygon): Polygon to classify and possibly split.
            plane (Plane): Splitting plane (unit normal vector and plane offset).

        Returns:
            tuple (str, Polygon or None, Polygon or None):
                - Classification: "FRONT", "BACK", "SPANNING", "COPLANAR" or
                  "NONE" (degenerate polygon)
                - Front part (the original polygon if not split), None if nothing is in front
                - Back part (the original polygon if not split), None if nothing is behind
        """
        n, d = plane.n, plane.d
        num_verts = polygon.coords.shape[0]

        # Calculate distances from plane in advance.
        # Apply normal vector and add distance from origin d to every vertice.
        distances = polygon.coords @ n + d

        d_min, d_max = distances.min(), distances.max()

        # A polygon that truly crosses the plane (vertices beyond EPSILON on 
        # both Sides) is split only if it reaches at least SLIVER_EPS past the 
        # plane on both sides. A smaller crossing is modelling noise: 
        # splitting would only produce an invisible sliver, so the whole 
        # polygon goes to one side.
        if d_max > EPSILON and d_min < -EPSILON: 
            front_tiny = d_max < SLIVER_EPS
            back_tiny = d_min > -SLIVER_EPS
        
            if front_tiny and back_tiny:
                if d_max >= -d_min:
                    return "FRONT", polygon, None
                return "BACK", None, polygon
            if front_tiny:
                # Only a sliver in front
                return "BACK", None, polygon
            if back_tiny:
                # Only a sliver behind
                return "FRONT", polygon, None

        front_verts = []
        back_verts = []
        coplanar_count = 0

        for i in range(num_verts):
            v_a = polygon.coords[i]
            v_b = polygon.coords[(i + 1) % num_verts]
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

        if coplanar_count >= 3 and coplanar_count == num_verts:
            return "COPLANAR", polygon, polygon
        elif len(front_verts) >= 3 and len(back_verts) >= 3:
            return ("SPANNING",
                Polygon(np.array(front_verts), polygon.plane),
                Polygon(np.array(back_verts), polygon.plane))
        elif len(front_verts) >= 3:
            return "FRONT", polygon, None
        elif len(back_verts) >= 3:
            return "BACK", None, polygon
        else:
            return "NONE", None, None
