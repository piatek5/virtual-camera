import logging
import random
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

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
        coords (np.ndarray): np.ndarray of shape (V, 3) containing polygon
            vertex coordinates.
        indexes (list[int]): List of indexed vertice coordinate. Indexes point
            to the coords np.ndarray.
        plane (Plane): Plane the polygon lies in.
    """
    coords: np.ndarray
    indexes: list[int]
    plane: Plane


class Node:
    """Single BSP tree node. Holds the splitting plane, all polygons lying
    in it and the front/back subtrees.
    """
    def __init__(self, plane) -> None:
        self.front: Node | None = None
        self.back: Node | None = None
        self._polygon_coords_list: list[np.ndarray] = []
        self._polygon_indexes_list: list[list[int]] = []
        self.plane: Plane = plane
        self.color: tuple[int, ...] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ) # TODO: Change/delete (temporary showcase measure)

    def add_polygon_coords(self, polygon_coords: np.ndarray):
        self._polygon_coords_list.append(polygon_coords)

    def get_polygon_coords(self):
        return self._polygon_coords_list

    def add_polygon_indexes(self, polygon_indexes: list[int]):
            self._polygon_indexes_list.append(polygon_indexes)

    def get_polygon_indexes(self):
        return self._polygon_indexes_list


class BSPTree:
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
        self._coords_build: list[np.ndarray] = []
        self.coords: np.ndarray
        self.lam = 0.25
        self.plane_check_num = 10
        # Build statistics, only used for debug logging
        self._node_count = 0
        self._split_count = 0

    @staticmethod
    def get_plane(polygon_coords: np.ndarray) -> Plane:
        """Implementation of Newell's algorithm. Calculates the plane a polygon
        lies in: unit normal vector and plane offset. The normal direction
        follows the vertex order (counter-clockwise when viewed from the front).

        Args:
            polygon_coords (np.ndarray): np.ndarray of shape (V, 3) containing
                polygon vertex coordinates.

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

        # Unit normal vector
        n = n_raw / norm
        # Distance from origin
        d = -np.dot(n, p_0)

        return Plane(n, d)

    def build(self, coords: np.ndarray, polygons: list[list[int]]) -> Node | None:
        """Runs BSP tree generation. Converts input data into Polygon objects
        and skips damaged polygons beforehand. Also stores the result in
        self.root.

        Args:
            coords (np.ndarray): np.ndarray of shape (V, 3) containing all vertex coordinates.
            polygons (list[list[int]]): List of polygons described using vertex indexes.

        Returns:
            Node | None: BSP tree root node or None if no valid polygons were given.
        """
        # Combine and filter polygon input.
        polygon_list = self._transform_input(coords, polygons)

        self._coords_build = list(coords)
        self._node_count = 0
        self._split_count = 0
        self.root = self._build_tree(polygon_list)

        self.coords = np.array(self._coords_build)
        logger.debug(
            "BSP built: %d polygons in, %d nodes, %d splits, %d -> %d vertices",
            len(polygon_list), self._node_count, self._split_count,
            len(coords), len(self.coords))

        return self.root

    def _transform_input(self, coords: np.ndarray, polygons: list[list[int]]
    ) -> list[Polygon]:
        """Combines input polygon description into a list of Polygon objects
        and filters out damaged/incorrect polygons.

        Args:
            coords (np.ndarray): np.ndarray of shape (V, 3) containing all vertex coordinates.
            polygons (list[list[int]]): List of polygons described using vertex indexes.

        Returns:
            list[Polygon]: Valid polygons with their coordinates and planes.
        """
        polygon_list = []
        skipped = 0
        for polygon_indexed in polygons:
            polygon_coords = coords[polygon_indexed]
            try:
                plane = self.get_plane(polygon_coords)
            except ValueError as e:
                # One debug line per polygon, one warning summary below.
                logger.debug("Damaged polygon %s skipped: %s", polygon_indexed, e)
                skipped += 1
                continue

            polygon_obj = Polygon(polygon_coords, polygon_indexed, plane)
            polygon_list.append(polygon_obj)

        if skipped:
            logger.warning("%d damaged polygons skipped (run with -v for details)", skipped)

        return polygon_list

    def _build_tree(self, polygon_list: list[Polygon]) -> Node | None:
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
        self._node_count += 1
        node.add_polygon_coords(root_polygon.coords)
        node.add_polygon_indexes(root_polygon.indexes)

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
                self._split_count += 1
                front_list.append(poly_front)
                back_list.append(poly_back)
            elif classification == "FRONT":
                front_list.append(poly_front)
            elif classification == "BACK":
                back_list.append(poly_back)
            elif classification == "COPLANAR":
                node.add_polygon_coords(polygon.coords)
                node.add_polygon_indexes(polygon.indexes)

        node.front = self._build_tree(front_list)
        node.back = self._build_tree(back_list)
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
        # the best fit for representing a current root.
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
        ref_plane = ref_polygon.plane
        for curr_polygon in polygon_list:
            if curr_polygon is ref_polygon:
                continue

            dist = curr_polygon.coords @ ref_plane.n + ref_plane.d
            d_min, d_max = dist.min(), dist.max()

            label = self._classify_by_extent(d_min, d_max)
            if label == "SPANNING":
                splits += 1
            elif label == "FRONT":
                front += 1
            elif label == "BACK":
                back += 1

        return front, back, splits

    def _split_polygon(self, polygon: Polygon, plane: Plane
    ) -> tuple[str, Polygon | None, Polygon | None]:
        """Determines whether a polygon is in front of, behind or coplanar with
        the given plane and splits it into two if it crosses the plane. Vertices
        closer than EPSILON to the plane are treated as lying on it. A polygon
        that crosses the plane by less than SLIVER_EPS on one side is not split,
        but sent whole to the other side (or, if both sides are below
        SLIVER_EPS, to the side it reaches further into). Ignores degenerate polygons.

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
        # Apply normal vector and add distance from origin d to every vertex.
        distances = polygon.coords @ n + d

        d_min, d_max = distances.min(), distances.max()

        label = self._classify_by_extent(d_min, d_max)
        if label == "COPLANAR":
            return "COPLANAR", polygon, polygon
        if label == "FRONT":
            return "FRONT", polygon, None
        if label == "BACK":
            return "BACK", None, polygon

        front_vert_coords = []
        back_vert_coords = []

        front_vert_idxs = []
        back_vert_idxs =  []

        for i in range(num_verts):
            v_a = polygon.coords[i]
            v_b = polygon.coords[(i + 1) % num_verts]
            d_a = distances[i]
            d_b = distances[(i + 1) % num_verts]

            vert_idx = polygon.indexes[i]

            # Check whether current vertex is in front, behind or is
            # coplanar. Compare the distance to a minimum tolerance. Any point
            # closer than EPSILON is considered to be coplanar.
            if d_a > EPSILON:
                front_vert_coords.append(v_a)
                front_vert_idxs.append(vert_idx)
            elif d_a < -EPSILON:
                back_vert_coords.append(v_a)
                back_vert_idxs.append(vert_idx)
            else:
                # Point lies on the plane.
                front_vert_coords.append(v_a)
                back_vert_coords.append(v_a)

                front_vert_idxs.append(vert_idx)
                back_vert_idxs.append(vert_idx)

            # Check whether points lie on opposite sides of plane. True if both
            # distances from plane have different signs. If they are smaller
            # than EPSILON then they do not lie on any side (coplanar case
            # handled by _classify_by_extent).
            if (d_a > EPSILON and d_b < -EPSILON) or (
                d_a < -EPSILON and d_b > EPSILON
            ):
                t = d_a / (d_a - d_b)
                intersection = (1.0 - t) * v_a + t * v_b
                front_vert_coords.append(intersection)
                back_vert_coords.append(intersection)

                self._coords_build.append(intersection)
                last_coord_idx = len(self._coords_build) - 1

                front_vert_idxs.append(last_coord_idx)
                back_vert_idxs.append(last_coord_idx)

        if len(front_vert_coords) >= 3 and len(back_vert_coords) >= 3:
            return ("SPANNING",
                # Split fragments keep their parent's plane, as cutting a flat
                # polygon does not change its plane.
                Polygon(np.array(front_vert_coords), front_vert_idxs, polygon.plane),
                Polygon(np.array(back_vert_coords), back_vert_idxs, polygon.plane))
        else:
            return "NONE", None, None

    @staticmethod
    def _classify_by_extent(d_min: float, d_max: float) -> str:
        """Decides from the extreme vertex distances alone whether a polygon
        has to be split by a plane. Vertices closer than EPSILON are treated
        as lying on the plane, so a polygon is never split unless it has
        vertices beyond EPSILON on both sides. Even then it is split only if
        it reaches at least SLIVER_EPS past the plane on both sides. A smaller
        crossing is considered modelling noise: splitting would only produce
        an almost invisible sliver, so the whole polygon goes to one side.

        Args:
            d_min (float): Smallest signed vertex distance to the plane.
            d_max (float): Largest signed vertex distance to the plane.

        Returns:
            str: "COPLANAR", "FRONT", "BACK" or "SPANNING" (a real split is needed).
        """
        # Checking the positions of extremes (and by extension all other negative
        # or positive distances) in relation to EPSILON. If all are within EPS
        # a polygon is considered coplanar. If not all are and if no distance
        # makes it past negative EPS, then a polygon is in front. The opposite
        # goes for positive distances.
        # This fixes splitting off anything that from one or both sides
        # is confined within EPS.
        if d_max <= EPSILON and d_min >= -EPSILON:
            return "COPLANAR"
        if d_min >= -EPSILON:
            return "FRONT"
        if d_max <= EPSILON:
            return "BACK"

        # If no condition is met then we have a polygon that exceeds EPS
        # in both direction (can be considered ready for splitting). However
        # we check whether any eligible vertex is above |SLIVER_EPS|. It is
        # used to discard a side that could be split off but still is a short
        # distance from a plane. We simply lose the "lesser" side if it is
        # small enough not to create a very thin split off polygon.
        front_tiny = d_max < SLIVER_EPS
        back_tiny = d_min > -SLIVER_EPS

        if front_tiny and back_tiny:
            # Both sides are slivers but are not coplanar.
            # We pick the thicker one.
            if d_max >= -d_min:
                return "FRONT"
            return "BACK"
        if front_tiny:
            # Only a sliver in front.
            return "BACK"
        if back_tiny:
            # Only a sliver behind.
            return "FRONT"

        return "SPANNING"
