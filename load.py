from pathlib import Path

import numpy as np


def load_scene(paths: list[Path]):
    """Load vertex and edge data from multiple Wavefront OBJ file.

    Args:
        paths (list[Path]): List of paths pointing to OBJ files.

    Returns:
        tuple(np.ndarray, list[list[int]]]):
            - np.ndarray of shape (V, 3) containing all vertice coordinates.
            - a list of all polygons each described as a list of integer indexes 
            pointing to a position in the vertice list
    """
    all_coords, all_polygons = [], []
    offset = 0

    for path in paths:
        coords, polygons = load_obj(path)
        all_coords.append(coords)

        for polygon in polygons:
            polygon_shifted = []

            for poly_idx in polygon:
                polygon_shifted.append(poly_idx + offset)

            all_polygons.append(polygon_shifted)

        offset += coords.shape[0]

    return np.vstack(all_coords), all_polygons

def load_obj(filename: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Load vertex and edge data from a Wavefront OBJ file.

    Args:
        filename (Path): Path to OBJ file.

    Returns:
        tuple(np.ndarray, list[list[int]]]):
            - np.ndarray of shape (V, 3) containing all vertice coordinates.
            - a list of all polygons each described as a list of integer indexes 
            pointing to a position in the vertice list
    """
    coords: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    with open(filename) as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue

            elif parts[0] == 'v':
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                coords.append(xyz)

            elif parts[0] == 'f':
                # Skip anything after a "/" in indice description
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                polygons.append(face_indices)

    coords_matrix = np.array(coords, dtype=np.float64)

    return coords_matrix, polygons
