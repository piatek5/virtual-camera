from pathlib import Path

import numpy as np


def load_obj(filename: Path) -> tuple[list[np.ndarray], list[list[int]]]:
    """Load vertex and edge data from a Wavefront OBJ file.

    Args:
        filename (Path): Path to OBJ file.

    Returns:
        tuple(list[np.ndarray], list[list[int]]]):
            - a list of all vertice coordinates (x, y, z) each being a NumPy array
            - a list of all polygons each described as a list of integer indexes 
            pointing to a position in the vertice list
    """
    coords: list[np.ndarray] = []
    polygons: list[list[int]] = []
    with open(filename) as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            elif parts[0] == 'v':
                xyz = np.array((float(parts[1]), float(parts[2]), float(parts[3])))
                coords.append(xyz)
            elif parts[0] == 'f':
                # Skipping anything after a "/" in indice description
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]

                polygons.append(face_indices)

    return coords, polygons
