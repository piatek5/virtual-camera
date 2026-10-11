from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import numpy as np

@dataclass
class vertices_d:
    cords: list[np.ndarray] = field(default_factory=list)
    neighbours: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))


def load_obj(filename: Path) -> tuple[vertices_d, set[tuple[int, int]]]:
    """Load vertex and edge data from a Wavefront OBJ file.

    Args:
        filename: Path to the OBJ file.

    Returns:
        A tuple containing a numpy array of vertices and a list of unique edges.
    """
    vertices: vertices_d = vertices_d()
    edges: set[tuple[int, int]] = set()
    with open(filename) as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            elif parts[0] == 'v':
                xyz = np.array((float(parts[1]), float(parts[2]), float(parts[3])))
                vertices.cords.append(xyz)
            elif parts[0] == 'f':
                # Skipping anything after a "/" in indice description
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                
                for i in range(len(face_indices)):
                    p1 = face_indices[i]
                    p2 = face_indices[(i + 1) % len(face_indices)]
                    
                    edge = (min(p1, p2), max(p1, p2))
                    edges.add(edge)
                    vertices.neighbours[p1].append(p2)
                    vertices.neighbours[p2].append(p1)

    return vertices, edges
