from model import Model

from pathlib import Path
import tomllib

import numpy as np


def merge_models(models: list[Model]):
    all_coords, all_polygons = [], []
    offset = 0

    for model in models:
        all_coords.append(model.get_world_coords())

        for polygon in model.polygons:
            polygon_shifted = []

            for poly_idx in polygon:
                polygon_shifted.append(poly_idx + offset)

            all_polygons.append(polygon_shifted)

        offset += model.coords.shape[0]

    if not models:
        raise ValueError("No models were loaded, nothing to render.")

    return np.vstack(all_coords), all_polygons

def load_models(paths: list[Path]) -> list[Model]:
    models = []
    for path in paths:
        if path.suffix == '.toml':
            models += load_scene_file(path)
        elif path.suffix == '.obj':
            models.append(load_obj(path))
        else:
            print('Wrong file format!')
            continue
            #TODO: handle wrong format
    
    return models

def load_scene_file(path: Path) -> list[Model]:
    with open(path, "rb") as f: 
        data = tomllib.load(f)

    models = []
    for i, entry in enumerate(data.get('model', []), start=1):
        if 'file' not in entry:
            print(f"{path}: model #{i} has no 'file'")
            continue
            #TODO: handle file error

        model = load_obj(path.parent / entry["file"])
        model = load_obj(path.parent / entry["file"])
        if "pos" in entry:
            model.pos = entry["pos"]
        if "rot" in entry:
            model.rot = np.radians(entry["rot"])   # only file values (degrees) get converted
        if "scale" in entry:
            model.scale = entry["scale"]

        models.append(model)

    return models

def load_obj(filename: Path) -> Model:
    """Load vertex and edge data from a Wavefront OBJ file.

    Args:
        filename (Path): Path to OBJ file.

    Returns:
        Model object representing one OBJ model.
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

    return Model(coords_matrix, polygons)
