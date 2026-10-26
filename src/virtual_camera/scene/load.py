from virtual_camera.scene.model import Model

from pathlib import Path
import logging
import tomllib

import numpy as np

logger = logging.getLogger(__name__)


def merge_models(models: list[Model]):
    if not models:
        raise ValueError("No models were loaded, nothing to render.")

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

    return np.vstack(all_coords), all_polygons

def load_models(paths: list[Path]) -> list[Model]:
    """Load models from OBJ and TOML scene files. Files that can't be loaded
    are skipped with a warning.
    """
    models = []
    for path in paths:
        try:
            if path.suffix == '.toml':
                models += load_scene_file(path)
            elif path.suffix == '.obj':
                models.append(load_obj(path))
            else:
                raise ValueError(f"Unsupported file type '{path.suffix}'")
        # TOMLDecodeError is a ValueError
        except (OSError, ValueError) as e:
            logger.warning("%s skipped: %s", path, e)

    return models

def load_scene_file(path: Path) -> list[Model]:
    """Load all [[model]] entries of a TOML scene file. A broken entry is
    skipped with a warning.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    models = []
    for i, entry in enumerate(data.get('model', []), start=1):
        try:
            models.append(_load_scene_entry(path, entry))
        except (OSError, ValueError) as e:
            logger.warning("%s: model #%d skipped: %s", path, i, e)

    logger.debug("%s: loaded %d models", path, len(models))
    return models

def _load_scene_entry(path: Path, entry: dict) -> Model:
    if 'file' not in entry:
        raise ValueError("Missing 'file'")

    model = load_obj(path.parent / entry["file"])
    if "pos" in entry:
        model.pos = entry["pos"]
    if "rot" in entry:
        model.rot = np.radians(np.array(entry["rot"], dtype=float))
    if "scale" in entry:
        model.scale = entry["scale"]

    return model

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

    # Keep the (V, 3) shape even for a file with no vertices
    coords_matrix = np.array(coords, dtype=np.float64).reshape(-1, 3)

    if not polygons:
        logger.warning("%s has no faces, nothing of it will be drawn", filename)
    logger.debug("%s: %d vertices, %d polygons", filename, len(coords), len(polygons))

    return Model(coords_matrix, polygons)
