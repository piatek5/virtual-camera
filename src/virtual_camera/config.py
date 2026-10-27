from importlib.resources import files
from pathlib import Path

import numpy as np


ASSETS_DIR = files("virtual_camera") / "assets"
DEFAULT_SCENE = ASSETS_DIR / "scenes" / "example.toml"
SCREENSHOT_DIR = Path("screenshots")

WIDTH = 1200
HEIGHT = 800
FPS = 60

STARTING_POS = np.array([0.0, 0.0, -5.0])
STARTING_ROT = np.array([0.0, 0.0, 0.0])
STARTING_FOV = 90.0

MOVE_SPEED = 6.0    # units per second
ROT_SPEED = 3.0     # radians per second
ZOOM_SPEED = 60.0   # FOV degrees per second
FOV_MIN = 10.0
FOV_MAX = 170.0
