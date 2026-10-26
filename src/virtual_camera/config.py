from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_SCENE = ASSETS_DIR / "scenes" / "example.toml"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

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
