from pathlib import Path

import numpy as np


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

DEFAULT_OBJ = Path("example.toml")

SCREENSHOT_DIR = Path("screenshots")
