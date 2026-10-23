import config

from datetime import datetime
from pathlib import Path

import pygame
from pygame import Surface

def save_screenshot(screen: Surface, folder: Path = config.SCREENSHOT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = folder / f"screenshot_{timestamp}.png"
    pygame.image.save(screen, path)
    
    return path
