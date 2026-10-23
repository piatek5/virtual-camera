"""Virtual camera renderer using PyGame and simple OBJ edge rendering."""

from controls import InputHandler
from renderer import Renderer
from camera import Camera
from load import load_obj
from hud import draw_hud
import config

import pygame
import numpy as np


from pathlib import Path

# Solid location seletion
FOLDER = 'solids'
FILE = 'cube.obj'


def main():
    """Run the PyGame render loop for the virtual camera viewer."""
    # Starting camera cords and fov
    starting_pos: np.ndarray = np.array([0.0, 0.0, -5.0])
    starting_rot: np.ndarray = np.array([0.0, 0.0, 0.0])
    starting_fov = 90.0
    camera = Camera(starting_pos, starting_rot, starting_fov)
    
    pygame.init()

    screen = pygame.display.set_mode((camera.WIDTH, camera.HEIGHT))
    font = pygame.font.SysFont('Arial', 18)

    pygame.display.set_caption("Virtual Camera")

    folder = Path(FOLDER)
    file_path = folder / FILE
    coords, polygons = load_obj(file_path)

    renderer = Renderer(camera, screen, polygons, coords)
    clock = pygame.time.Clock()

    move_speed = config.MOVE_SPEED
    rot_speed = config.ROT_SPEED
    screenshots = config.SCREENSHOT_DIR
    input_handler = InputHandler(
        camera=camera,
        screen=screen,
        screenshots=screenshots,
        move_speed=move_speed,
        rot_speed=rot_speed
    )

    while True:
        dt = clock.tick(camera.FPS) / 1000.0
        screen.fill((20, 20, 20))

        # Handle pressed keys
        for event in pygame.event.get():
            input_handler.handle_event(event)

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        input_handler.apply_held_keys(keys, dt)
        
        renderer.render()

        draw_hud(screen, font, camera)

        pygame.display.flip()

if __name__ == "__main__":
    main()
