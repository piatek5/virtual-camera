"""Virtual camera renderer using PyGame and simple OBJ edge rendering."""

from controls import InputHandler
from renderer import Renderer
from camera import Camera
from load import load_scene
from hud import draw_hud
import config

import pygame
import numpy as np

from pathlib import Path
import argparse


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Virtual camera OBJ viewer')
    parser.add_argument('paths', nargs='*', type=Path,
                        default=[config.DEFAULT_OBJ],
                        help='OBJ files to load (default: %(default)s)')
    parser.add_argument('-s', '--screenshot-dir', type=Path,
                        default=config.SCREENSHOT_DIR,
                        help='Path to screenshot save directory (default: %(default)s)')
    parser.add_argument('-m', '--move-speed', type=float,
                        default=config.MOVE_SPEED,
                        help='Move speed in units per second (default: %(default)s)')
    parser.add_argument('-r', '--rot-speed', type=float,
                        default=config.ROT_SPEED,
                        help='Rotation speed in radians per second (default: %(default)s)')
    
    return parser.parse_args(argv)

def main():
    """Run the PyGame render loop for the virtual camera viewer."""
    args = parse_args()

    # Starting camera cords and fov
    starting_pos: np.ndarray = np.array([0.0, 0.0, -5.0])
    starting_rot: np.ndarray = np.array([0.0, 0.0, 0.0])
    starting_fov = 90.0
    camera = Camera(starting_pos, starting_rot, starting_fov)
    
    pygame.init()

    screen = pygame.display.set_mode((camera.WIDTH, camera.HEIGHT))
    font = pygame.font.SysFont('Arial', 18)

    pygame.display.set_caption("Virtual Camera")

    coords, polygons = load_scene(args.paths)

    renderer = Renderer(camera, screen, polygons, coords)
    clock = pygame.time.Clock()

    input_handler = InputHandler(
        camera=camera,
        screen=screen,
        screenshots=args.screenshot_dir,
        move_speed=args.move_speed,
        rot_speed=args.rot_speed
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
