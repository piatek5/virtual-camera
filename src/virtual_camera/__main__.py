"""Virtual camera renderer using PyGame and simple OBJ edge rendering."""

from virtual_camera.app.controls import InputHandler
from virtual_camera.render.renderer import Renderer
from virtual_camera.render.camera import Camera
from virtual_camera.scene.load import load_models, merge_models
from virtual_camera.app.hud import draw_hud
import virtual_camera.config as config

import pygame

from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Virtual camera')
    parser.add_argument('paths', nargs='*', type=Path,
                        default=[config.DEFAULT_OBJ],
                        help='OBJ or TOML files to load (default: %(default)s)')
    parser.add_argument('-s', '--screenshot-dir', type=Path,
                        default=config.SCREENSHOT_DIR,
                        help='Path to screenshot save directory (default: %(default)s)')
    parser.add_argument('-m', '--move-speed', type=float,
                        default=config.MOVE_SPEED,
                        help='Move speed in units per second (default: %(default)s)')
    parser.add_argument('-r', '--rot-speed', type=float,
                        default=config.ROT_SPEED,
                        help='Rotation speed in radians per second (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show debug output')

    return parser.parse_args(argv)

def main():
    """Run the PyGame render loop for the virtual camera viewer."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    camera = Camera(config.STARTING_POS, config.STARTING_ROT, config.STARTING_FOV)
    
    pygame.init()

    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    font = pygame.font.SysFont('Arial', 18)

    pygame.display.set_caption("Virtual Camera")

    models = load_models(args.paths)
    coords, polygons = merge_models(models)
    logger.info("Loaded %d models: %d vertices, %d polygons",
                len(models), len(coords), len(polygons))

    renderer = Renderer(camera, screen, polygons, coords)
    clock = pygame.time.Clock()

    input_handler = InputHandler(
        camera=camera,
        screen=screen,
        screenshots=args.screenshot_dir,
        move_speed=args.move_speed,
        rot_speed=args.rot_speed
    )

    running = True
    while running:
        dt = clock.tick(config.FPS) / 1000.0
        screen.fill((20, 20, 20))

        # Handle pressed keys
        for event in pygame.event.get():
            if not input_handler.handle_event(event):
                running = False

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        input_handler.apply_held_keys(keys, dt)
        
        renderer.render()

        draw_hud(screen, font, camera)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
