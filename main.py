"""Virtual camera renderer using PyGame and simple OBJ edge rendering."""

from renderer import Renderer
from camera import Camera
from load import load_obj

import pygame
import numpy as np

from datetime import datetime
from pathlib import Path
import sys

# Solid location seletion
FOLDER = 'solids'
FILE = 'cube.obj'


def main():
    """Run the PyGame render loop for the virtual camera viewer."""
    # Starting camera cords
    starting_pos: np.ndarray = np.array([0, 0, -5])
    starting_rot: np.ndarray = np.array([0, 0, 0])
    starting_fov = 90.0
    camera = Camera(starting_pos, starting_rot)
    
    pygame.init()
    screen = pygame.display.set_mode((camera.WIDTH, camera.HEIGHT))
    pygame.display.set_caption("Virtual Camera")
    clock = pygame.time.Clock()

    folder = Path(FOLDER)
    file_path = folder / FILE
    coords, polygons = load_obj(file_path)

    renderer = Renderer(camera, screen, polygons, coords)

    font = pygame.font.SysFont('Arial', 18)

    Path("screenshots").mkdir(exist_ok=True)

    running = True
    while running:
        clock.tick(camera.FPS)
        screen.fill((20, 20, 20))

        for event in pygame.event.get():
            # Clsoe the programm
            if event.type == pygame.QUIT:
                running = False
            # Zoom and unzoom wityh mouse scroll
            elif event.type == pygame.MOUSEWHEEL:
                camera.fov -= event.y * 5
                camera.fov = max(10.0, min(170.0, camera.fov))
            elif event.type == pygame.KEYDOWN:
                # Take a screenshot with F12
                if event.key == pygame.K_F12:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"screenshots/screenshot_{timestamp}.png"
                    pygame.image.save(screen, filename)
                    print(f"Saved: {filename}")
                # Reset position, rotaion and FOV with R
                if event.key == pygame.K_r:
                    camera.pos = starting_pos
                    camera.rot = starting_rot
                    camera.fov = starting_fov
                # Exit with ESC
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        speed = 0.1
        rot_speed = 0.05

        # Scale
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]: camera.fov -= 1.0 # Zoom
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: camera.fov += 1.0 # Unzoom
        camera.fov = max(10.0, min(170.0, camera.fov))

        # Rotation
        if keys[pygame.K_UP]: camera.rot[0] -= rot_speed # Look up
        if keys[pygame.K_DOWN]: camera.rot[0] += rot_speed # Look down
        if keys[pygame.K_LEFT]: camera.rot[1] -= rot_speed # Look left
        if keys[pygame.K_RIGHT]: camera.rot[1] += rot_speed # Look right
        if keys[pygame.K_z]: camera.rot[2] -= rot_speed # Rotate left
        if keys[pygame.K_c]: camera.rot[2] += rot_speed # Rotate right
        # Translation
        move_delta = np.zeros(4) # Tracking only the delta movement vector
        move_delta[3] = 1.0
        if keys[pygame.K_w]: move_delta[2] += speed # Forward
        if keys[pygame.K_s]: move_delta[2] -= speed # Backwards
        if keys[pygame.K_d]: move_delta[0] += speed # Right
        if keys[pygame.K_a]: move_delta[0] -= speed # Left
        if keys[pygame.K_q]: move_delta[1] += speed # Up
        if keys[pygame.K_e]: move_delta[1] -= speed # Down
        
        # Rotate and translate the small movement delta vector
        if np.any(move_delta[:3] != 0.0):
            rotation_matrix = camera.get_rotation_matrix()
            move_delta = rotation_matrix @ move_delta
            camera.pos = camera.pos + move_delta[:3]

        # Render vertices and edges
        renderer.render()

        rot_deg = (np.degrees(camera.rot) % 360)
        # Positional and FOV information
        pos_text = font.render(f"XYZ: {camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f}", True, (255, 255, 255))
        rot_text = font.render(f"ROT: {rot_deg[0]:.0f}°, {rot_deg[1]:.0f}°, {rot_deg[2]:.0f}°", True, (255, 255, 255))
        fov_text = font.render(f"FOV: {camera.fov:.0f}°", True, (255, 255, 255))

        screen.blit(pos_text, (10, 10))
        screen.blit(rot_text, (10, 30))
        screen.blit(fov_text, (10, 50))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

main()
