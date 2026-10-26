from virtual_camera.render.transform import get_rotation_matrix
from virtual_camera.render.camera import Camera
from virtual_camera.app.screenshot import save_screenshot
import virtual_camera.config as config
 
from pathlib import Path
import logging

import pygame
import numpy as np
from pygame import Event, Surface

logger = logging.getLogger(__name__)


class InputHandler:
    def __init__(self, 
        camera: Camera, 
        screen: Surface,
        screenshots: Path,
        move_speed=config.MOVE_SPEED,
        rot_speed=config.ROT_SPEED,
        zoom_speed=config.ZOOM_SPEED
    ) -> None:
        self.camera = camera
        self.screen = screen
        self.screenshots = screenshots
        self.move_speed = move_speed
        self.rot_speed = rot_speed
        self.zoom_speed = zoom_speed
        
    def handle_event(self, event: Event) -> bool:
        """Handle a single event. Returns False when the user asked to quit."""
        # Close the programm
        if event.type == pygame.QUIT:
            return False
        # Zoom and unzoom wityh mouse scroll
        elif event.type == pygame.MOUSEWHEEL:
            self.camera.fov -= event.y * 5
        elif event.type == pygame.KEYDOWN:
            # Take a screenshot with F12
            if event.key == pygame.K_F12:
                path = save_screenshot(self.screen, self.screenshots)
                logger.info("Saved screenshot: %s", path)

            # Reset position, rotaion and FOV with R
            if event.key == pygame.K_r:
                self.camera.reset()
            # Exit with ESC
            if event.key == pygame.K_ESCAPE:
                return False

        return True

    def apply_held_keys(self, keys, dt):
        # Scale
        zoom_speed = self.zoom_speed * dt
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]: self.camera.fov -= zoom_speed # Zoom
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: self.camera.fov += zoom_speed # Unzoom

        # Rotation
        rot_speed = self.rot_speed * dt
        if keys[pygame.K_UP]: self.camera.rot[0] -= rot_speed # Look up
        if keys[pygame.K_DOWN]: self.camera.rot[0] += rot_speed # Look down
        if keys[pygame.K_LEFT]: self.camera.rot[1] -= rot_speed # Look left
        if keys[pygame.K_RIGHT]: self.camera.rot[1] += rot_speed # Look right
        if keys[pygame.K_z]: self.camera.rot[2] -= rot_speed # Rotate left
        if keys[pygame.K_c]: self.camera.rot[2] += rot_speed # Rotate right

        # Translation
        move_delta = np.zeros(4) # Tracking only the delta movement vector
        move_delta[3] = 1.0
        move_speed = self.move_speed * dt
        if keys[pygame.K_w]: move_delta[2] += move_speed # Forward
        if keys[pygame.K_s]: move_delta[2] -= move_speed # Backwards
        if keys[pygame.K_d]: move_delta[0] += move_speed # Right
        if keys[pygame.K_a]: move_delta[0] -= move_speed # Left
        if keys[pygame.K_q]: move_delta[1] += move_speed # Up
        if keys[pygame.K_e]: move_delta[1] -= move_speed # Down
        
        # Rotate and translate the small movement delta vector
        if np.any(move_delta[:3] != 0.0):
            rotation_matrix = get_rotation_matrix(self.camera.rot)
            move_delta = rotation_matrix @ move_delta
            self.camera.pos = self.camera.pos + move_delta[:3]
