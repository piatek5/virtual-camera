import numpy as np
from pygame import Surface, Font

from virtual_camera.render.camera import Camera


def draw_hud(screen: Surface, font: Font, camera: Camera):
    rot_deg = (np.degrees(camera.rot) % 360)
    text_color = (255, 255, 255)

    # Positional and FOV information
    pos_text = font.render(f"XYZ: {camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f}", True, text_color)
    rot_text = font.render(f"ROT: {rot_deg[0]:.0f}°, {rot_deg[1]:.0f}°, {rot_deg[2]:.0f}°", True, text_color)
    fov_text = font.render(f"FOV: {camera.fov:.0f}°", True, text_color)

    screen.blit(pos_text, (10, 10))
    screen.blit(rot_text, (10, 30))
    screen.blit(fov_text, (10, 50))
