import math
import sys

import pygame
import numpy as np

WIDTH = 800
HEIGHT = 600
FPS = 60

vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
])

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), # Back
    (4, 5), (5, 6), (6, 7), (7, 4), # Front
    (0, 4), (1, 5), (2, 6), (3, 7)  # Inbetween
]

class Camera:
    def __init__(self, pos: list[float], rot: list[float]):
        self.pos = np.array([*pos, 1.0], dtype=float)
        self.rot = np.array(rot, dtype=float)

    def get_rotation_matrix(self) -> np.ndarray:
        rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(self.rot[0]), -math.sin(self.rot[0]), 0],
            [0, math.sin(self.rot[0]),  math.cos(self.rot[0]), 0],
            [0, 0, 0, 1]
        ])

        ry = np.array([
            [ math.cos(self.rot[1]), 0, math.sin(self.rot[1]), 0],
            [0, 1, 0, 0],
            [-math.sin(self.rot[1]), 0, math.cos(self.rot[1]), 0],
            [0, 0, 0, 1]
        ])

        rz = np.array([
            [math.cos(self.rot[2]), -math.sin(self.rot[2]), 0, 0],
            [math.sin(self.rot[2]),  math.cos(self.rot[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return rz @ ry @ rx
    
    def get_translation_matrix(self) -> np.ndarray:
        t = np.array([
            [1, 0, 0, self.pos[0]],
            [0, 1, 0, self.pos[1]],
            [0, 0, 1, self.pos[2]],
            [0, 0, 0, 1]
        ])

        return t
    
    def get_model_matrix(self) -> np.ndarray:
        r = self.get_rotation_matrix()
        t = self.get_translation_matrix()

        return t @ r
    
    # TODO add fov param
    def to_projection_matrix(self, v: np.ndarray, d: float = 1):
        v[0] = (v[0] * d) / v[2]
        v[1] = (v[1] * d) / v[2]
        v[2] = d
        v[3] = 1.0

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Virtual Camera")
    clock = pygame.time.Clock()

    camera = Camera([0, 0, -5], [0, 0, 0])

    running = True
    while running:
        clock.tick(FPS)
        screen.fill((20, 20, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        speed = 0.1
        rot_speed = 0.05

        # Rotation
        if keys[pygame.K_UP]:    camera.rot[0] += rot_speed # Look up
        if keys[pygame.K_DOWN]:  camera.rot[0] -= rot_speed # Look down
        if keys[pygame.K_LEFT]:  camera.rot[1] -= rot_speed # Look left
        if keys[pygame.K_RIGHT]: camera.rot[1] += rot_speed # Look right
        if keys[pygame.K_z]:     camera.rot[2] -= rot_speed # Rotate left
        if keys[pygame.K_c]:     camera.rot[2] += rot_speed # Rotate right

        # Translation
        move_dir = np.zeros(4)
        move_dir[3] = 1.0
        if keys[pygame.K_w]: move_dir[2] += speed # Forward
        if keys[pygame.K_s]: move_dir[2] -= speed # Bakcwards
        if keys[pygame.K_a]: move_dir[0] -= speed # Left
        if keys[pygame.K_d]: move_dir[0] += speed # Right
        if keys[pygame.K_q]: move_dir[1] -= speed # Up
        if keys[pygame.K_e]: move_dir[1] += speed # Down
        
        if np.any(move_dir[:3] != 0.0):
            rotation_matrix = camera.get_model_matrix()
            move_dir = rotation_matrix @ move_dir
            print(move_dir)
            camera.pos = move_dir

        model_matrix = camera.get_model_matrix()
        view_matrix = np.linalg.inv(model_matrix)
        projected_points: list[tuple[int, int] | None] = []

        for v in vertices:
            v_4d = np.array([v[0], v[1], v[2], 1.0])
            v_view = view_matrix @ v_4d 

            if v_view[2] > 0.01:
                camera.to_projection_matrix(v_view)
                
                scale = WIDTH / 2
                screen_x = int(v_view[0] * scale + WIDTH / 2)
                screen_y = int(v_view[1] * scale + HEIGHT / 2)
                
                projected_points.append((screen_x, screen_y))
            else:
                projected_points.append(None)

        for edge in edges:
            p1 = projected_points[edge[0]]
            p2 = projected_points[edge[1]]
            
            if p1 is not None and p2 is not None:
                    pygame.draw.line(screen, (0, 255, 0), p1, p2, 2)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

main()