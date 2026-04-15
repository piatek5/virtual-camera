from datetime import datetime
import math
import sys

from pathlib import Path
import pygame
import numpy as np

FOLDER = 'solids'
FILE = 'city.obj'

WIDTH = 800
HEIGHT = 600
FPS = 60
FOV = 90


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
    
    def get_projection_matrix(self, fov: float = FOV) -> np.ndarray:
        fov_rad = math.radians(fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        # Pusta macierz 4x4
        proj = np.zeros((4, 4))
        
        # Skalowanie X i Y na podstawie FOV
        proj[0, 0] = f
        proj[1, 1] = f
        
        # Zachowujemy Z (później można tu dodać near/far planes)
        proj[2, 2] = 1.0 
        
        # MAGIA: Przepisujemy Z do czwartego wymiaru W!
        proj[3, 2] = 1.0 
        proj[3, 3] = 0.0
        
        return proj

@staticmethod
def load_obj(filename: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    vertices: list[list[float]] = []
    edges: set[tuple[int, int]] = set()
    with open(filename) as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            elif parts[0] == 'v':
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(xyz)
            elif parts[0] == 'f':
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                
                for i in range(len(face_indices)):
                    p1 = face_indices[i]
                    p2 = face_indices[(i + 1) % len(face_indices)]
                    
                    edge = (min(p1, p2), max(p1, p2))
                    edges.add(edge)

    return np.array(vertices), list(edges)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Virtual Camera")
    clock = pygame.time.Clock()

    camera = Camera([0, 0, -5], [0, 0, 0])

    folder = Path(FOLDER)
    file_path = folder / FILE
    vertices, edges = load_obj(file_path)

    font = pygame.font.SysFont('Arial', 18)

    Path("screenshots").mkdir(exist_ok=True)

    current_fov = FOV

    take_screenshot = False

    running = True
    while running:
        clock.tick(FPS)
        screen.fill((20, 20, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                current_fov -= event.y * 5 
                current_fov = max(10.0, min(170.0, current_fov))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F12:
                    take_screenshot = True

        keys = pygame.key.get_pressed()
        speed = 0.1
        rot_speed = 0.05

        # Scale
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]: current_fov -= 1.0 # Zoom
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: current_fov += 1.0 # Unzoom
        current_fov = max(10.0, min(170.0, current_fov))

        # Rotation
        if keys[pygame.K_UP]: camera.rot[0] += rot_speed # Look up
        if keys[pygame.K_DOWN]: camera.rot[0] -= rot_speed # Look down
        if keys[pygame.K_LEFT]: camera.rot[1] -= rot_speed # Look left
        if keys[pygame.K_RIGHT]: camera.rot[1] += rot_speed # Look right
        if keys[pygame.K_z]: camera.rot[2] -= rot_speed # Rotate left
        if keys[pygame.K_c]: camera.rot[2] += rot_speed # Rotate right

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
            camera.pos = move_dir

        model_matrix = camera.get_model_matrix()
        view_matrix = np.linalg.inv(model_matrix)
        proj_matrix = camera.get_projection_matrix(fov=current_fov)

        projected_points: list[tuple[int, int] | None] = []

        for v in vertices:
            v_4d = np.array([v[0], v[1], v[2], 1.0])
            v_view = view_matrix @ v_4d 

            if v_view[2] > 0.01:
                v_proj = proj_matrix @ v_view
                
                w = v_proj[3]
                if w != 0:
                    v_proj[0] /= w
                    v_proj[1] /= w
                    v_proj[2] /= w
                    v_proj[3] = 1.0

                scale = WIDTH / 2
                screen_x = int(v_proj[0] * scale + WIDTH / 2)
                screen_y = int(v_proj[1] * scale + HEIGHT / 2)
                
                projected_points.append((screen_x, screen_y))
            else:
                projected_points.append(None)

        for edge in edges:
            p1 = projected_points[edge[0]]
            p2 = projected_points[edge[1]]
            
            if p1 is not None and p2 is not None:
                pygame.draw.line(screen, (0, 255, 0), p1, p2, 2)

        rot_deg = np.degrees(camera.rot)
        pos_text = font.render(f"XYZ: {camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f}", True, (255, 255, 255))
        rot_text = font.render(f"ROT: {rot_deg[0]:.0f}°, {rot_deg[1]:.0f}°, {rot_deg[2]:.0f}°", True, (255, 255, 255))
        fov_text = font.render(f"FOV: {current_fov:.0f}°", True, (255, 255, 255))

        screen.blit(pos_text, (10, 10))
        screen.blit(rot_text, (10, 30))
        screen.blit(fov_text, (10, 50))

        if take_screenshot:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshots/screenshot_{timestamp}.png"
            pygame.image.save(screen, filename)
            print(f"Zapisano: {filename}")
            take_screenshot = False 

        pygame.display.flip()

    pygame.quit()
    sys.exit()

main()