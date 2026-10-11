"""Virtual camera renderer using PyGame and simple OBJ edge rendering."""

from datetime import datetime
import math
import sys

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import pygame
import numpy as np

# Solid location seletion
FOLDER = 'solids'
FILE = 'cube.obj'

WIDTH = 1600
HEIGHT = 1200
FPS = 60
FOV = 90


@dataclass
class vertices_d:
    cords: list[np.ndarray] = field(default_factory=list)
    neighbours: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))


class Camera:
    """Represents a 3D camera with position and Euler rotation."""

    def __init__(self, pos: list[float], rot: list[float]):
        """Initialize the camera position and rotation.

        Args:
            pos: A 3-element list of the camera position in world space.
            rot: A 3-element list of Euler angles (x, y, z) in radians.
        """
        self.pos = np.array(pos, dtype=float)
        self.rot = np.array(rot, dtype=float)

    def get_rotation_matrix(self) -> np.ndarray:
        """Create the camera rotation matrix from Euler angles.

        Returns:
            A 4x4 numpy array that rotates points by the camera rotation.
        """
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
        """Create the translation matrix for the camera position.

        Returns:
            A 4x4 numpy array that translates points by the camera position.
        """
        t = np.array([
            [1, 0, 0, self.pos[0]],
            [0, 1, 0, self.pos[1]],
            [0, 0, 1, self.pos[2]],
            [0, 0, 0, 1]
        ])

        return t
    
    def get_model_matrix(self) -> np.ndarray:
        """Compute the combined camera model matrix.

        Returns:
            A 4x4 numpy array representing translation followed by rotation.
        """
        r = self.get_rotation_matrix()
        t = self.get_translation_matrix()

        return t @ r

    @staticmethod
    def get_projection_matrix(fov: float = FOV) -> np.ndarray:
        """Create a perspective projection matrix.

        Args:
            fov: Field of view in degrees.

        Returns:
            A 4x4 numpy array that projects 3D points into clip space.
        """
        fov_rad = math.radians(fov)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.array([
                    [f, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0]
                ])
        
        return proj

    @staticmethod
    def norm_scale(v_proj):
        w = v_proj[3]
        # Divide by w to get the normalization
        if w != 0:
            v_proj[0] /= w
            v_proj[1] /= w
            v_proj[2] /= w
            v_proj[3] = 1.0

        # Scale the points from normalized values (-1,1)
        scale = WIDTH / 2
        screen_x = int(v_proj[0] * scale + WIDTH / 2)
        screen_y = int(v_proj[1] * scale + HEIGHT / 2)

        return screen_x, screen_y

def load_obj(filename: Path) -> tuple[vertices_d, set[tuple[int, int]]]:
    """Load vertex and edge data from a Wavefront OBJ file.

    Args:
        filename: Path to the OBJ file.

    Returns:
        A tuple containing a numpy array of vertices and a list of unique edges.
    """
    vertices: vertices_d = vertices_d()
    edges: set[tuple[int, int]] = set()
    with open(filename) as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            elif parts[0] == 'v':
                xyz = np.array((float(parts[1]), float(parts[2]), float(parts[3])))
                vertices.cords.append(xyz)
            elif parts[0] == 'f':
                # Skipping anything after a "/" in indice description
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                
                for i in range(len(face_indices)):
                    p1 = face_indices[i]
                    p2 = face_indices[(i + 1) % len(face_indices)]
                    
                    edge = (min(p1, p2), max(p1, p2))
                    edges.add(edge)
                    vertices.neighbours[p1].append(p2)
                    vertices.neighbours[p2].append(p1)

    return vertices, edges


def main():
    """Run the PyGame render loop for the virtual camera viewer."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Virtual Camera")
    clock = pygame.time.Clock()

    # Starting camera cords
    starting_pos: list[float] = [0, -1, -5]
    starting_rot: list[float] = [0, 0, 0]
    camera = Camera(starting_pos, starting_rot)

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
            # Clsoe the programm
            if event.type == pygame.QUIT:
                running = False
            # Zoom and unzoom wityh mouse scroll
            elif event.type == pygame.MOUSEWHEEL:
                current_fov -= event.y * 5
                current_fov = max(10.0, min(170.0, current_fov))
            elif event.type == pygame.KEYDOWN:
                # Take a screenshot with F12
                if event.key == pygame.K_F12:
                    take_screenshot = True
                # Reset position, rotaion and FOV with R
                if event.key == pygame.K_r:
                    camera.pos = np.array(starting_pos, dtype=float)
                    camera.rot = np.array(starting_rot, dtype=float)
                    current_fov = FOV
                # Exit with ESC
                if event.key == pygame.K_ESCAPE:
                    running = False


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
        move_dir = np.zeros(4) # Tracking only the differencial movement vector
        move_dir[3] = 1.0
        if keys[pygame.K_w]: move_dir[2] += speed # Forward
        if keys[pygame.K_s]: move_dir[2] -= speed # Bakcwards
        if keys[pygame.K_a]: move_dir[0] -= speed # Left
        if keys[pygame.K_d]: move_dir[0] += speed # Right
        if keys[pygame.K_q]: move_dir[1] -= speed # Up
        if keys[pygame.K_e]: move_dir[1] += speed # Down
        
        # Rotate and translate the small movement differencial vector
        if np.any(move_dir[:3] != 0.0):
            rotation_matrix = camera.get_model_matrix()
            move_dir = rotation_matrix @ move_dir
            camera.pos = move_dir

        model_matrix = camera.get_model_matrix()
        view_matrix = np.linalg.inv(model_matrix)
        proj_matrix = camera.get_projection_matrix(fov=current_fov)

        projected_points: list[tuple[int, int] | None] = []

        for i, v in enumerate(vertices.cords):
            v_4d = np.array([v[0], v[1], v[2], 1.0])
            # Apply view matrix
            v_view = view_matrix @ v_4d 

            # If current vertice is in back
            if v_view[2] < 0.01:
                idxs_v_neighbours = vertices.neighbours[i]
                for idx_v_neigh in idxs_v_neighbours:
                    # Apply view matrix to neighbour vertice
                    v_neigh = vertices.cords[idx_v_neigh]
                    v_neigh_4d = np.concatenate((v_neigh, [1.0]))
                    v_neigh_view = view_matrix @ v_neigh_4d
                    # If neighbour vertice is in front
                    if v_neigh_view[2] >= 0.01:
                        u_vect = v_neigh_view - v_view
                        z = 0.01
                        t = (z - v_view[2]) / u_vect[2]
                        y = v_view[1] + t * u_vect[1]
                        x = v_view[0] + t * u_vect[0]
                        # Apply projection, normalize and scale neighbour vertice
                        v_neigh_proj = proj_matrix @ v_neigh_view
                        v_neigh_norm = camera.norm_scale(v_neigh_proj)
                        # Apply projection, normalize and scale new vertice
                        v_mid_view = np.array((x, y, z, 1.0))
                        v_mid_proj = proj_matrix @ v_mid_view
                        v_mid_norm = camera.norm_scale(v_mid_proj)

                        pygame.draw.line(screen, (0, 255, 0), v_mid_norm, v_neigh_norm, 2)

                projected_points.append(None)
            else:
                # Apply projection matrix
                v_proj = proj_matrix @ v_view
                # Normalize and scale
                screen_x, screen_y = camera.norm_scale(v_proj)
                
                projected_points.append((screen_x, screen_y))

        for edge in edges:
            p1 = projected_points[edge[0]]
            p2 = projected_points[edge[1]]
            
            # Don't draw any edge with a None (hidden) vertice
            if p1 is not None and p2 is not None:
                pygame.draw.line(screen, (0, 255, 0), p1, p2, 2)

        rot_deg = np.degrees(camera.rot)
        # Positional and FOV information
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
