from camera import Camera
from load import vertices_d

import pygame
import numpy as np

class Renderer:
    def __init__(self, camera: Camera, screen: pygame.Surface):
        self.camera = camera
        self.width = camera.WIDTH
        self.height = camera.HEIGHT
        self.screen = screen

    def render(self, vertices: vertices_d, edges: set[tuple[int, int]]):
        model_matrix = self.camera.get_model_matrix()
        view_matrix = np.linalg.inv(model_matrix)
        proj_matrix = self.camera.get_projection_matrix()
        
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
                        v_neigh_norm = self.norm_scale(v_neigh_proj)
                        # Apply projection, normalize and scale new vertice
                        v_mid_view = np.array((x, y, z, 1.0))
                        v_mid_proj = proj_matrix @ v_mid_view
                        v_mid_norm = self.norm_scale(v_mid_proj)

                        pygame.draw.line(self.screen, (0, 255, 0), v_mid_norm, v_neigh_norm, 2)

                projected_points.append(None)
            else:
                # Apply projection matrix
                v_proj = proj_matrix @ v_view
                # Normalize and scale
                screen_x, screen_y = self.norm_scale(v_proj)
                
                projected_points.append((screen_x, screen_y))

        for edge in edges:
            p1 = projected_points[edge[0]]
            p2 = projected_points[edge[1]]
            
            # Don't draw any edge with a None (hidden) vertice
            if p1 is not None and p2 is not None:
                pygame.draw.line(self.screen, (0, 255, 0), p1, p2, 2)

    def norm_scale(self, v_proj):
        w = v_proj[3]
        # Divide by w to get the normalization
        if w != 0:
            v_proj[0] /= w
            v_proj[1] /= w
            v_proj[2] /= w
            v_proj[3] = 1.0

        # Scale the points from normalized values (-1,1)
        scale = self.width / 2
        screen_x = int(v_proj[0] * scale + self.width / 2)
        screen_y = int(v_proj[1] * scale + self.height / 2)

        return screen_x, screen_y
