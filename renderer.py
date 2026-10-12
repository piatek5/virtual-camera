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

    def render(self, vertices: vertices_d, polygons: list[list[int]]):
        model_matrix = self.camera.get_model_matrix()
        view_matrix = np.linalg.inv(model_matrix)
        proj_matrix = self.camera.get_projection_matrix()
        
        for polygon in polygons:
            views = []
            # Calculate view vertices ahead of time
            for i in range(len(polygon)):
                vert_4d = np.concatenate((vertices.cords[polygon[i]], [1.0]))
                vert_view = view_matrix @ vert_4d
                views.append(vert_view)

            clipped_views = []
            z_near = 0.01
            # Go through every vertice pair of a polygon to check 
            # if some edges need to be clipped
            for i in range(len(polygon)):
                a_view = views[i]
                b_idx = (i + 1) % len(polygon)
                b_view = views[b_idx]
                a_infront = a_view[2] >= 0.01
                b_infront = b_view[2] >= 0.01
                if a_infront or b_infront:
                    if not (a_infront and b_infront):
                        # Calculate the intersection point 
                        # of edge and near plane (clipping point)
                        u = a_view - b_view 
                        t = (z_near - b_view[2]) / u[2]
                        y = b_view[1] + t * u[1]
                        x = b_view[0] + t * u[0]

                        # Keep one of the vertices that happens to be visible
                        if a_infront:
                            clipped_views.append(a_view)

                        clipped_views.append(np.array((x, y, z_near, 1.0)))

                    else:
                        clipped_views.append(a_view)

            # Minimum for a polygon
            if len(clipped_views) >= 3:
                screen_vertices= []
                # Apply the projection matrix and normalization 
                # to every (clipped) view vertice
                for v_view in clipped_views:
                    v_proj = proj_matrix @ v_view
                    v_norm = self.norm_scale(v_proj)
                    screen_vertices.append(v_norm)

                pygame.draw.polygon(self.screen, (0, 255, 0), screen_vertices)

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
