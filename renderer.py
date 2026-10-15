from bsp_tree import BSP_tree, Node
from camera import Camera

import pygame
import numpy as np

class Renderer:
    Z_NEAR = 0.1

    def __init__(self, camera: Camera, screen: pygame.Surface, polygons: list[list[int]], coords: np.ndarray):
        self.camera = camera
        self.width = camera.WIDTH
        self.height = camera.HEIGHT
        self.screen = screen
        self.bsp_tree = BSP_tree().build(coords, polygons)

    def render(self):
        model_matrix = self.camera.get_model_matrix()
        view_matrix = np.linalg.inv(model_matrix)
        proj_matrix = self.camera.get_projection_matrix()
        self._parse_tree(self.bsp_tree, view_matrix, proj_matrix)

    def _parse_tree(
        self,
        node: Node | None,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ):
        if node is None:
            return
        n, d = node.plane.n, node.plane.d
        # If camera is positioned in front of current plane (polygon)
        if (np.dot(n, self.camera.pos) + d) > 0:
            self._parse_tree(node.back, view_matrix, proj_matrix)
            for polygon_coords in node.get_polygon_coords():
                self._draw(polygon_coords, node.color, view_matrix, proj_matrix)
            self._parse_tree(node.front, view_matrix, proj_matrix)
        else:
            self._parse_tree(node.front, view_matrix, proj_matrix)
            for polygon_coords in node.get_polygon_coords():
                self._draw(polygon_coords, node.color, view_matrix, proj_matrix)
            self._parse_tree(node.back, view_matrix, proj_matrix)

    def _draw(self, 
        polygon_coords: np.ndarray, 
        color: tuple[int, ...],
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray
    ):
        # Calculate view vertices ahead of time
        polygon_coords_4d = np.pad(polygon_coords, ((0, 0), (0, 1)), constant_values=1.0)
        coords_view = polygon_coords_4d @ view_matrix.T

        clipped_coords_view = []
        verts_num = polygon_coords.shape[0]
        z_near = self.Z_NEAR
        # Go through every vertice pair of a polygon to check 
        # if some edges need to be clipped
        for i in range(verts_num):
            a_view = coords_view[i]
            b_idx = (i + 1) % verts_num
            b_view = coords_view[b_idx]
            a_infront = a_view[2] >= 0.1
            b_infront = b_view[2] >= 0.1
            if a_infront or b_infront:
                if not (a_infront and b_infront):
                    # Calculate the intersection point 
                    # of current edge and near plane (clipping point)
                    u = a_view - b_view 
                    t = (z_near - b_view[2]) / u[2]
                    y = b_view[1] + t * u[1]
                    x = b_view[0] + t * u[0]

                    # Keep current vertice if it is viisble
                    if a_infront:
                        clipped_coords_view.append(a_view)

                    clipped_coords_view.append(np.array((x, y, z_near, 1.0)))

                else:
                    clipped_coords_view.append(a_view)

        # Minimum for a polygon
        clipped_coords_view = np.array(clipped_coords_view)
        if clipped_coords_view.shape[0] >= 3:
            # Apply the projection matrix and normalization 
            # to every (clipped) view vertice
            projected = clipped_coords_view @ proj_matrix.T 
            screen_coords = self.norm_scale(projected)

            pygame.draw.polygon(self.screen, color, screen_coords.tolist())        

    def norm_scale(self, coords: np.ndarray) -> np.ndarray:
        scale = self.width / 2
        x = coords[:, 0]
        y = coords[:, 1]
        w = coords[:, 3]

        w = np.where(w == 0, self.Z_NEAR, w)

        # Divide by w to get the normalization
        norm_x = x / w
        norm_y = y / w

        # Scale values by screen width and height
        screen_x = (norm_x * scale + self.width / 2)
        screen_y = (norm_y * scale + self.height / 2)

        points_2d = np.column_stack((screen_x, screen_y))

        return points_2d
