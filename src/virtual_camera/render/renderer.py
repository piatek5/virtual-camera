from virtual_camera.render.transform import get_model_matrix, get_projection_matrix
from virtual_camera.render.bsp_tree import BSP_tree, Node
from virtual_camera.render.camera import Camera
import virtual_camera.config as config

import pygame
import numpy as np

class Renderer:
    Z_NEAR = 0.1

    def __init__(self, 
        camera: Camera, 
        screen: pygame.Surface, 
        polygons: list[list[int]],
        coords: np.ndarray
    ):
        self.camera = camera
        self.screen = screen
        self.bsp_tree = BSP_tree()
        self.bsp_tree.build(coords, polygons)
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.aspect_ratio = config.WIDTH / config.HEIGHT

    def render(self):
        model_matrix = get_model_matrix(self.camera.pos, self.camera.rot)
        view_matrix = np.linalg.inv(model_matrix)
        proj_matrix = get_projection_matrix(self.camera.fov, self.aspect_ratio)

        # Calculate view vertices ahead of time.
        # The view matrix is (4, 4) but coordinates are (V, 3). To skip adding 
        # a fourth dimension, we only do matrix calculations for the first 3 
        # rows. To keep coords (V, 3) and not make them (V, 4) the fourth
        # column is skipped. Addition is for the last row and implicitly
        # multiplied by a fourth dim (w) value equal to 1.0.
        view_coords = self.bsp_tree.coords @ view_matrix.T[:3,:3] + view_matrix.T[3, :3]

        # Apply projection and normalization to every view vertice.
        # Fourth dimension for coordinates will be needed and so all columns
        # of projection matrix are used to produce (V, 4). The last row of 
        # projection matrix is zeros so can be skipped.
        proj_coords = view_coords @ proj_matrix.T[:3, :]
        screan_coords = self.norm_scale(proj_coords)

        in_front = view_coords[:, 2] >= self.Z_NEAR

        self._parse_tree(self.bsp_tree.root, view_coords, proj_matrix, screan_coords, in_front)

    def _parse_tree(
        self,
        node: Node | None,
        view_coords: np.ndarray,
        proj_matrix: np.ndarray,
        screan_coords: np.ndarray,
        in_front: np.ndarray
    ):
        if node is None:
            return
        n, d = node.plane.n, node.plane.d
        # If camera is positioned in front of current plane (polygon)
        if (np.dot(n, self.camera.pos) + d) > 0:
            self._parse_tree(node.back, view_coords, proj_matrix, screan_coords, in_front)
            for polygon_indexes in node.get_polygon_indexes():
                self._draw(polygon_indexes, node.color, view_coords, proj_matrix, screan_coords, in_front)
            self._parse_tree(node.front, view_coords, proj_matrix, screan_coords, in_front)
        else:
            self._parse_tree(node.front, view_coords, proj_matrix, screan_coords, in_front)
            for polygon_indexes in node.get_polygon_indexes():
                self._draw(polygon_indexes, node.color, view_coords, proj_matrix, screan_coords, in_front)
            self._parse_tree(node.back, view_coords, proj_matrix, screan_coords, in_front)

    def _draw(self, 
        polygon_indexes: list[int], 
        color: tuple[int, ...],
        view_coords: np.ndarray,
        proj_matrix: np.ndarray,
        screan_coords: np.ndarray,
        in_front: np.ndarray
    ):
        polygon_in_front = in_front[polygon_indexes]

        if not polygon_in_front.any():
            return
        
        elif polygon_in_front.all():
            curr_screan_coords = [screan_coords[idx] for idx in polygon_indexes]
            pygame.draw.polygon(self.screen, color, curr_screan_coords) 
            return
        
        else:
            z_near = self.Z_NEAR
            curr_view_coords = []
            verts_num = len(polygon_indexes)
            # Go through every vertice pair of a polygon to check 
            # if some edges need to be clipped
            for i in range(verts_num):
                a_view = view_coords[polygon_indexes[i]]
                b_idx = (i + 1) % verts_num
                b_view = view_coords[polygon_indexes[b_idx]]
                a_infront = a_view[2] >= z_near
                b_infront = b_view[2] >= z_near
                if a_infront or b_infront:
                    if not (a_infront and b_infront):
                        # Calculate the intersection point 
                        # of current edge and near plane (clipping point)
                        u = a_view - b_view 
                        t = (z_near - b_view[2]) / u[2]
                        x = b_view[0] + t * u[0]
                        y = b_view[1] + t * u[1]     

                        # Keep current vertice if it is viisble
                        if a_infront:
                            curr_view_coords.append(a_view)

                        curr_view_coords.append(np.array((x, y, z_near)))

                    else:
                        curr_view_coords.append(a_view)

            curr_view_coords = np.array(curr_view_coords)

            # Minimum for a polygon
            if curr_view_coords.shape[0] >= 3:
                # Apply projection and normalization to every view vertice. Picking
                # certain rows for proj_matrix explained before.
                projected = curr_view_coords @ proj_matrix.T[:3, :]
                screen_coords = self.norm_scale(projected)

                pygame.draw.polygon(self.screen, color, screen_coords.tolist())        

    def norm_scale(self, coords: np.ndarray) -> np.ndarray:
        x = coords[:, 0]
        y = coords[:, 1]
        w = coords[:, 3]

        w = np.where(w == 0, self.Z_NEAR, w)

        # Divide by w to get the normalization
        norm_x = x / w
        norm_y = y / w

        # Scale values by screen width and height
        scale_x = self.width
        scale_y = self.height
        screen_x = (norm_x * scale_x / 2 + self.width / 2)
        screen_y = (-norm_y * scale_y / 2 + self.height / 2)

        points_2d = np.column_stack((screen_x, screen_y))

        return points_2d
