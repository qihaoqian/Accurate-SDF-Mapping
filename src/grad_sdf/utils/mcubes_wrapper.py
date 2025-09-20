import mcubes
import numpy as np


class MarchingCubesWrapper:
    """
    A wrapper for mcubes to have the same interface as erl_geometry.MarchingCubes.
    """

    @staticmethod
    def collect_valid_cubes(*args, **kwargs):
        print("Warning: Using mcubes which does not support collect_valid_cubes. Returning nothing.")
        return [[]]

    @staticmethod
    def process_valid_cubes(valid_cubes, coords_min, grid_res, grid_shape, grid_values, iso_value, *args, **kwargs):
        print("Warning: Using mcubes which ignores valid_cubes")
        assert grid_values.shape[0] == grid_shape[0]
        assert grid_values.shape[1] == grid_shape[1]
        assert grid_values.shape[2] == grid_shape[2]

        vertices, triangles = mcubes.marching_cubes(grid_values, iso_value)
        vertices = vertices * grid_res + coords_min
        triangles = triangles[:, ::-1]

        p0 = vertices[triangles[:, 0], :]
        p1 = vertices[triangles[:, 1], :]
        p2 = vertices[triangles[:, 2], :]

        face_normals = np.cross(p1 - p0, p2 - p0)  # (n_faces, 3)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        face_normals = face_normals / norms

        vertices = vertices.astype(np.float64)
        triangles = triangles.astype(np.int32)
        face_normals = face_normals.astype(np.float64)

        return vertices, triangles, face_normals
