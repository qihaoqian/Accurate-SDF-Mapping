import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from sdf_prior_with_grad import sdf_prior_with_grad_2d, sdf_prior_without_grad_2d

X_MIN = -5
X_MAX = 5
Y_MIN = -5
Y_MAX = 5
GRID_SIZE = 100


class Demo:
    def __init__(self, semi_sparse: bool):
        self.semi_sparse = semi_sparse
        if semi_sparse:
            # Initialize semi-sparse structure
            pass
            # min, max, depth
            self.voxels = np.array(
                [
                    [0.0, 2.0, 2.0, 4.0, 1],
                    [2.0, 2.0, 4.0, 4.0, 1],
                    [0.0, 0.0, 2.0, 2.0, 1],
                    [2.0, 0.0, 4.0, 2.0, 1],
                    [0.0, 0.0, 1.0, 1.0, 2],
                ],
                dtype=np.float64,
            )
            self.vertex_indices = np.array(
                [
                    [3, 4, 0, 1],
                    [4, 5, 1, 2],
                    [8, 10, 3, 4],
                    [10, 11, 4, 5],
                    [8, 9, 6, 7],
                ],
                dtype=np.int32,
            )
            self.vertices = np.array(
                [
                    [0.0, 4.0],
                    [2.0, 4.0],
                    [4.0, 4.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [4.0, 2.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            # Initialize sparse structure
            self.voxels = np.array(
                [
                    [0.0, 0.0, 4.0, 4.0, 0],
                    [0.0, 0.0, 2.0, 2.0, 1],
                    [0.0, 0.0, 1.0, 1.0, 2],
                ],
                dtype=np.float64,
            )
            self.vertex_indices = np.array(
                [
                    [6, 9, 0, 1],
                    [6, 8, 2, 3],
                    [6, 7, 4, 5],
                ],
                dtype=np.int32,
            )
            self.vertices = np.array(
                [
                    [0.0, 4.0],
                    [4.0, 4.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 0.0],
                ],
                dtype=np.float64,
            )
        self.circle_center = np.array([-1.0, -1.0], dtype=np.float64)
        self.circle_radius = 1.8

        self.circle = Circle(
            self.circle_center,
            self.circle_radius,
            fill=True,
            edgecolor="r",
            facecolor="r",
            alpha=0.5,
        )
        self.squares = [
            Rectangle(
                (v[0], v[1]),
                v[2] - v[0],
                v[3] - v[1],
                fill=False,
                edgecolor="k",
            )
            for v in self.voxels
        ]

        self.fig, self.ax = plt.subplots()
        self.ax.add_patch(self.circle)
        for square in self.squares:
            self.ax.add_patch(square)
        self.ax.set_xlim(X_MIN, X_MAX)
        self.ax.set_ylim(Y_MIN, Y_MAX)

    def get_grid_points(self):
        x = np.linspace(0.0, 4.0, GRID_SIZE)
        y = np.linspace(0.0, 4.0, GRID_SIZE)
        return np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape(-1, 2)

    def compute_vertex_prior(self):
        diff = self.vertices - self.circle_center
        dists = np.linalg.norm(diff, axis=-1, keepdims=True)
        grad = diff / dists
        sdf = dists - self.circle_radius
        return sdf, grad

    def interpolate(self):
        grid_points = self.get_grid_points()
        n_voxels = self.voxels.shape[0]
        sdf0, grad0 = self.compute_vertex_prior()
        sdf_results = np.zeros((self.vertices.shape[0],), dtype=np.float64)

        for voxel_id in range(n_voxels):
            voxel_min = self.voxels[voxel_id, :2]
            voxel_max = self.voxels[voxel_id, 2:4]

            pass
        sdf_prior_without_grad_2d(sdf0, self.vertices, points)

    def interpolate_with_gradient(self):
        sdf_prior_with_grad_2d(sdf0, grad, self.vertices, points)
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semi-sparse", type=bool, help="Use semi-sparse structure")
    args = parser.parse_args()
    demo = Demo(semi_sparse=args.semi_sparse)
    demo.ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
