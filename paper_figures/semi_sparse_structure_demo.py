import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from sdf_prior_with_grad import sdf_prior_with_grad_2d, sdf_prior_without_grad_2d

X_MIN = -0.5
X_MAX = 4.1
Y_MIN = -0.5
Y_MAX = 4.1
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
        self.grid_points = self.get_grid_points()
        self.circle_center = np.array([-1.0, -1.0], dtype=np.float64)
        self.circle_radius = 2

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
                linewidth=1.0,
                linestyle="--",
            )
            for v in self.voxels
        ]

        self.fig, self.ax = plt.subplots(figsize=(7.2, 6))
        self.ax.add_patch(self.circle)
        for square in self.squares:
            self.ax.add_patch(square)

    def get_grid_points(self):
        x = np.linspace(0.0, 4.0, GRID_SIZE)
        y = np.linspace(0.0, 4.0, GRID_SIZE)
        return np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape(-1, 2)

    def compute_vertex_prior(self):
        diff = self.vertices - self.circle_center
        dists = np.linalg.norm(diff, axis=-1, keepdims=True)
        grad = diff / dists
        sdf = (dists - self.circle_radius).flatten()
        return sdf, grad

    def interpolate(self):
        sdf0, grad0 = self.compute_vertex_prior()
        n_voxels = self.voxels.shape[0]
        sdf_results = np.zeros((self.grid_points.shape[0],), dtype=np.float64)
        for voxel_id in range(n_voxels):
            voxel_min = self.voxels[voxel_id, :2]
            voxel_max = self.voxels[voxel_id, 2:4]
            in_voxel_mask = np.all((self.grid_points >= voxel_min) & (self.grid_points <= voxel_max), axis=-1)
            vertex_indices = self.vertex_indices[voxel_id]
            sdf_results[in_voxel_mask] = sdf_prior_without_grad_2d(
                sdf0[vertex_indices],
                self.vertices[vertex_indices],
                self.grid_points[in_voxel_mask],
            )
        return sdf_results

    def interpolate_with_gradient(self):
        sdf0, grad0 = self.compute_vertex_prior()
        n_voxels = self.voxels.shape[0]
        sdf_results = np.zeros((self.grid_points.shape[0],), dtype=np.float64)
        for voxel_id in range(n_voxels):
            voxel_min = self.voxels[voxel_id, :2]
            voxel_max = self.voxels[voxel_id, 2:4]
            in_voxel_mask = np.all((self.grid_points >= voxel_min) & (self.grid_points <= voxel_max), axis=-1)
            vertex_indices = self.vertex_indices[voxel_id]
            sdf_results[in_voxel_mask] = sdf_prior_with_grad_2d(
                sdf0[vertex_indices],
                grad0[vertex_indices],
                self.vertices[vertex_indices],
                self.grid_points[in_voxel_mask],
            )
        return sdf_results

    def show(self, with_grad: bool, show_error: bool, output_png: str = None):
        if with_grad:
            sdf = self.interpolate_with_gradient()
        else:
            sdf = self.interpolate()
        sdf = sdf.reshape(GRID_SIZE, GRID_SIZE)
        if show_error:
            sdf_gt = np.linalg.norm(self.grid_points - self.circle_center, axis=-1) - self.circle_radius
            sdf = sdf - sdf_gt.reshape(GRID_SIZE, GRID_SIZE)
        img = self.ax.imshow(
            sdf,
            extent=(0, 4, 0, 4),
            origin="lower",
            cmap="jet",
        )
        self.fig.colorbar(
            img,
            fraction=0.04,
            pad=0.0,
        ).ax.tick_params(labelsize=21)
        if not show_error:
            self.ax.contour(
                sdf,
                levels=10,
                extent=(0, 4, 0, 4),
                origin="lower",
                colors="k",
                linewidths=1.0,
            )
        self.ax.axis("off")
        plt.tight_layout()
        self.ax.axis("equal")
        self.ax.set_xlim(X_MIN, X_MAX)
        self.ax.set_ylim(Y_MIN, Y_MAX)
        if output_png is not None:
            plt.savefig(output_png, dpi=300)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semi-sparse", action="store_true", help="Use semi-sparse structure")
    parser.add_argument("--with-grad", action="store_true", help="Use gradient for interpolation")
    parser.add_argument("--show-error", action="store_true", help="Show error visualization")
    parser.add_argument("--output-png", type=str, help="Output PNG file path")
    args = parser.parse_args()
    demo = Demo(semi_sparse=args.semi_sparse)
    demo.show(args.with_grad, args.show_error, args.output_png)


if __name__ == "__main__":
    main()
