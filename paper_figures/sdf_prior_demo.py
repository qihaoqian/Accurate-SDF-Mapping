import argparse
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, Rectangle
from numerical_hessian import numerical_hessian
from sdf_prior_with_grad import sdf_prior_with_grad_2d, sdf_prior_without_grad_2d


class SdfPriorDemo:
    X_MIN = -5
    X_MAX = 5
    Y_MIN = -5
    Y_MAX = 5
    BOX_X_MIN = -1.0
    BOX_X_MAX = 1.0
    BOX_Y_MIN = -1.0
    BOX_Y_MAX = 1.0
    GRID_SIZE = 300

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--load", type=str, help="Path to load circles from")
        parser.add_argument("--x-min", type=float, default=self.X_MIN, help="Minimum x-axis value")
        parser.add_argument("--x-max", type=float, default=self.X_MAX, help="Maximum x-axis value")
        parser.add_argument("--y-min", type=float, default=self.Y_MIN, help="Minimum y-axis value")
        parser.add_argument("--y-max", type=float, default=self.Y_MAX, help="Maximum y-axis value")
        parser.add_argument("--no-gradients", action="store_false", help="Disable gradients")
        parser.add_argument("--draw-error", action="store_true", help="Draw error between GT and prior")
        parser.add_argument("--draw-gt", action="store_true", help="Draw ground truth SDF")
        parser.add_argument("--output-png", type=str, help="Path to save output PNG")
        parser.add_argument("--no-title", action="store_true", help="Disable title")
        parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
        self.args = parser.parse_args()

        self.X_MIN = self.args.x_min
        self.X_MAX = self.args.x_max
        self.Y_MIN = self.args.y_min
        self.Y_MAX = self.args.y_max

        self.fig, self.ax = plt.subplots()
        # self.ax.set_title("Draw circles by click and drag")

        self.circles = []
        self.circle_centers = []
        self.center_scatter = None
        self.press = None
        self.current_circle = None
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.ref_points = np.array(
            [
                [self.BOX_X_MIN, self.BOX_Y_MIN],
                [self.BOX_X_MAX, self.BOX_Y_MIN],
                [self.BOX_X_MIN, self.BOX_Y_MAX],
                [self.BOX_X_MAX, self.BOX_Y_MAX],
            ]
        )
        xs, ys = np.meshgrid(
            np.linspace(self.BOX_X_MIN, self.BOX_X_MAX, self.GRID_SIZE),
            np.linspace(self.BOX_Y_MIN, self.BOX_Y_MAX, self.GRID_SIZE),
            indexing="xy",
        )
        self.grid = np.stack([xs, ys], axis=-1).reshape(-1, 2)  # (RES*RES, 2)
        self.grid_img = None
        self.colorbar = None
        self.with_gradients = self.args.no_gradients
        self.draw_error = self.args.draw_error
        self.draw_gt = self.args.draw_gt
        self.grad_arrows = []

        self.ax.add_patch(
            Rectangle(
                (self.BOX_X_MIN, self.BOX_Y_MIN),
                self.BOX_X_MAX - self.BOX_X_MIN,
                self.BOX_Y_MAX - self.BOX_Y_MIN,
                fill=False,
                edgecolor="k",
                lw=2,
            )
        )
        self.set_ax_limits()
        self.draw_ref_points()

        if self.args.load:
            self.load(self.args.load)

    def draw_ref_points(self):
        """Draw reference points as red dots"""
        self.ax.scatter(self.ref_points[:, 0], self.ref_points[:, 1], color="red", s=50, zorder=5, marker="s")

    def set_ax_limits(self):
        self.ax.tick_params(axis="both", which="major", labelsize=21)  # set axis font size
        plt.tight_layout()
        plt.axis("equal")
        self.ax.set_xlim(self.X_MIN, self.X_MAX)
        self.ax.set_ylim(self.Y_MIN, self.Y_MAX)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press = (event.xdata, event.ydata)
        self.current_circle = Circle((event.xdata, event.ydata), 0, fill=True, edgecolor="r", facecolor="r", alpha=0.5)
        self.ax.add_patch(self.current_circle)
        self.circles.append(self.current_circle)
        self.circle_centers.append((event.xdata, event.ydata))
        if self.center_scatter:
            self.center_scatter.remove()
        self.center_scatter = self.ax.scatter(*zip(*self.circle_centers), color="b", s=30, zorder=3)

        self.set_ax_limits()
        self.fig.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        x0, y0 = self.press
        dx = event.xdata - x0
        dy = event.ydata - y0
        radius = (dx**2 + dy**2) ** 0.5
        self.current_circle.set_radius(radius)
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        print("Circle drawn at:", self.current_circle.center, "with radius:", self.current_circle.radius)
        self.current_circle = None
        self.press = None

        self.update_grid_visualization()

        self.set_ax_limits()
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == "c":
            # Clear all circles
            for circle in self.circles:
                circle.remove()
            self.circles.clear()
            self.circle_centers.clear()

            # Remove scatter points
            if self.center_scatter:
                self.center_scatter.remove()
                self.center_scatter = None

            # Remove grid image
            if self.grid_img:
                self.grid_img.remove()
                self.grid_img = None

            # Remove gradient arrows
            for arrow in self.grad_arrows:
                arrow.remove()
            self.grad_arrows.clear()

            self.draw_error = False
            self.draw_gt = False
            self.fig.canvas.draw()

        elif event.key == "e":
            # Toggle error visualization
            self.draw_error = not self.draw_error
            self.draw_gt = False
            if self.grid_img:
                self.grid_img.remove()
                self.grid_img = None
            self.update_grid_visualization()

        elif event.key == "d":
            # Toggle if interpolation with gradients is used
            self.with_gradients = not self.with_gradients
            self.update_grid_visualization()

        elif event.key == "t":
            # Toggle ground truth visualization
            self.draw_gt = not self.draw_gt
            self.draw_error = False
            if self.grid_img:
                self.grid_img.remove()
                self.grid_img = None
            self.update_grid_visualization()

        elif event.key == "n":
            # save circles
            circles = [
                [
                    float(circle.center[0]),
                    float(circle.center[1]),
                    float(circle.radius),
                ]
                for circle in self.circles
            ]
            data = dict(
                circles=circles,
                x_min=self.X_MIN,
                x_max=self.X_MAX,
                y_min=self.Y_MIN,
                y_max=self.Y_MAX,
            )
            with open("data.yaml", "w") as f:
                yaml.dump(data, f)

        elif event.key == "m":
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            file_path = filedialog.askopenfilename(
                title="Select a file",
                filetypes=(("Text files", "*.yaml"), ("All files", "*.*")),
            )
            if file_path:
                print("Loading circles from:", file_path)
                self.load(file_path)

    def load(self, file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        circles = data.get("circles", [])
        self.X_MIN = data.get("x_min", self.X_MIN)
        self.X_MAX = data.get("x_max", self.X_MAX)
        self.Y_MIN = data.get("y_min", self.Y_MIN)
        self.Y_MAX = data.get("y_max", self.Y_MAX)
        for circle in self.circles:
            circle.remove()
        self.circles.clear()
        self.circle_centers.clear()
        for x, y, r in circles:
            new_circle = Circle((x, y), r, fill=True, edgecolor="r", facecolor="r", alpha=0.5)
            self.ax.add_patch(new_circle)
            self.circles.append(new_circle)
            self.circle_centers.append((x, y))
        if self.center_scatter:
            self.center_scatter.remove()
        if self.circles:
            self.center_scatter = self.ax.scatter(*zip(*self.circle_centers), color="b", s=30, zorder=3)
        else:
            self.center_scatter = None
        self.update_grid_visualization()

    def update_grid_visualization(self):
        """Update the grid visualization based on current mode"""
        if not self.circles:
            return
        error_bound = self.compute_error_bound()
        print("Current error bound:", error_bound)
        if self.draw_error:
            # Show error (difference between ground truth and prior)
            sdf_gt = self.compute_grid_gt().reshape(self.GRID_SIZE, self.GRID_SIZE)
            sdf_prior = self.compute_grid_prior().reshape(self.GRID_SIZE, self.GRID_SIZE)
            data_to_show = sdf_prior - sdf_gt
        elif self.draw_gt:
            # Show ground truth
            data_to_show = self.compute_grid_gt().reshape(self.GRID_SIZE, self.GRID_SIZE)
        else:
            # Show prior
            data_to_show = self.compute_grid_prior().reshape(self.GRID_SIZE, self.GRID_SIZE)

        if self.grid_img:
            self.grid_img.set_data(data_to_show)
        else:
            self.grid_img = self.ax.imshow(
                data_to_show,
                extent=(self.BOX_X_MIN, self.BOX_X_MAX, self.BOX_Y_MIN, self.BOX_Y_MAX),
                origin="lower",
                cmap="jet",
                alpha=1.0,
                zorder=1,
            )

        if self.colorbar:
            # Update the image data and colorbar manually
            self.grid_img.set_clim(vmin=np.min(data_to_show), vmax=np.max(data_to_show))
            self.colorbar.update_normal(self.grid_img)
            # self.colorbar.set_label(title)
        else:
            self.colorbar = self.fig.colorbar(self.grid_img, ax=self.ax)
            self.colorbar.ax.tick_params(labelsize=21)  # set colorbar font size

        if not self.args.no_title:
            # Set title based on mode
            if self.draw_error:
                title = "Error (Prior - GT)"
                if self.with_gradients:
                    title += " (with Gradients)"
                else:
                    title += " (without Gradients)"
            elif self.draw_gt:
                title = "SDF Ground Truth"
            else:
                title = "Interpolation"
                if self.with_gradients:
                    title += " (with Gradients)"
                else:
                    title += " (without Gradients)"
            self.ax.set_title(title)

        # Draw gradient arrows at vertex points
        if self.with_gradients:
            self.draw_grad_arrows()
        else:
            for arrow in self.grad_arrows:
                arrow.remove()
            self.grad_arrows.clear()

        plt.tight_layout()
        self.set_ax_limits()
        self.fig.canvas.draw()
        if self.args.output_png:
            self.fig.savefig(self.args.output_png)

    def draw_grad_arrows(self):
        """Draw gradient arrows at vertex points"""
        # Remove existing arrows
        for arrow in self.grad_arrows:
            arrow.remove()
        self.grad_arrows.clear()

        # Compute gradients at vertex points
        sdf0, grad = self.compute_vertex_prior()

        # Draw arrows
        for i, (point, gradient) in enumerate(zip(self.ref_points, grad)):
            arrow = self.ax.arrow(
                point[0],
                point[1],
                gradient[0] * 0.2,
                gradient[1] * 0.2,  # Scale factor for visibility
                head_width=0.05,
                head_length=0.05,
                fc="black",
                ec="black",
                linewidth=2,
                zorder=6,
            )
            self.grad_arrows.append(arrow)

    def show(self):
        if not self.args.no_interactive:
            plt.show()

    def draw_grid(self):
        """Draw grid lines within the box region"""
        # Create grid lines
        x_lines = np.linspace(self.BOX_X_MIN, self.BOX_X_MAX, 11)  # 10 intervals
        y_lines = np.linspace(self.BOX_Y_MIN, self.BOX_Y_MAX, 11)  # 10 intervals

        # Draw vertical lines
        for x in x_lines:
            self.ax.plot([x, x], [self.BOX_Y_MIN, self.BOX_Y_MAX], "k-", alpha=0.3, linewidth=0.5)

        # Draw horizontal lines
        for y in y_lines:
            self.ax.plot([self.BOX_X_MIN, self.BOX_X_MAX], [y, y], "k-", alpha=0.3, linewidth=0.5)

    def compute_sdf(self, points: np.ndarray, with_grad: bool = False):
        centers = np.array(self.circle_centers)  # (M, 2)
        radii = np.array([circle.radius for circle in self.circles])  # (M, )

        dists = np.linalg.norm(  # (N, M)
            points.reshape(-1, 1, 2) - centers.reshape(1, -1, 2),
            axis=2,
        ) - radii.reshape(1, -1)
        indices = np.argmin(dists, axis=1)
        sdf = dists[np.arange(len(points)), indices]
        if with_grad:
            grad = points - centers[indices]
            grad = grad / np.linalg.norm(grad, axis=1, keepdims=True)
            return sdf, grad
        return sdf, None

    def compute_vertex_prior(self) -> tuple[np.ndarray, np.ndarray]:
        return self.compute_sdf(self.ref_points, with_grad=True)

    def compute_grid_prior(self):
        sdf0, grad = self.compute_vertex_prior()
        if self.with_gradients:
            sdf_prior = sdf_prior_with_grad_2d(sdf0, grad, self.ref_points, self.grid)
        else:
            sdf_prior = sdf_prior_without_grad_2d(sdf0, self.ref_points, self.grid)

        return sdf_prior

    def compute_grid_gt(self):
        return self.compute_sdf(self.grid, with_grad=False)[0]

    def compute_error_bound(self):
        x_size = self.BOX_X_MAX - self.BOX_X_MIN
        y_size = self.BOX_Y_MAX - self.BOX_Y_MIN

        if self.with_gradients:
            sdf = self.compute_grid_gt().reshape(self.GRID_SIZE, self.GRID_SIZE)
            x_res = x_size / self.GRID_SIZE
            y_res = y_size / self.GRID_SIZE
            hessian = numerical_hessian(x_res, y_res, sdf)
            eigvals_abs = np.abs(np.linalg.eigvals(hessian))
            mean = eigvals_abs.mean()
            # remove large outliers where hessian is not defined.
            eigvals_abs = eigvals_abs[eigvals_abs < mean]
            m = eigvals_abs.max()
            return m * (x_size**2 + y_size**2) / 8
        else:
            return np.sqrt(x_size**2 + y_size**2) / 2


if __name__ == "__main__":
    drawer = SdfPriorDemo()
    drawer.show()
