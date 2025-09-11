import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


class Animation:
    def __init__(
        self, points, triangles, oil_distribution_values, velocities, cells, neighbors_triangles, midpoints, areas
    ):
        self.points = points
        self.triangles = triangles
        self.oil_distribution_values = oil_distribution_values
        self.velocities = velocities
        self.cells = cells
        self.neighbors_triangles = neighbors_triangles
        self.midpoints = midpoints
        self.areas = areas
        self.triang = Triangulation(self.points[:, 0], self.points[:, 1], self.triangles)
        self.umin, self.umax = (
            np.min(oil_distribution_values[: len(triangles)]),
            np.max(oil_distribution_values[: len(triangles)]),
        )

    def setup_colorbar(self, im, fig, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0)
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label("Amount of oil")  # Add label to the colorbar
        return colorbar

    def setup_plot(self):
        fig, ax = plt.subplots()
        self.sm = plt.cm.ScalarMappable(cmap="viridis")
        self.sm.set_array(self.oil_distribution_values[: len(self.triangles)])

        facecolors = (self.oil_distribution_values[: len(self.triangles)] - self.umin) / (self.umax - self.umin)
        im = ax.tripcolor(
            self.triang,
            facecolors=facecolors,
            cmap="viridis",
            vmin=self.umin,
            vmax=self.umax,
            edgecolors="face",
            alpha=0.9,
        )
        colorbar = self.setup_colorbar(im, fig, ax)

        ax.set_title("Time: 0.0")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig("Oil_Time_0.png")

        return fig, ax, im, colorbar

    def update_plot(self, frame, fig, im, colorbar, update_oil_concentration_in_place, delta_t):
        self.oil_distribution_values = update_oil_concentration_in_place(
            self.oil_distribution_values,
            delta_t,
            self.velocities,
            self.cells,
            self.neighbors_triangles,
            self.midpoints,
            self.points,
            self.areas,
        )
        facecolors = (self.oil_distribution_values[: len(self.triangles)] - self.umin) / (self.umax - self.umin)
        im.set_array(facecolors)

        # Update the color limits dynamically
        umin, umax = (
            np.min(self.oil_distribution_values[: len(self.triangles)]),
            np.max(self.oil_distribution_values[: len(self.triangles)]),
        )

        # Update ScalarMappable and colorbar
        self.sm.set_array(self.oil_distribution_values[: len(self.triangles)])
        self.sm.set_clim(umin, umax)
        colorbar.update_normal(self.sm)

        # Updating plot with the current time
        current_time = frame * delta_t
        fig.axes[0].set_title(f"Time: {current_time:.2f}")

        return (im,)

    def create_animation(self, update_oil_concentration_in_place, n_steps, delta_t, output_file="simulation.mp4"):
        fig, ax, im, colorbar = self.setup_plot()

        # progress bar creation
        progress_bar = tqdm(total=n_steps)

        def wrapped_update(frame):
            result = self.update_plot(frame, fig, im, colorbar, update_oil_concentration_in_place, delta_t)
            progress_bar.update(1)
            return result

        anim = FuncAnimation(fig, wrapped_update, frames=n_steps, blit=False)

        # Save the animation directly to the requested output file
        writer = FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
        anim.save(output_file, writer=writer)  # uses the passed filename
