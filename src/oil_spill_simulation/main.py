import argparse
import logging
import os
import sys

import numpy as np

from .config import TOMLFileReader
from .mesh import Mesh
from .simulation import OilCalculation
from .visualization import Animation


# Set up logging
def setup_logging(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(output_dir, "simulation_summary.log"), "w")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


class Simulation:
    def __init__(self, config, output_dir):
        # Initialize logging
        self.logger = setup_logging(output_dir)

        # Read configuration parameters
        mesh_file = self.resolve_mesh_path(config["geometry"]["meshName"])
        t_start = config["settings"]["tStart"]
        t_end = config["settings"]["tEnd"]
        n_steps = config["settings"]["nSteps"]
        delta_t = (t_end - t_start) / n_steps
        write_frequency = config["settings"].get("writeFrequency", None)

        self.output_dir = output_dir
        self.t_start = t_start
        self.delta_t = delta_t
        self.n_steps = n_steps
        self.write_frequency = write_frequency

        # Initialize mesh
        self.mesh = Mesh(mesh_file)

        # Compute midpoints
        midpoints_triangles = self.mesh.compute_midpoints(self.mesh.triangles)
        midpoints_lines = self.mesh.compute_midpoints(self.mesh.lines, is_line=True)
        self.midpoints = np.concatenate((midpoints_triangles, midpoints_lines))

        print("Starting simulation")

        # Initialize oil calculation
        self.oil_calc = OilCalculation(delta_t)

        # Initialize oil distribution and velocity field
        self.oil_distribution = self.oil_calc.compute_initial_oil_distribution(self.midpoints)
        self.velocities = self.oil_calc.compute_velocities(self.midpoints)

        # Compute areas for triangles
        from .geometry import Geometry

        self.areas = np.array(
            [
                Geometry.triangle_area(self.mesh.points[tri[0]], self.mesh.points[tri[1]], self.mesh.points[tri[2]])
                for tri in self.mesh.triangles
            ]
        )

        # Compute scaled vectors
        self.scaled_vectors = self.oil_calc.compute_scaled_vectors(self.mesh.points, self.mesh.cells, self.midpoints)

        # Borders for calculating oil concentration
        self.borders = config["geometry"]["borders"]

        # Log simulation parameters
        self.log_parameters(config)

    def resolve_mesh_path(self, mesh_file):
        """Resolve mesh file path, checking current directory and data/meshes/"""
        # If absolute path or exists in current directory, use as is
        if os.path.isabs(mesh_file) or os.path.exists(mesh_file):
            return mesh_file

        # Try in data/meshes/ directory
        data_mesh_path = os.path.join("data", "meshes", mesh_file)
        if os.path.exists(data_mesh_path):
            return data_mesh_path

        # If not found anywhere, return original (let meshio handle the error)
        return mesh_file

    def log_parameters(self, config):
        # Log the simulation parameters
        self.logger.info("Simulation Parameters:")
        for section, params in config.items():
            self.logger.info(f"[{section}]")
            for key, value in params.items():
                self.logger.info(f"{key} = {value}")

    def run(self, output_file, final_plot_file, restart_file=None):
        # Load the simulation state if a restart file is provided
        if restart_file and restart_file != "":
            self.oil_distribution, _ = self.oil_calc.load_state(restart_file)
            self.t_start = self.oil_calc.get_restart_time(restart_file)
            self.logger.info(f"Restarting simulation from time: {self.t_start}")

        # Create animation
        animation = Animation(
            self.mesh.points,
            self.mesh.triangles,
            self.oil_distribution,
            self.velocities,
            self.mesh.cells,
            self.mesh.neighbors_triangles,
            self.midpoints,
            self.areas,
        )

        # Create oil updater function
        def oil_updater(oil_distribution_values, delta_t, velocities, cells, neighbors, midpoints, points, areas):
            return self.oil_calc.oil_updater(
                oil_distribution_values,
                velocities,
                self.mesh.triangles,
                neighbors,
                areas,
                self.scaled_vectors,
            )

        # Create enhanced animation with oil tracking
        class TrackedAnimation(Animation):
            def __init__(self, parent_simulation, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent_simulation = parent_simulation
                self.current_time = parent_simulation.t_start
                self.elapsed_time = 0.0  # Track elapsed time from 0
                self.step_count = 0

            def setup_plot(self):
                # Call parent setup_plot method
                fig, ax, im, colorbar = super().setup_plot()

                # Update title to show elapsed time 0.0
                fig.axes[0].set_title("Time: 0.0")

                # Save Oil_Time_0.png in the correct output directory instead of current directory
                import matplotlib.pyplot as plt

                initial_plot_path = os.path.join(self.parent_simulation.output_dir, "Oil_Time_0.png")
                plt.savefig(initial_plot_path)

                return fig, ax, im, colorbar

            def update_plot(self, frame, fig, im, colorbar, update_oil_concentration_in_place, delta_t):
                # Run simulation steps up to the current frame
                write_freq = self.parent_simulation.write_frequency or 1
                steps_to_run = write_freq

                for _ in range(steps_to_run):
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
                    self.step_count += 1
                    self.current_time += delta_t
                    self.elapsed_time += delta_t

                    # Log oil tracking every 50 actual steps
                    if self.step_count % 50 == 0:
                        oil_amount = self.parent_simulation.oil_calc.calculate_oil_in_fishing_grounds(
                            self.oil_distribution_values,
                            self.parent_simulation.midpoints,
                            self.parent_simulation.borders,
                        )
                        self.parent_simulation.logger.info(
                            f"Time {self.current_time:.4f}: Oil in fishing grounds = {oil_amount:.6f}"
                        )

                # Update the plot visualization
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

                # Update plot title with elapsed time
                fig.axes[0].set_title(f"Time: {self.elapsed_time:.2f}")

                return (im,)

        # Create tracked animation
        tracked_animation = TrackedAnimation(
            self,
            self.mesh.points,
            self.mesh.triangles,
            self.oil_distribution,
            self.velocities,
            self.mesh.cells,
            self.mesh.neighbors_triangles,
            self.midpoints,
            self.areas,
        )

        # Calculate animation frames based on writeFrequency
        if self.write_frequency:
            animation_frames = self.n_steps // self.write_frequency
        else:
            animation_frames = self.n_steps

        # Run the animation
        tracked_animation.create_animation(oil_updater, animation_frames, self.delta_t, output_file)

        # Save final plot using the final state from animation
        self.save_final_plot(final_plot_file, tracked_animation.oil_distribution_values)

        # Save final state as text file
        final_time = self.t_start + self.n_steps * self.delta_t
        final_state_file = os.path.join(self.output_dir, "final_simulation_state.txt")
        self.oil_calc.save_state(
            final_state_file, tracked_animation.oil_distribution_values, final_time, self.midpoints
        )
        self.logger.info(f"Final state saved to {final_state_file}")

        # Log final simulation summary
        self.log_final_summary(tracked_animation.oil_distribution_values)

    def save_final_plot(self, final_plot_file, final_oil_distribution):
        """Create and save a final plot of oil distribution"""
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.triangles)

        # Plot the final oil distribution (only triangles)
        im = ax.tripcolor(
            triang, final_oil_distribution[: len(self.mesh.triangles)], cmap="viridis", edgecolors="face", alpha=0.9
        )

        ax.set_title("Final Oil Distribution")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add colorbar with same setup as animation (no spacing)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0)
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label("Amount of oil")

        plt.savefig(final_plot_file)
        plt.close()

        self.logger.info(f"Final plot saved to {final_plot_file}")

    def log_final_summary(self, final_oil_distribution):
        """Log comprehensive simulation summary"""
        self.logger.info("=" * 50)
        self.logger.info("SIMULATION SUMMARY")
        self.logger.info("=" * 50)

        # Final time and oil statistics
        final_time = self.t_start + self.n_steps * self.delta_t
        final_oil = self.oil_calc.calculate_oil_in_fishing_grounds(final_oil_distribution, self.midpoints, self.borders)
        total_oil = np.sum(final_oil_distribution)

        self.logger.info(f"Final simulation time: {final_time:.4f}")
        self.logger.info(f"Total oil in domain: {total_oil:.6f}")
        self.logger.info(f"Final oil in fishing grounds: {final_oil:.6f}")
        self.logger.info(f"Percentage in fishing grounds: {(final_oil / total_oil) * 100:.2f}%")

        # Simulation performance
        self.logger.info(f"Total simulation steps: {self.n_steps}")
        self.logger.info(f"Time step size: {self.delta_t:.6f}")
        if self.write_frequency:
            self.logger.info(f"Video frames created: {self.n_steps // self.write_frequency}")

        self.logger.info("Simulation completed successfully")
        self.logger.info("=" * 50)

    def calculate_fishing_ground_oil_concentration(self):
        # Calculate and log the final average oil concentration within the fishing grounds
        final_oil = self.oil_calc.calculate_oil_in_fishing_grounds(self.oil_distribution, self.midpoints, self.borders)
        self.logger.info(f"Final oil concentration within the fishing grounds: {final_oil:.6f}")

    def save_state(self, filename):
        # Save the current state of the simulation
        self.oil_calc.save_state(filename, self.oil_distribution, self.t_start, self.midpoints)

    def load_state(self, filename):
        # Load a saved state of the simulation
        self.oil_distribution, _ = self.oil_calc.load_state(filename)
        self.t_start = self.oil_calc.get_restart_time(filename)


def list_config_files(directory):
    # List all TOML configuration files in a directory, excluding pyproject.toml
    toml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".toml") and file != "pyproject.toml":
                toml_files.append(os.path.join(root, file))
    return toml_files


def ensure_directory_exists(directory):
    # Ensure that the specified directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif not os.path.isdir(directory):
        raise Exception(f"Path {directory} exists and is not a directory.")
    else:
        # Check if directory contains simulation results (has expected files)
        expected_files = ["simulation_summary.log"]
        has_expected_files = any(os.path.exists(os.path.join(directory, f)) for f in expected_files)

        if not has_expected_files:
            # Directory exists but doesn't look like a results folder
            raise Exception(
                f"Directory {directory} already exists and may not be a simulation results folder. "
                f"Please remove it or use a different output location."
            )


def main():
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description="Run oil spill simulation.")
    parser.add_argument("--config-file", type=str, help="Path to a specific configuration file")
    parser.add_argument("--config-dir", type=str, help="Directory containing configuration files", default=".")
    parser.add_argument("--save", action="store_true", help="Save the simulation state after running")
    parser.add_argument("--load", action="store_true", help="Load the simulation state before running")

    args = parser.parse_args()

    # Determine which configuration files to use
    if args.config_file:
        config_files = [args.config_file]
    else:
        config_files = list_config_files(args.config_dir)
        if not config_files:
            default_config_file = os.path.join(args.config_dir, "input.toml")
            if os.path.exists(default_config_file):
                config_files = [default_config_file]
            else:
                print(f"No configuration files found in the directory: {args.config_dir}")
                print("Please specify a configuration file using --config-file option.")
                sys.exit(1)

    print("Configuration files found:")
    for i, config_file in enumerate(config_files):
        print(f"{i + 1}. {config_file}")

    choice = input(
        "Enter 'all' to run all configuration files or select specific ones by entering their numbers (comma-separated): "
    ).strip()

    if choice.lower() == "all":
        selected_files = config_files
    else:
        indices = [int(x) - 1 for x in choice.split(",")]
        selected_files = [config_files[i] for i in indices]

    for config_file in selected_files:
        print(f"Processing config file: {config_file}")
        reader = TOMLFileReader(config_file)
        config = reader.get_config()

        # Create meaningful output directory name
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        if config_name == "input":
            # Use more descriptive name for input.toml
            output_dir = os.path.join(args.config_dir, "simulation_results")
        else:
            output_dir = os.path.join(args.config_dir, f"{config_name}_results")
        ensure_directory_exists(output_dir)

        # Create a Simulation instance and run the simulation
        simulation = Simulation(config, output_dir)
        animation_file = os.path.join(output_dir, config["IO"].get("logName", "oil_distribution.mp4"))
        final_plot_file = os.path.join(output_dir, "final_oil_distribution.png")
        restart_file = config["IO"].get("restartFile", None)

        if args.load:
            simulation.load_state(restart_file or os.path.join(output_dir, "simulation_state.txt"))

        simulation.run(animation_file, final_plot_file, restart_file)

        if args.save:
            # Save intermediate state for potential restart
            intermediate_state_file = os.path.join(output_dir, "simulation_state.txt")
            simulation.save_state(intermediate_state_file)
            print(f"Simulation state saved to {intermediate_state_file}")


if __name__ == "__main__":
    main()
