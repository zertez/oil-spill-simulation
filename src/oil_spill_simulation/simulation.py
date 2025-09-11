import numpy as np

from .geometry import Geometry


class OilCalculation:
    def __init__(self, delta_t):
        self.delta_t = delta_t

    def oil_distribution(self, x, y, xk=0.8, yk=0.55, sigma=0.01):
        x_vec = np.array([x, y])
        xk_vec = np.array([xk, yk])
        return np.exp(-(np.linalg.norm(x_vec - xk_vec) ** 2) / sigma)

    def velocity_field(self, x, y):
        return np.array([y - 0.2 * x, -x])

    def g(self, a, b, v, v_prime):
        """
        Flux function that determines oil transport across cell interfaces.

        This implements the FLUX function from the finite volume method:
        - When dot(v, v') > 0: flow is outward from current cell (upwind scheme)
        - When dot(v, v') <= 0: flow is inward to current cell (upwind scheme)

        Args:
            a: Oil concentration in current cell (upwind value)
            b: Oil concentration in neighbor cell (upwind value)
            v: Scaled outward normal vector at interface
            v_prime: Average velocity at interface

        Returns:
            Flux value for oil transport
        """
        dot_product_v = np.dot(v, v_prime)
        return a * dot_product_v if dot_product_v > 0 else b * dot_product_v

    def oil_flux(self, area, u_i, u_ngh, v_i_l, v_mid):
        return -self.delta_t / area * self.g(u_i, u_ngh, v_i_l, v_mid)

    def compute_initial_oil_distribution(self, midpoints):
        xk_vec = np.array([0.35, 0.45])
        distances = np.linalg.norm(midpoints - xk_vec, axis=1)
        return np.exp(-(distances**2) / 0.01)

    def compute_velocities(self, midpoints):
        x, y = midpoints[:, 0], midpoints[:, 1]
        return np.column_stack((y - 0.2 * x, -x))

    def compute_scaled_vectors(self, points, cells, midpoints):
        scaled_vectors = np.zeros((len(cells), 3, 2))
        for i, cell in enumerate(cells):
            if len(cell) != 3:
                continue
            midpoint = midpoints[i]
            n1, n2, n3, e1, e2, e3 = Geometry.compute_normals(
                points[cell[0]], points[cell[1]], points[cell[2]], midpoint
            )
            scaled_vectors[i] = [
                Geometry.scaled_normal_vector(n1, e1),
                Geometry.scaled_normal_vector(n2, e2),
                Geometry.scaled_normal_vector(n3, e3),
            ]
        return scaled_vectors

    def oil_updater(self, oil_distribution_values, velocities, triangles, neighbors, areas, scaled_vectors):
        u_new = oil_distribution_values.copy()

        # Only update triangles, not lines (boundary elements remain fixed)
        for i in range(len(triangles)):
            u_i = oil_distribution_values[i]
            flux_sum = 0
            for m, ngh in enumerate(neighbors[i]):
                if ngh == -1:  # Boundary condition - skip boundary edges
                    continue
                v_i_l = scaled_vectors[i][m]
                u_ngh = oil_distribution_values[ngh]
                v_mid = 0.5 * (velocities[i] + velocities[ngh])
                flux = self.oil_flux(areas[i], u_i, u_ngh, v_i_l, v_mid)
                flux_sum += flux
            u_new[i] = u_i + flux_sum

        # Lines (boundary elements) keep their initial values unchanged
        return u_new

    def save_state(self, filename, oil_distribution, current_time, midpoints):
        """Save simulation state to text file with metadata."""
        header = f"Time: {current_time}\nColumns: x, y, oil_concentration"
        data = np.column_stack((midpoints, oil_distribution))
        np.savetxt(filename, data, header=header, fmt="%.8f")

    def load_state(self, filename):
        """Load simulation state from text file."""
        try:
            data = np.loadtxt(filename)
            if data.ndim == 1:
                # Single point case - ensure consistent array shapes
                oil_distribution = np.array([data[2]])  # Make it a 1D array
                midpoints = data[:2].reshape(1, -1)  # Make it a 2D array (1, 2)
                return oil_distribution, midpoints
            else:
                # Multiple points case
                oil_distribution = data[:, 2]  # Third column
                midpoints = data[:, :2]  # First two columns
                return oil_distribution, midpoints
        except Exception as e:
            raise ValueError(f"Failed to load state from {filename}: {str(e)}")

    def get_restart_time(self, filename):
        """Extract the time from the saved state file."""
        try:
            with open(filename, "r") as f:
                first_line = f.readline().strip()
                if first_line.startswith("# Time: "):
                    return float(first_line.split("Time: ")[1])
                else:
                    raise ValueError("Time information not found in restart file")
        except Exception as e:
            raise ValueError(f"Failed to read time from {filename}: {str(e)}")

    def calculate_oil_in_fishing_grounds(self, oil_distribution, midpoints, borders):
        """Calculate total oil amount within fishing ground boundaries."""
        total_oil = 0.0

        # Handle empty borders
        if not borders:
            return total_oil

        # Handle single area format: [[x_min, x_max], [y_min, y_max]]
        if len(borders) == 2 and all(isinstance(border, list) and len(border) == 2 for border in borders):
            # Single fishing area
            x_min, x_max = borders[0]
            y_min, y_max = borders[1]

            for i, (x, y) in enumerate(midpoints):
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    total_oil += oil_distribution[i]
        else:
            # Multiple fishing areas format: [[[x_min, x_max], [y_min, y_max]], ...]
            for area in borders:
                x_min, x_max = area[0]
                y_min, y_max = area[1]

                for i, (x, y) in enumerate(midpoints):
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        total_oil += oil_distribution[i]

        return total_oil
