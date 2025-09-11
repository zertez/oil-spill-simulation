import numpy as np


class Geometry:
    @staticmethod
    def vector_subtraction(p1, p2):
        return np.array(p1) - np.array(p2)

    @staticmethod
    def triangle_area(p1, p2, p3):
        # Using NumPy arrays for better performance
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        return 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))

    @staticmethod
    def compute_normals(p1, p2, p3, midpoint):
        p1, p2, p3, midpoint = np.array(p1), np.array(p2), np.array(p3), np.array(midpoint)
        e1 = p2 - p1
        e2 = p3 - p2
        e3 = p1 - p3

        n1 = np.array([-e1[1], e1[0]]) / np.linalg.norm(e1)
        n2 = np.array([-e2[1], e2[0]]) / np.linalg.norm(e2)
        n3 = np.array([-e3[1], e3[0]]) / np.linalg.norm(e3)

        if np.dot(p1 - midpoint, n1) < 0:
            n1 = -n1
        if np.dot(p2 - midpoint, n2) < 0:
            n2 = -n2
        if np.dot(p3 - midpoint, n3) < 0:
            n3 = -n3

        return n1, n2, n3, e1, e2, e3

    @staticmethod
    def scaled_normal_vector(normal, edge):
        return normal * np.linalg.norm(edge)
