from collections import defaultdict

import meshio
import numpy as np


class Mesh:
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.points, self.triangles, self.lines, self.cells, self.cached_faces = self.load_mesh()
        self.neighbors_triangles = self.get_neighbors_triangles()

    def load_mesh(self):
        mesh = meshio.read(self.mesh_file)
        points = mesh.points[:, :2]
        triangles = mesh.cells_dict["triangle"]
        lines = mesh.cells_dict["line"]
        cells = list(triangles) + list(lines)

        cached_faces = defaultdict(list)
        for i in range(len(cells)):
            for node in cells[i]:
                cached_faces[node].append(i)

        return points, triangles, lines, cells, cached_faces

    def find_faces_with_node(self, index):
        return self.cached_faces[index]

    def find_neighbor_faces_by_edge(self, index):
        face = self.triangles[index]
        a = set(f for f in self.find_faces_with_node(face[0]))
        a.remove(index)
        b = set(f for f in self.find_faces_with_node(face[1]))
        b.remove(index)
        c = set(f for f in self.find_faces_with_node(face[2]))
        c.remove(index)

        # Boundary edge indicator
        neighbors = []
        for intersection in [a.intersection(b), b.intersection(c), a.intersection(c)]:
            if intersection:
                neighbors.append(list(intersection)[0])
            else:
                neighbors.append(-1)
        return neighbors

    def get_neighbors_triangles(self):
        return [self.find_neighbor_faces_by_edge(i) for i in range(len(self.triangles))]

    def compute_midpoints(self, elements, is_line=False):
        if is_line:
            return np.array([(self.points[line[0]] + self.points[line[1]]) / 2 for line in elements])
        return np.array([(self.points[tri[0]] + self.points[tri[1]] + self.points[tri[2]]) / 3 for tri in elements])
