import meshio
import numpy as np
import pytest

# Import from the installed package
from oil_spill_simulation.mesh import Mesh


@pytest.fixture
def mock_mesh():
    # Mock mesh object with points and cells_dict attributes
    class MockMesh:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        cells_dict = {
            "triangle": np.array([[0, 1, 2], [1, 2, 3], [3, 2, 1], [3, 2, 4]]),
            "line": np.array([[0, 1], [1, 2], [2, 3]]),
        }

    return MockMesh()


@pytest.fixture
def mesh_instance(monkeypatch, mock_mesh):
    # Mock the meshio.read function to return the mock mesh
    def mock_read(file):
        return mock_mesh

    monkeypatch.setattr(meshio, "read", mock_read)

    # Initialize Mesh with a dummy file
    return Mesh("dummy_file.msh")


def test_mesh_initialization(mesh_instance, mock_mesh):
    # Check if the points, triangles, and lines are correctly initialized
    np.testing.assert_array_equal(mesh_instance.points, mock_mesh.points[:, :2])
    np.testing.assert_array_equal(mesh_instance.triangles, mock_mesh.cells_dict["triangle"])
    np.testing.assert_array_equal(mesh_instance.lines, mock_mesh.cells_dict["line"])


def test_find_faces_with_node(mesh_instance):
    # Test finding faces that contain a specific node
    faces_with_node_0 = mesh_instance.find_faces_with_node(0)
    # Node 0 should be in triangle 0 and line 0 (which is cell index 4)
    assert 0 in faces_with_node_0
    assert 4 in faces_with_node_0  # Line [0,1] is at index 4 in combined cells


def test_neighbors_triangles(mesh_instance):
    # Test that neighbors are computed correctly
    neighbors = mesh_instance.neighbors_triangles
    assert len(neighbors) == len(mesh_instance.triangles)

    # Each neighbor list should have 3 elements (for triangle neighbors)
    for neighbor_list in neighbors:
        assert len(neighbor_list) == 3


def test_compute_midpoints_triangles(mesh_instance):
    # Test midpoint computation for triangles
    midpoints = mesh_instance.compute_midpoints(mesh_instance.triangles)

    # Should have one midpoint per triangle
    assert len(midpoints) == len(mesh_instance.triangles)

    # Each midpoint should be 2D
    assert midpoints.shape[1] == 2

    # Test first triangle midpoint calculation
    first_triangle = mesh_instance.triangles[0]  # [0, 1, 2]
    expected_midpoint = (mesh_instance.points[0] + mesh_instance.points[1] + mesh_instance.points[2]) / 3
    np.testing.assert_array_almost_equal(midpoints[0], expected_midpoint)


def test_compute_midpoints_lines(mesh_instance):
    # Test midpoint computation for lines
    midpoints = mesh_instance.compute_midpoints(mesh_instance.lines, is_line=True)

    # Should have one midpoint per line
    assert len(midpoints) == len(mesh_instance.lines)

    # Each midpoint should be 2D
    assert midpoints.shape[1] == 2

    # Test first line midpoint calculation
    first_line = mesh_instance.lines[0]  # [0, 1]
    expected_midpoint = (mesh_instance.points[0] + mesh_instance.points[1]) / 2
    np.testing.assert_array_almost_equal(midpoints[0], expected_midpoint)


def test_find_neighbor_faces_by_edge(mesh_instance):
    # Test finding neighbors by edge for a specific triangle
    neighbors = mesh_instance.find_neighbor_faces_by_edge(0)

    # Should return list of 3 neighbors (one for each edge)
    assert len(neighbors) == 3

    # Neighbors should be integers or -1 (for boundary)
    for neighbor in neighbors:
        assert isinstance(neighbor, (int, np.integer)) or neighbor == -1


def test_cached_faces(mesh_instance):
    # Test that cached faces are populated correctly
    assert hasattr(mesh_instance, "cached_faces")

    # Should have entries for each node
    for node_id in range(len(mesh_instance.points)):
        faces = mesh_instance.find_faces_with_node(node_id)
        assert isinstance(faces, list)


if __name__ == "__main__":
    pytest.main()
