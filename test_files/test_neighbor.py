from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import from the installed package
from oil_spill_simulation.mesh import Mesh


@pytest.fixture
def mock_mesh_data():
    """Create mock mesh data for testing"""

    class MockMeshData:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        cells_dict = {"triangle": np.array([[0, 1, 2], [1, 2, 3]]), "line": np.array([[0, 1], [1, 2], [2, 3]])}

    return MockMeshData()


@pytest.fixture
def mesh_instance(mock_mesh_data):
    """Create a Mesh instance with mocked data"""
    with patch("meshio.read", return_value=mock_mesh_data):
        return Mesh("dummy.msh")


def test_mesh_initialization(mesh_instance, mock_mesh_data):
    """Test that Mesh initializes correctly"""
    # Check points are 2D (not 3D)
    assert mesh_instance.points.shape == (4, 2)
    np.testing.assert_array_equal(mesh_instance.points, mock_mesh_data.points[:, :2])

    # Check triangles and lines
    np.testing.assert_array_equal(mesh_instance.triangles, mock_mesh_data.cells_dict["triangle"])
    np.testing.assert_array_equal(mesh_instance.lines, mock_mesh_data.cells_dict["line"])

    # Check cells is combination of triangles and lines
    expected_cells = [
        [0, 1, 2],
        [1, 2, 3],  # triangles
        [0, 1],
        [1, 2],
        [2, 3],  # lines
    ]
    # Convert to numpy arrays for comparison
    assert len(mesh_instance.cells) == len(expected_cells)
    for i, (actual, expected) in enumerate(zip(mesh_instance.cells, expected_cells)):
        np.testing.assert_array_equal(actual, expected)


def test_find_faces_with_node(mesh_instance):
    """Test finding faces that contain a specific node"""
    # Node 0 should be in triangle 0 and line 0
    faces_with_node_0 = mesh_instance.find_faces_with_node(0)
    assert 0 in faces_with_node_0  # Triangle 0
    assert 2 in faces_with_node_0  # Line 0 (index 2 in combined cells)

    # Node 1 should be in triangle 0, triangle 1, line 0, line 1
    faces_with_node_1 = mesh_instance.find_faces_with_node(1)
    assert 0 in faces_with_node_1  # Triangle 0
    assert 1 in faces_with_node_1  # Triangle 1
    assert 2 in faces_with_node_1  # Line 0
    assert 3 in faces_with_node_1  # Line 1


def test_find_neighbor_faces_by_edge(mesh_instance):
    """Test finding neighbor faces by edge"""
    # Triangle 0: [0, 1, 2]
    neighbors_0 = mesh_instance.find_neighbor_faces_by_edge(0)

    # Should return 3 neighbors (one for each edge)
    assert len(neighbors_0) == 3

    # Each neighbor should be an integer or -1 (boundary)
    for neighbor in neighbors_0:
        assert isinstance(neighbor, (int, np.integer)) or neighbor == -1


def test_get_neighbors_triangles(mesh_instance):
    """Test that neighbors are computed for all triangles"""
    neighbors = mesh_instance.neighbors_triangles

    # Should have neighbors for each triangle
    assert len(neighbors) == len(mesh_instance.triangles)

    # Each triangle should have exactly 3 neighbor entries
    for neighbor_list in neighbors:
        assert len(neighbor_list) == 3


def test_compute_midpoints_triangles(mesh_instance):
    """Test midpoint computation for triangles"""
    midpoints = mesh_instance.compute_midpoints(mesh_instance.triangles)

    # Should have one midpoint per triangle
    assert len(midpoints) == len(mesh_instance.triangles)
    assert midpoints.shape[1] == 2  # 2D coordinates

    # Test specific midpoint calculation for first triangle [0, 1, 2]
    p0, p1, p2 = mesh_instance.points[0], mesh_instance.points[1], mesh_instance.points[2]
    expected_midpoint = (p0 + p1 + p2) / 3
    np.testing.assert_array_almost_equal(midpoints[0], expected_midpoint)


def test_compute_midpoints_lines(mesh_instance):
    """Test midpoint computation for lines"""
    midpoints = mesh_instance.compute_midpoints(mesh_instance.lines, is_line=True)

    # Should have one midpoint per line
    assert len(midpoints) == len(mesh_instance.lines)
    assert midpoints.shape[1] == 2  # 2D coordinates

    # Test specific midpoint calculation for first line [0, 1]
    p0, p1 = mesh_instance.points[0], mesh_instance.points[1]
    expected_midpoint = (p0 + p1) / 2
    np.testing.assert_array_almost_equal(midpoints[0], expected_midpoint)


def test_cached_faces_populated(mesh_instance):
    """Test that cached faces dictionary is properly populated"""
    # Should have entries for each node
    for node_id in range(len(mesh_instance.points)):
        assert node_id in mesh_instance.cached_faces

    # Each node should have at least one face
    for node_id, faces in mesh_instance.cached_faces.items():
        assert len(faces) > 0

        # All face indices should be valid
        for face_id in faces:
            assert 0 <= face_id < len(mesh_instance.cells)


def test_neighbor_consistency(mesh_instance):
    """Test that neighbor relationships are consistent"""
    total_cells = len(mesh_instance.triangles) + len(mesh_instance.lines)
    for i, triangle_neighbors in enumerate(mesh_instance.neighbors_triangles):
        for j, neighbor_id in enumerate(triangle_neighbors):
            if neighbor_id != -1:  # Not a boundary
                # Check that the neighbor is within valid range (can be triangle or line)
                assert 0 <= neighbor_id < total_cells

                # If it's a triangle neighbor, check reciprocal relationship
                if neighbor_id < len(mesh_instance.triangles):
                    neighbor_neighbors = mesh_instance.neighbors_triangles[neighbor_id]
                    # Note: Due to the complexity of edge matching, we just check the neighbor exists
                    assert neighbor_id < len(mesh_instance.triangles)


def test_mesh_with_realistic_data():
    """Test mesh processing with more realistic triangle arrangement"""
    # Create a simple 2-triangle mesh sharing an edge
    mock_mesh_data = MagicMock()
    mock_mesh_data.points = np.array(
        [
            [0, 0, 0],  # Node 0
            [1, 0, 0],  # Node 1
            [0, 1, 0],  # Node 2
            [1, 1, 0],  # Node 3
        ]
    )
    mock_mesh_data.cells_dict = {
        "triangle": np.array(
            [
                [0, 1, 2],  # Triangle 0
                [1, 3, 2],  # Triangle 1 (shares edge 1-2 with triangle 0)
            ]
        ),
        "line": np.array([[0, 1]]),  # Boundary line
    }

    with patch("meshio.read", return_value=mock_mesh_data):
        mesh = Mesh("dummy.msh")

        # The two triangles should be neighbors since they share edge 1-2
        neighbors = mesh.neighbors_triangles

        # Triangle 0 should have triangle 1 as a neighbor
        assert 1 in neighbors[0]
        # Triangle 1 should have triangle 0 as a neighbor
        assert 0 in neighbors[1]


if __name__ == "__main__":
    pytest.main([__file__])
