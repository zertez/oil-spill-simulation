
import numpy as np
import pytest

# Import from the installed package
from oil_spill_simulation.geometry import Geometry


def test_vector_subtraction():
    p1 = [2, 3]
    p2 = [1, 1]
    result = Geometry.vector_subtraction(p1, p2)
    expected = np.array([1, 2])
    np.testing.assert_array_equal(result, expected)


def test_scaled_normal_vector():
    normal = np.array([0, 1])
    edge = np.array([3, 4])
    result = Geometry.scaled_normal_vector(normal, edge)
    expected = np.array([0, 5])  # Length of edge is 5
    np.testing.assert_array_equal(result, expected)


def test_triangle_area():
    p1 = [0, 0]
    p2 = [1, 0]
    p3 = [0, 1]
    result = Geometry.triangle_area(p1, p2, p3)
    expected = 0.5
    assert result == expected


def test_compute_normals():
    p1 = [0, 0]
    p2 = [1, 0]
    p3 = [0, 1]
    midpoint = np.array([1 / 3, 1 / 3])

    n1, n2, n3, e1, e2, e3 = Geometry.compute_normals(p1, p2, p3, midpoint)

    expected_e1 = np.array([1, 0])
    expected_e2 = np.array([-1, 1])
    expected_e3 = np.array([0, -1])

    # Calculate expected normals based on edge directions
    expected_n1 = np.array([0, 1])
    expected_n2 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
    expected_n3 = np.array([-1, 0])

    np.testing.assert_array_almost_equal(e1, expected_e1)
    np.testing.assert_array_almost_equal(e2, expected_e2)
    np.testing.assert_array_almost_equal(e3, expected_e3)

    # Ensure the normals are pointing away from the midpoint
    assert np.dot(Geometry.vector_subtraction(p1, midpoint), n1) >= 0
    assert np.dot(Geometry.vector_subtraction(p2, midpoint), n2) >= 0
    assert np.dot(Geometry.vector_subtraction(p3, midpoint), n3) >= 0


if __name__ == "__main__":
    pytest.main()
