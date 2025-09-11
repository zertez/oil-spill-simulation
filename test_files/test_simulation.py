import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import from the installed package
from oil_spill_simulation.config import TOMLFileReader
from oil_spill_simulation.main import Simulation, ensure_directory_exists, list_config_files
from oil_spill_simulation.mesh import Mesh
from oil_spill_simulation.simulation import OilCalculation


@pytest.fixture
def mock_config():
    return {
        "geometry": {"meshName": "dummy_mesh.msh", "borders": [[0.0, 0.75], [0.0, 0.2]]},
        "settings": {"tStart": 0.0, "tEnd": 0.5, "nSteps": 100, "writeFrequency": 10},
        "IO": {"logName": "oil_distribution.mp4", "restartFile": None},
    }


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_oil_calculation_initialization():
    """Test OilCalculation class initialization"""
    delta_t = 0.01
    oil_calc = OilCalculation(delta_t)
    assert oil_calc.delta_t == delta_t


def test_oil_distribution_computation():
    """Test initial oil distribution computation"""
    oil_calc = OilCalculation(0.01)
    midpoints = np.array([[0.35, 0.45], [0.5, 0.5], [0.1, 0.1]])

    oil_dist = oil_calc.compute_initial_oil_distribution(midpoints)

    # Should have same length as midpoints
    assert len(oil_dist) == len(midpoints)

    # First point is close to source (0.35, 0.45), should have high concentration
    assert oil_dist[0] > oil_dist[1] > oil_dist[2]


def test_velocity_computation():
    """Test velocity field computation"""
    oil_calc = OilCalculation(0.01)
    midpoints = np.array([[0.1, 0.2], [0.3, 0.4]])

    velocities = oil_calc.compute_velocities(midpoints)

    # Should have same number of velocity vectors as midpoints
    assert velocities.shape == (2, 2)

    # Test specific values: v = [y - 0.2*x, -x]
    expected_v1 = np.array([0.2 - 0.2 * 0.1, -0.1])  # [0.18, -0.1]
    expected_v2 = np.array([0.4 - 0.2 * 0.3, -0.3])  # [0.34, -0.3]

    np.testing.assert_array_almost_equal(velocities[0], expected_v1)
    np.testing.assert_array_almost_equal(velocities[1], expected_v2)


def test_save_load_state():
    """Test save and load state functionality"""
    oil_calc = OilCalculation(0.01)

    # Test data
    midpoints = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    oil_distribution = np.array([0.8, 0.6, 0.4])
    current_time = 0.25

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filename = f.name

    try:
        # Save state
        oil_calc.save_state(filename, oil_distribution, current_time, midpoints)
        assert os.path.exists(filename)

        # Load state
        loaded_oil, loaded_midpoints = oil_calc.load_state(filename)

        # Verify data integrity
        np.testing.assert_array_almost_equal(loaded_oil, oil_distribution)
        np.testing.assert_array_almost_equal(loaded_midpoints, midpoints)

        # Test restart time
        restart_time = oil_calc.get_restart_time(filename)
        assert restart_time == current_time

    finally:
        if os.path.exists(filename):
            os.unlink(filename)


def test_calculate_oil_in_fishing_grounds():
    """Test oil calculation within fishing ground boundaries"""
    oil_calc = OilCalculation(0.01)

    # Test data - points at different locations
    midpoints = np.array([[0.1, 0.1], [0.5, 0.1], [0.1, 0.5], [0.9, 0.9]])
    oil_distribution = np.array([1.0, 2.0, 3.0, 4.0])

    # Fishing grounds: x=[0.0, 0.75], y=[0.0, 0.2]
    borders = [[0.0, 0.75], [0.0, 0.2]]

    oil_amount = oil_calc.calculate_oil_in_fishing_grounds(oil_distribution, midpoints, borders)

    # Only points (0.1, 0.1) and (0.5, 0.1) should be within bounds
    expected_oil = 1.0 + 2.0  # Sum of oil at valid points
    assert abs(oil_amount - expected_oil) < 1e-6


def test_flux_function():
    """Test the flux function g()"""
    oil_calc = OilCalculation(0.01)

    # Test outward flow (dot product > 0)
    a, b = 1.0, 0.5
    v = np.array([1, 0])
    v_prime = np.array([1, 0])  # Same direction
    result = oil_calc.g(a, b, v, v_prime)
    expected = a * np.dot(v, v_prime)  # Should use upwind value a
    assert result == expected

    # Test inward flow (dot product < 0)
    v_prime = np.array([-1, 0])  # Opposite direction
    result = oil_calc.g(a, b, v, v_prime)
    expected = b * np.dot(v, v_prime)  # Should use upwind value b
    assert result == expected


@patch("oil_spill_simulation.mesh.meshio.read")
def test_mesh_initialization(mock_meshio_read):
    """Test Mesh class initialization"""
    # Mock mesh data
    mock_mesh_data = MagicMock()
    mock_mesh_data.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_mesh_data.cells_dict = {"triangle": np.array([[0, 1, 2]]), "line": np.array([[0, 1]])}
    mock_meshio_read.return_value = mock_mesh_data

    mesh = Mesh("dummy.msh")

    # Verify initialization
    assert mesh.points.shape == (3, 2)  # Should be 2D points
    assert len(mesh.triangles) == 1
    assert len(mesh.lines) == 1
    assert len(mesh.neighbors_triangles) == 1


def test_list_config_files(temp_output_dir):
    """Test listing TOML configuration files"""
    # Create test TOML files
    config1 = os.path.join(temp_output_dir, "config1.toml")
    config2 = os.path.join(temp_output_dir, "config2.toml")
    pyproject = os.path.join(temp_output_dir, "pyproject.toml")
    other_file = os.path.join(temp_output_dir, "other.txt")

    for file_path in [config1, config2, pyproject, other_file]:
        with open(file_path, "w") as f:
            f.write("dummy content")

    config_files = list_config_files(temp_output_dir)

    # Should find config1.toml and config2.toml, but NOT pyproject.toml
    assert len(config_files) == 2
    assert config1 in config_files
    assert config2 in config_files
    assert pyproject not in config_files
    assert other_file not in config_files


def test_ensure_directory_exists(temp_output_dir):
    """Test directory creation and validation"""
    # Test creating new directory
    new_dir = os.path.join(temp_output_dir, "new_directory")
    ensure_directory_exists(new_dir)
    assert os.path.exists(new_dir)
    assert os.path.isdir(new_dir)

    # Test existing directory with simulation results
    existing_dir = os.path.join(temp_output_dir, "existing_with_results")
    os.makedirs(existing_dir)

    # Create a simulation log file to make it look like a results folder
    log_file = os.path.join(existing_dir, "simulation_summary.log")
    with open(log_file, "w") as f:
        f.write("dummy log")

    # Should not raise error
    ensure_directory_exists(existing_dir)

    # Test existing directory without simulation results
    existing_dir_no_results = os.path.join(temp_output_dir, "existing_no_results")
    os.makedirs(existing_dir_no_results)

    # Should raise error
    with pytest.raises(Exception) as exc_info:
        ensure_directory_exists(existing_dir_no_results)
    assert "already exists and may not be a simulation results folder" in str(exc_info.value)


@patch("oil_spill_simulation.main.Mesh")
@patch("oil_spill_simulation.main.setup_logging")
def test_simulation_initialization(mock_logging, mock_mesh_class, mock_config, temp_output_dir):
    """Test Simulation class initialization"""
    # Mock mesh instance
    mock_mesh = MagicMock()
    mock_mesh.triangles = np.array([[0, 1, 2]])
    mock_mesh.lines = np.array([[0, 1]])
    mock_mesh.points = np.array([[0, 0], [1, 0], [0, 1]])
    mock_mesh.cells = [[0, 1, 2], [0, 1]]
    mock_mesh.neighbors_triangles = [[-1, -1, -1]]
    mock_mesh.compute_midpoints.return_value = np.array([[1 / 3, 1 / 3]])

    mock_mesh_class.return_value = mock_mesh

    # Mock logger
    mock_logger = MagicMock()
    mock_logging.return_value = mock_logger

    # Create simulation
    simulation = Simulation(mock_config, temp_output_dir)

    # Verify initialization
    assert simulation.output_dir == temp_output_dir
    assert simulation.n_steps == 100
    assert simulation.delta_t == 0.005  # (0.5 - 0.0) / 100
    assert hasattr(simulation, "oil_calc")
    assert hasattr(simulation, "mesh")


def test_toml_file_reader():
    """Test TOML file reader functionality"""
    # Create temporary TOML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[settings]
nSteps = 100
tStart = 0.0
tEnd = 1.0

[geometry]
meshName = "test.msh"
borders = [[0.0, 1.0], [0.0, 1.0]]

[IO]
logName = "test.mp4"
""")
        filename = f.name

    try:
        reader = TOMLFileReader(filename)
        config = reader.get_config()

        # Verify config structure
        assert "settings" in config
        assert "geometry" in config
        assert "IO" in config

        # Verify values
        assert config["settings"]["nSteps"] == 100
        assert config["geometry"]["meshName"] == "test.msh"
        assert config["IO"]["logName"] == "test.mp4"

    finally:
        os.unlink(filename)


def test_toml_validation_missing_section():
    """Test TOML validation catches missing required sections"""
    # Create TOML file missing required section
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[settings]
nSteps = 100
tEnd = 1.0
# Missing [geometry] section
""")
        filename = f.name

    try:
        reader = TOMLFileReader(filename)
        with pytest.raises(ValueError) as exc_info:
            reader.get_config()
        assert "Missing required section: geometry" in str(exc_info.value)

    finally:
        os.unlink(filename)


if __name__ == "__main__":
    pytest.main([__file__])
