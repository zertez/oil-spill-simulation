# Oil Spill Simulation - INF203 Project

*University Project - NMBU, June 2024*

A comprehensive finite volume method simulation for modeling oil spill dispersion in coastal waters, featuring advanced restart capabilities, real-time oil tracking, and extensive logging.

## Features

- **Finite Volume Method Implementation** - Accurate numerical simulation using flux calculations
- **TOML Configuration System** - Flexible parameter management with validation
- **Video Animation Generation** - Create MP4 animations of oil dispersion over time
- **Text File State Management** - Save and restart simulations from any point
- **Fishing Ground Monitoring** - Real-time tracking of oil amounts in specified areas
- **Multiple Simulation Support** - Batch processing of configuration files
- **Comprehensive Logging** - Detailed simulation summaries and parameter tracking
- **Command Line Interface** - Easy-to-use CLI with flexible options
- **Robust Testing Suite** - Comprehensive test coverage for all components

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd oil_spill_sim
```

2. Install dependencies:
```bash
pip install -e .
# or
pip install matplotlib numpy meshio toml tqdm
```

### Basic Usage

Run a simulation with the default configuration:
```bash
python -m src.oil_spill_sim.main
```

Run with a specific configuration file:
```bash
python -m src.oil_spill_sim.main --config-file config_files/my_config.toml
```

Run multiple simulations from a directory:
```bash
python -m src.oil_spill_sim.main --config-dir config_files/
```

## Configuration

### TOML Configuration Structure

```toml
[settings]
nSteps = 500      # Number of time steps
tStart = 0.1      # Start time (use > 0 for restart)
tEnd = 0.5        # End time

[geometry]
meshName = "bay.msh"                        # Mesh file path
borders = [[0.0, 0.75], [0.0, 0.2]]       # Fishing ground boundaries [x_range, y_range]

[IO]
logName = "simulation_log"                  # Log file name
writeFrequency = 10                         # Video frame frequency
restartFile = ""                           # Path to restart file (optional)
```

### Configuration Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `nSteps` | integer | Number of simulation time steps |
| `tStart` | float | ❌ | Start time (default: 0.0) |
| `tEnd` | float | End time of simulation |
| `meshName` | string | Path to mesh file (.msh format) |
| `borders` | array | Fishing ground boundaries as [[x_min, x_max], [y_min, y_max]] |
| `logName` | string | ❌ | Log file prefix (default: "logfile") |
| `writeFrequency` | integer | ❌ | Video output frequency (frames per step) |
| `restartFile` | string | ❌ | Path to restart state file |

## Restart Functionality

The simulation supports saving and loading states for continuation:

### Automatic State Saving
- States are automatically saved every 100 simulation steps
- Final state is always saved as `final_state.txt`
- State files contain: time, coordinates, and oil concentrations

### Restart Configuration
To restart from a saved state:

```toml
[settings]
tStart = 0.25                    # Must match time in restart file
tEnd = 0.75
nSteps = 250

[IO]
restartFile = "state_step_100.txt"  # Path to saved state
```

**Important:** When using `restartFile`, you must provide `tStart > 0` matching the saved time.

## Oil Tracking & Logging

### Fishing Ground Monitoring
The simulation tracks oil amounts within specified fishing grounds:

```toml
[geometry]
borders = [
    [0.0, 0.4], [0.1, 0.3],    # First fishing area
    [0.6, 1.0], [0.7, 0.9]     # Second fishing area
]
```

### Logging Output
The simulation provides comprehensive logging:

```
=== SIMULATION PARAMETERS ===
[settings]
  nSteps = 500
  tStart = 0.1
  tEnd = 0.5
[geometry]
  meshName = bay.msh
  borders = [[0.0, 0.75], [0.0, 0.2]]
=== END PARAMETERS ===

Time 0.1000: Oil in fishing grounds = 0.234567
Time 0.1500: Oil in fishing grounds = 0.189432
Time 0.2000: Oil in fishing grounds = 0.156789
...
```

## Command Line Options

```bash
python -m src.oil_spill_sim.main [OPTIONS]

Options:
  --config-file PATH    Path to single TOML configuration file
  --config-dir PATH     Directory containing multiple TOML files
  --output-dir PATH     Base directory for results (default: results)
  -h, --help           Show help message
```

### Usage Examples

```bash
# Single simulation with default config
python -m src.oil_spill_sim.main

# Single simulation with custom config
python -m src.oil_spill_sim.main --config-file my_simulation.toml

# Batch processing
python -m src.oil_spill_sim.main --config-dir batch_configs/

# Custom output location
python -m src.oil_spill_sim.main --config-file sim.toml --output-dir my_results/
```

## Project Structure

```

```

## Testing

Run the test suite:

```bash
python test_files/test_save_load.py
python test_files/test_geometry.py
python test_files/test_mesh.py
python test_files/test_simulation.py
```

## Output Files

### Generated Files
- **Video Animation**: `{config_name}.mp4` - Visual simulation of oil dispersion
- **State Files**: `state_step_*.txt` - Periodic simulation states for restart
- **Final State**: `final_state.txt` - Final simulation state
- **Initial Plot**: `Oil_Time_0.png` - Initial oil distribution visualization

### State File Format
```
# Time: 0.250000
# Columns: x, y, oil_concentration
0.12345678 0.23456789 0.87654321
0.34567890 0.45678901 0.76543210
...
```

## Mathematical Model

The simulation implements the finite volume method for the advection equation:

```
∂u/∂t + ∇·(v⃗u) = 0
```

Where:
- `u(t,x⃗)` = oil concentration at position x⃗ and time t
- `v⃗(x⃗) = (y - 0.2x, -x)` = velocity field
- Initial condition: `u(0,x⃗) = exp(-||x⃗ - x⃗ₖ||²/0.01)` with x⃗ₖ = (0.35, 0.45)

## Academic Context

This project was developed for **INF203** at NMBU (Norwegian University of Life Sciences) in June 2024. It demonstrates:

- Numerical methods for partial differential equations
- Software engineering best practices
- Object-oriented programming in Python
- Scientific computing and visualization
- Configuration management and testing

## Troubleshooting

### Common Issues

**Configuration Errors:**
```
ValueError: Missing required section: settings
```
- Ensure all required TOML sections are present

**Restart Errors:**
```
ValueError: When restartFile is provided, tStart must be provided and greater than 0
```
- Set `tStart` to match the time in your restart file

**Mesh File Errors:**
```
FileNotFoundError: Mesh file not found: bay.msh
```
- Ensure mesh file path is correct relative to working directory

### Debug Mode
Enable detailed logging by modifying the logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## License

This project is developed for educational purposes as part of university coursework.

## Authors

- Marcus Dalaker Figenschou - NMBU INF203 Student

---

*For questions about the implementation or usage, please refer to the source code documentation or the test files for examples.*
