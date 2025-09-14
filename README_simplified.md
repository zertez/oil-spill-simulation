# Oil Spill Simulation - INF203 Project

*University Project - NMBU, June 2024*

A finite volume method simulation for modeling oil spill dispersion in coastal waters with restart capabilities and real-time oil tracking.

![Oil Spill Simulation](simulation_preview.png)

## Quick Start

### Installation
```bash
git clone <your-repo-url>
cd oil_spill_sim
pip install -e .
```

### Run Simulation
```bash
# Default simulation
oil-spill-sim

# With custom config
oil-spill-sim --config-file config_files/input.toml
```

## Features

- **Finite Volume Method** - Numerical solution of advection equation
- **TOML Configuration** - Easy parameter management
- **Video Output** - MP4 animations of oil dispersion
- **Restart Capability** - Save/load simulation states
- **Fishing Ground Tracking** - Monitor oil in specified areas
- **Command Line Interface** - Simple usage with flexible options

## Configuration Example

```toml
[settings]
nSteps = 500
tEnd = 0.5

[geometry]
meshName = "bay.msh"
borders = [[0.0, 0.75], [0.0, 0.2]]

[IO]
logName = "simulation_log"
```

## Mathematical Model

Solves the advection equation using finite volume method:
```
∂u/∂t + ∇·(v⃗u) = 0
```

Where:
- `u(t,x⃗)` = oil concentration
- `v⃗(x⃗) = (y - 0.2x, -x)` = velocity field
- Initial condition: Gaussian distribution at `(0.35, 0.45)`

## Testing

```bash
pytest test_files/
```

## Output Files

- `{config_name}.mp4` - Animation video
- `state_step_*.txt` - Periodic simulation states
- `Oil_Time_0.png` - Initial distribution plot

## Project Structure

```
oil_spill_sim/
├── src/oil_spill_simulation/    # Main package
├── test_files/                  # Test suite
├── config_files/                # Example configurations
├── data/                        # Mesh files
└── simulation_results/          # Output directory
```

## Academic Context

Developed for **INF203** at NMBU demonstrating:
- Numerical methods for PDEs
- Python packaging and CLI development
- Scientific computing with NumPy/Matplotlib
- Software testing with pytest

## Authors

- Marcus Dalaker Figenschou - NMBU INF203 Student
- Sindri Thorsteinsson - NMBU INF203 Student

---

*2-week project showcasing finite volume methods and scientific Python programming*