#!/usr/bin/env python3
"""
Oil Spill Simulation - Main entry point for zipapp execution

This module allows the package to be run directly as:
- python -m oil_spill_simulation
- python oil_spill_sim.pyz
- ./oil_spill_sim.pyz (if executable)
"""

from .main import main

if __name__ == "__main__":
    main()
