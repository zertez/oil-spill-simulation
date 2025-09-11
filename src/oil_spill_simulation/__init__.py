# src/oil_spill_simulation/__init__.py
"""Oil Spill Simulation Package"""

__version__ = "1.0.0"

from .config import TOMLFileReader
from .mesh import Mesh
from .simulation import OilCalculation
from .visualization import Animation

__all__ = ["OilCalculation", "Mesh", "Animation", "TOMLFileReader"]
