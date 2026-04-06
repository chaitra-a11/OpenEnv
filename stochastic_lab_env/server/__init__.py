"""Server components for the stochastic lab environment."""

from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from .stochastic_lab_environment import StochasticLabEnvironment

__all__ = ["StochasticLabEnvironment"]
