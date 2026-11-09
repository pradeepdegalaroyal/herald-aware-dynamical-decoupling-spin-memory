"""Herald-aware dynamical decoupling schedules for a spin quantum memory."""

from .noise import PRESETS, NoiseModel
from .objective import (
    expected_infidelity,
    free_evolution_infidelity,
    oracle_cpmg_infidelity,
)
from .schedules import cpmg_spacings, expand_tail, geometric_spacings, refocus_points
from .timing import HeraldDistribution

__all__ = [
    "NoiseModel",
    "PRESETS",
    "HeraldDistribution",
    "cpmg_spacings",
    "geometric_spacings",
    "expand_tail",
    "refocus_points",
    "expected_infidelity",
    "free_evolution_infidelity",
    "oracle_cpmg_infidelity",
]
__version__ = "0.1.0"
