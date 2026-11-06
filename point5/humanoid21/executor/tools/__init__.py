"""Tools — stateless P5 physics/geometry utilities used by the executor.

Each module is a pure function family: P5/state in -> result out.
See README.md for the per-tool reference.
"""
from .com_solver import com_estimate, mass_estimate
from .inertia_solver import inertia_estimate, inertia_per_mass
from .capability_solver import Capability, capability
from .cop_solver import cop_ranges, foot_cop_range
from .landmark_viz import render_landmarks

__all__ = [
    "com_estimate", "mass_estimate",
    "inertia_estimate", "inertia_per_mass",
    "Capability", "capability",
    "cop_ranges", "foot_cop_range",
    "render_landmarks",
]
