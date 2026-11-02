"""Tool 1: P5 local pose -> Humanoid21 reference pose + action.

Interface for the pose-completion prior.  Implementation options:

- exact: ``mechanics.solve_equilibrium`` (two-stage, ~seconds)
- fast: quantized LUT over (head, feet) -> stage-1 body solution,
  then closed-form stage-2 arm IK — warm start + polish

Both paths return the same ``PoseResult``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .contracts import PoseResult


class Humanoid21PoseSolver:
    """Humanoid21-specific P5 -> reference pose.

    TODO: implement ``solve`` (exact) and ``solve_fast`` (LUT +
    polish).  ``solve`` may just wrap mechanics.solve_equilibrium.
    """

    def __init__(self, iface=None, skeleton=None):
        self.iface = iface            # SimInterface for q->action norm
        self.skeleton = skeleton
        self._lut = None              # quantized warm-start table

    def solve(self, local_p5: np.ndarray) -> PoseResult:
        """Exact solve (equilibrium solver)."""
        raise NotImplementedError

    def solve_fast(self, local_p5: np.ndarray) -> PoseResult:
        """LUT warm-start + polish; falls back to ``solve`` on miss."""
        raise NotImplementedError

    def build_lut(self, samples: np.ndarray) -> None:
        """Precompute the (head,feet)->stage1 body table."""
        raise NotImplementedError
