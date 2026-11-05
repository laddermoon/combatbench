"""Tool 2: P5 local pose -> Humanoid21 reference pose + action.

Wraps ``mechanics.solve_equilibrium`` (two-stage, expensive ~seconds).
``solve_fast``/``build_lut`` are reserved for the quantized warm-start
table: (head, feet) -> stage-1 body solution, then closed-form arm IK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PoseResult:
    """A solved Humanoid21 reference pose for one local P5."""

    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    q_rad: np.ndarray                    # (21,) joint radians
    action: np.ndarray                   # (21,) normalized PD target
    satisfied: bool
    failed_stage: str = ""
    energy: float = 0.0


class Humanoid21PoseSolver:
    """Humanoid21-specific P5 -> reference pose.

    ``solve`` wraps mechanics.solve_equilibrium; the solver's own
    ``q_norm`` is already the simulator's normalized PD target space.
    """

    def __init__(self, skeleton=None):
        self.skeleton = skeleton
        self._lut = None              # quantized warm-start table

    def _sk(self):
        if self.skeleton is None:
            from point5.humanoid21.mechanics import Skeleton
            self.skeleton = Skeleton()
        return self.skeleton

    def solve(self, local_p5: np.ndarray,
              tolerance: float = 0.02) -> PoseResult:
        """Exact solve (equilibrium solver)."""
        from point5.humanoid21.mechanics import solve_equilibrium
        res = solve_equilibrium(np.asarray(local_p5, np.float64),
                                skeleton=self._sk(),
                                tolerance=tolerance)
        return PoseResult(
            root_pos=np.asarray(res["root_pos"]),
            root_quat_wxyz=np.asarray(res["root_quat_wxyz"]),
            q_rad=np.asarray(res["q_rad"]),
            action=np.clip(np.asarray(res["q_norm"]), -1.0, 1.0),
            satisfied=bool(res["satisfied"]),
            failed_stage=str(res.get("failed_stage") or ""),
            energy=float(res.get("energy", 0.0)))

    def solve_fast(self, local_p5: np.ndarray) -> PoseResult:
        """LUT warm-start + polish; falls back to ``solve`` on miss.

        TODO: quantized (head,feet)->stage1 body table.
        """
        return self.solve(local_p5)

    def build_lut(self, samples: np.ndarray) -> None:
        """Precompute the (head,feet)->stage1 body table.

        TODO.
        """
        raise NotImplementedError
