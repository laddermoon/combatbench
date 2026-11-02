"""Algorithm 1: centroidal motion planning.

Input : Observation (obs vector) -> extracted P5 + current centroidal
        state (com, com_vel, ang_mom) + the remaining P5Clip.
Output: CentroidalPlan — timed centroidal trajectory (pos/vel per
        step), contact/support phases, keyframe event times, and the
        next control step's required centroidal state CNext.

Priors available: tool1 ``com_estimate`` (P5 -> recommended COM),
support detection from keyframe foot displacements, capability caps
for timing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import CentroidalPlan, Context, PlanError


class CentroidalPlanner:
    """Stateless centroidal planner used by Executor.gen_action."""

    def extract_centroidal_state(self, observation: np.ndarray
                                 ) -> np.ndarray:
        """Obs vector -> (com_pos, com_vel, ang_mom).

        TODO: implement extraction (needs the obs keypoint/COM
        channels or P5 decode + tool1).
        """
        raise PlanError("centroidal-state extraction not implemented")

    def plan(self, ctx: Context, c0: np.ndarray) -> CentroidalPlan:
        """Centroidal plan from current state + remaining clip.

        TODO: implement.
        """
        raise PlanError("algorithm 1 (centroidal planning) "
                        "not implemented")
