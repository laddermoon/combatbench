"""Algorithm 2: next-action computation.

Input : Observation + Context (remaining clip) + CentroidalPlan.
Output: Action — the (21,) normalized PD target for one control step.

Steps (per bootstrip):
    CNext -> P5Next (progress-consistent next-frame P5)
          -> optional Humanoid21RefNext (tool2 pose_solver)
          -> Action via MPC / simulation search on the working sim.

The hard part learned earlier: nothing on this robot is open-loop
stable — the action layer must close the loop on balance, not just
track references.
"""
from __future__ import annotations

import numpy as np

from .centroidal import CentroidalTrajectory
from .contracts import Context, PlanError


class ActionComputer:
    """Stateless action computation used by Executor.gen_action."""

    def compute(self, ctx: Context, plan: CentroidalTrajectory,
                wsim) -> np.ndarray:
        """Return the next (21,) action.

        ``wsim`` is the executor's internal working simulator for
        candidate rollouts.  TODO: implement.
        """
        raise PlanError("algorithm 2 (action computation) "
                        "not implemented")
