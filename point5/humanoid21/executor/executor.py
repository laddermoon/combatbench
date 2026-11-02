"""Executor — stateless GenAction(context) -> Action.

Flow per call (per bootstrip.md):

    0. verify clip frame 0 anchored to the observation's P5
    1. extract centroidal motion state from the observation
    2. centroidal motion planning  (algorithm 1 — stub)
    3. compute the next action     (algorithm 2 — stub:
       CNext -> P5Next -> [optional Humanoid21RefNext] -> Action)

The only owned resource is the internal working simulator used for
rollouts; no progress, plan, or cache survives a call.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .contracts import (Action, AnchorError, CentroidalPlan, Context,
                        ExecutorError, Observation, PlanError)


class Executor:
    """Stateless servo: ``gen_action`` is a pure per-call computation."""

    anchor_atol: float = 1e-6            # m, frame-0 identity check

    def __init__(self, wsim=None, robot_id: str = "robot_a",
                 cfg: Optional[Any] = None):
        """Initializes ONLY the internal working simulator."""
        self.robot_id = robot_id
        self.other_id = ("robot_b" if robot_id == "robot_a"
                         else "robot_a")
        self.cfg = cfg
        if wsim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            wsim = Humanoid21Simulator(initial_distance=6.0,
                                       initial_pose_a="standing")
            wsim.reset(seed=0)
        self.wsim = wsim

    # -- public API ----------------------------------------------------------
    def gen_action(self, ctx: Context) -> Action:
        """Context -> Action.  Stateless."""
        self._verify_anchor(ctx)
        c0 = self._centroidal_state(ctx.observation)
        plan = self._plan_centroidal(ctx, c0)
        return self._compute_action(ctx, plan)

    # -- pipeline steps ------------------------------------------------------
    def _verify_anchor(self, ctx: Context) -> None:
        """Step 0: clip frame0 must equal the current local P5."""
        dev = np.linalg.norm(
            np.asarray(ctx.clip.keyframes[0].pos)
            - np.asarray(ctx.observation.p5.pos), axis=1)
        if float(dev.max()) > self.anchor_atol:
            raise AnchorError(
                f"clip frame0 not anchored to current "
                f"(max dev {dev.max():.4g} > {self.anchor_atol})")

    def _centroidal_state(self, obs: Observation) -> np.ndarray:
        """Step 1: true COM position/velocity/ang-mom — the plan's
        starting state (already measured in Observation)."""
        return np.concatenate([obs.com_pos, obs.com_vel,
                               obs.ang_mom])

    def _plan_centroidal(self, ctx: Context,
                         c0: np.ndarray) -> CentroidalPlan:
        """Step 2 (algorithm 1): centroidal motion planning.

        From current COM kinematics + remaining P5 clip produce a
        timed centroidal trajectory (position/velocity per step,
        contact phases, keyframe event times) and the next step's
        required centroidal state CNext.

        Priors available: mass_proxy per keyframe (P5->COM estimate),
        support-phase detection, capability caps for timing.

        TODO: implement.
        """
        raise PlanError("algorithm 1 (centroidal planning) "
                        "not implemented")

    def _compute_action(self, ctx: Context,
                        plan: CentroidalPlan) -> Action:
        """Step 3 (algorithm 2): next action from plan + observation.

        CNext -> P5Next (progress-consistent next-frame P5)
              -> optional Humanoid21RefNext (pose_solver)
              -> Action via MPC/simulation search.

        TODO: implement.
        """
        raise PlanError("algorithm 2 (action computation) "
                        "not implemented")
