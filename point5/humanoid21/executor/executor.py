"""Executor — stateless GenAction(context) -> Action.

Flow per call (per bootstrip.md):

    0. verify clip frame 0 anchored to the observation's P5
    1. extract the start SupportLandMark from the observation (tool8)
    2. support planning             (algorithm-1 branch)
    3. SupportLandMark sequence generation (algorithm 1 — TODO)
    4. compute the next action      (algorithm 2 — stub)

The only owned resource is the internal working simulator used for
rollouts; no progress, plan, or cache survives a call.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .action_computer import ActionComputer
from .support_planner import SupportPlanner
from .contracts import (Action, AnchorError, Context,
                        ExecutorError, Observation, PlanError)

ActionT = Action  # (21,) normalized PD target — the sim input


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
        self.support = SupportPlanner()
        self.action_computer = ActionComputer()

    # -- public API ----------------------------------------------------------
    def gen_action(self, ctx: Context) -> Action:
        """Context -> Action.  Stateless."""
        self._verify_anchor(ctx)
        landmarks = self.support.plan(
            ctx.observation, ctx.clip)
        return self.action_computer.compute(
            ctx, landmarks, self.wsim)

    # -- pipeline steps ------------------------------------------------------
    def _extract_p5(self, observation: np.ndarray) -> np.ndarray:
        """96-dim obs -> local P5 (5,3) via the shared FK decode."""
        from .tools.obs_decoder import ObsDecoder
        local_p5, _ = ObsDecoder(self.robot_id
                                 ).decode_p5_local(observation)
        return local_p5

    def _verify_anchor(self, ctx: Context) -> None:
        """Step 0: clip frame0 must equal the current local P5."""
        pos = self._extract_p5(ctx.observation)
        dev = np.linalg.norm(
            np.asarray(ctx.clip.keyframes[0].pos)
            - np.asarray(pos), axis=1)
        if float(dev.max()) > self.anchor_atol:
            raise AnchorError(
                f"clip frame0 not anchored to current "
                f"(max dev {dev.max():.4g} > {self.anchor_atol})")


