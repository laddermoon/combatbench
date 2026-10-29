"""Strike target issuer — resolves stance-relative job params into a
world-frame target and issues the v2 effector command.

Reads ``episode_options`` params sampled by the experiment:

    {rid}_strike_hand      "left" | "right"
    {rid}_strike_dist      forward distance in stance frame (m)
    {rid}_strike_lat       lateral offset, + toward robot's left (m)
    {rid}_strike_z         absolute target height (m)

Runs AFTER the state-init plugin, before the observer dispatcher. Computes the initial
Point5State, derives the stance forward, resolves the world target,
issues ``set_effector_world`` via the mutator, and stores the pre-action
P5 and target as scalar episode metrics for the offline reward builder.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BasePlugin, SimContext
from envs.framework.observer_plugin import OBSERVER_DISPATCHER_PRIORITY
from point5.contracts import POINT_INDEX, StaticStartResolver
from point5.humanoid21.adapter import point5_from_view
from point5.punch_planner import _stance_forward, plan_punch
from point5.trajectory import build_reference


class StrikeTargetPlugin(BasePlugin):
    """Issue a world-fixed strike target per agent."""

    def __init__(self, target_robots=("robot_a", "robot_b")) -> None:
        self._targets = tuple(target_robots)

    @property
    def name(self) -> str:
        return "strike_target"

    @property
    def require_mutator(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return OBSERVER_DISPATCHER_PRIORITY + 1

    def on_pre_episode(self, ctx: SimContext) -> None:
        opts = ctx.episode_options or {}
        ds = ctx.accessor.get_derived_state(list(self._targets))
        for rid in self._targets:
            hand = str(opts.get(f"{rid}_strike_hand", "left"))
            dist = float(opts.get(f"{rid}_strike_dist", 0.6))
            lat = float(opts.get(f"{rid}_strike_lat", 0.0))
            z = float(opts.get(f"{rid}_strike_z", 1.10))
            p5 = point5_from_view(ds[rid], rid, with_vel=False)
            head = p5.point_pos("head")
            fwd = _stance_forward(p5.point_pos("foot_left"),
                                  p5.point_pos("foot_right"))
            if not np.any(fwd):
                fwd = np.array([1.0, 0.0, 0.0])
            left = np.array([-fwd[1], fwd[0], 0.0])
            tgt = np.array([head[0] + fwd[0] * dist + left[0] * lat,
                            head[1] + fwd[1] * dist + left[1] * lat,
                            z])
            side = 1 if hand == "left" else -1
            # Plan the punch here too: the reference's striking-hand
            # path becomes a moving guidance target (one world waypoint
            # per action step, ending at the fixed strike point).  A
            # static endpoint leaves the load->extend->snap timing
            # invisible to the policy — the proven s4 mechanism.
            plan = plan_punch(p5, tgt, hand)
            if plan.reachable:
                ref = build_reference(
                    plan.task,
                    StaticStartResolver(p5, np.zeros(3), np.eye(3)))
                hidx = POINT_INDEX[f"hand_{hand}"]
                n = int(np.ceil((ref.duration - 1e-9) / 0.05))
                track = np.stack([
                    ref.sample_time(min((i + 1) * 0.05,
                                      ref.duration)).pos[hidx]
                    for i in range(n)])
                ctx.mutator.set_effector_world_track(
                    rid, +1, side, track)
            else:
                ctx.mutator.set_effector_world(rid, +1, side, tgt)
            for i, value in enumerate(p5.pos.reshape(-1)):
                ctx.metrics[f"{rid}_strike_p5_{i}"] = float(value)
            ctx.metrics[f"{rid}_strike_tx"] = float(tgt[0])
            ctx.metrics[f"{rid}_strike_ty"] = float(tgt[1])
            ctx.metrics[f"{rid}_strike_tz"] = float(tgt[2])
            ctx.metrics[f"{rid}_strike_hand_i"] = (
                0.0 if hand == "left" else 1.0)
            ctx.metrics[f"{rid}_strike_ok"] = 1.0

    def to_blueprint(self) -> Dict[str, Any]:
        return {"target_robots": list(self._targets)}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StrikeTargetPlugin":
        return cls(**config)
