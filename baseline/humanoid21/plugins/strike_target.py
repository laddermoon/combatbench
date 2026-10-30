"""Strike target issuer — resolves stance-relative job params into a
world-frame target and issues the v2 effector command.

Reads ``episode_options`` params sampled by the experiment:

    {rid}_strike_hand      "left" | "right"
    {rid}_strike_dist      forward distance in stance frame (m)
    {rid}_strike_lat       lateral offset, + toward robot's left (m)
    {rid}_strike_z         absolute target height (m)

Runs AFTER the state-init plugin, before the observer dispatcher. Computes the initial
Point5State (v2 local frame), samples the target in that frame (local
+X is the torso facing, +Y the torso left), converts it to world via
the frame pose ``f1`` stored in the adapter metadata, and issues
``set_effector_world[_track]`` via the mutator.  The start P5, f1 and
both target frames are stored as scalar episode metrics for the
offline reward builder.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from baseline.humanoid21.hierarchy.p5_world import (
    clip_world_samples, f1_from_state, p5_to_world)
from envs.framework import BasePlugin, SimContext
from envs.framework.observer_plugin import OBSERVER_DISPATCHER_PRIORITY
from point5.contracts import POINT_INDEX
from point5.humanoid21.adapter import point5_from_view
from point5.humanoid21.punch_planner import plan_punch


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
            f1 = f1_from_state(p5)
            # Local frame IS the stance frame: +X = torso facing,
            # +Y = torso left, z untouched by the frame transform.
            head = p5.point_pos("head")
            tgt_local = np.array([head[0] + dist, head[1] + lat, z],
                                 dtype=np.float64)
            tgt = p5_to_world(tgt_local, f1)
            side = 1 if hand == "left" else -1
            # Plan the punch here too: the reference's striking-hand
            # path becomes a moving guidance target (one world waypoint
            # per action step, ending at the fixed strike point).  A
            # static endpoint leaves the load->extend->snap timing
            # invisible to the policy — the proven s4 mechanism.
            plan = plan_punch(p5, tgt_local, hand)
            if plan.reachable:
                hidx = POINT_INDEX[f"hand_{hand}"]
                track = clip_world_samples(plan.task, f1, 0.05)[:, hidx]
                ctx.mutator.set_effector_world_track(
                    rid, +1, side, track)
            else:
                ctx.mutator.set_effector_world(rid, +1, side, tgt)
            for i, value in enumerate(p5.pos.reshape(-1)):
                ctx.metrics[f"{rid}_strike_p5_{i}"] = float(value)
            ctx.metrics[f"{rid}_strike_f1_ox"] = float(f1[0][0])
            ctx.metrics[f"{rid}_strike_f1_oy"] = float(f1[0][1])
            ctx.metrics[f"{rid}_strike_f1_yaw"] = float(f1[1])
            ctx.metrics[f"{rid}_strike_tx"] = float(tgt[0])
            ctx.metrics[f"{rid}_strike_ty"] = float(tgt[1])
            ctx.metrics[f"{rid}_strike_tz"] = float(tgt[2])
            for i, value in enumerate(tgt_local):
                ctx.metrics[f"{rid}_strike_lt{'xyz'[i]}"] = float(value)
            ctx.metrics[f"{rid}_strike_hand_i"] = (
                0.0 if hand == "left" else 1.0)
            ctx.metrics[f"{rid}_strike_ok"] = 1.0

    def to_blueprint(self) -> Dict[str, Any]:
        return {"target_robots": list(self._targets)}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StrikeTargetPlugin":
        return cls(**config)
