"""Strike target issuer — resolves stance-relative job params into a
world-frame target and issues the v2 effector command.

Reads ``episode_options`` params sampled by the experiment:

    {rid}_strike_hand      "left" | "right"
    {rid}_strike_dist      forward distance in stance frame (m)
    {rid}_strike_lat       lateral offset, + toward robot's left (m)
    {rid}_strike_z         absolute target height (m)

Runs AFTER the state-init plugin (priority 40).  Computes the initial
Point5State, derives the stance forward, resolves the world target,
converts it to pelvis-ego and issues ``set_effector`` (v2 command,
hand strike) on the CommandSimulator — whose obs pipeline re-expresses
the target in the instantaneous ego frame each step.  Also writes
``ctx.metrics`` ({rid}_strike_target/hand/hand_i) for the offline
reward builder.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin, SimContext
from mocap.point5.humanoid21.adapter import point5_from_view
from mocap.point5.punch_planner import _norm, _stance_forward, _xy

_TGT_SCALE = 1.5


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
        return 40        # after PolicyInitStatePlugin (50)

    def on_pre_episode(self, ctx: SimContext) -> None:
        opts = ctx.episode_options or {}
        ds = ctx.accessor.get_derived_state(list(self._targets))
        cs = ctx.accessor.get_core_state()
        for rid in self._targets:
            hand = str(opts.get(f"{rid}_strike_hand", "left"))
            dist = float(opts.get(f"{rid}_strike_dist", 0.6))
            lat = float(opts.get(f"{rid}_strike_lat", 0.0))
            z = float(opts.get(f"{rid}_strike_z", 1.10))
            try:
                p5 = point5_from_view(ds[rid], rid, with_vel=False)
            except Exception:
                continue
            head = p5.point_pos("head")
            fwd = _stance_forward(p5.point_pos("foot_left"),
                                  p5.point_pos("foot_right"))
            if not np.any(fwd):
                fwd = np.array([1.0, 0.0, 0.0])
            left = np.array([-fwd[1], fwd[0], 0.0])
            tgt = np.array([head[0] + fwd[0] * dist + left[0] * lat,
                            head[1] + fwd[1] * dist + left[1] * lat,
                            z])
            # ego = Rz(-yaw) * (tgt - root_xy), z relative to root
            root = cs[rid]
            rpos = np.asarray(root["root_pos"], dtype=np.float64)
            q = np.asarray(root["root_rot"], dtype=np.float64)
            yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler(
                "xyz")[2]
            c, s = np.cos(-yaw), np.sin(-yaw)
            d = tgt - np.array([rpos[0], rpos[1], rpos[2]])
            ego = np.array([c * d[0] - s * d[1],
                            s * d[0] + c * d[1], d[2]])
            sim = ctx.mutator
            if hasattr(sim, "set_effector"):
                side = 1 if hand == "left" else -1
                sim.set_effector(rid, +1, side, tuple(ego))
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
