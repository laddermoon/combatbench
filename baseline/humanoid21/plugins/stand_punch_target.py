"""Stand-punch target issuer — issues a world-fixed strike target plus
the planner's strike-hand guidance track from the standard standing
pose.

Reads ``episode_options`` params sampled by the experiment (targets are
pre-validated offline against ``plan_stand_punch``):

    {rid}_sp_hand         "left" | "right"
    {rid}_sp_tx/ty/tz     target point in the robot's initial local frame
                          (+X = torso facing, +Y = torso left)

Runs after episode reset, before the observer dispatcher.  Measures
the post-reset Point5State (v2 local frame), resolves the pre-planned
clip under the measured frame pose f1 (the plan runs once, offline, in
``build_jobs`` — no re-planning here and no silent fallback), and
stores the start P5 / f1 / local+world targets as scalar episode
metrics — same key names as StrikeTargetPlugin so the punch scorer
works unchanged.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from baseline.humanoid21.hierarchy.p5_world import (
    clip_world_samples, f1_from_state, p5_to_world, prepend_start,
    retime_clip)
from envs.framework import BasePlugin, SimContext
from envs.framework.observer_plugin import OBSERVER_DISPATCHER_PRIORITY
from point5.contracts import P5Clip, POINT_INDEX
from point5.humanoid21.adapter import point5_from_view

# Per-segment real-time assignment (P5 clip dur is a schema placeholder):
# stand->coil, coil->swing, swing->land, land->strike.
SEGMENT_DURATIONS = (0.10, 0.20, 0.20, 0.15)   # total 0.65 s = 13 actions
DT = 0.05


def timed_stand_clip(clip, start_local_pos) -> object:
    """Prepend the measured stand pose and assign real durations."""
    clip = prepend_start(clip, start_local_pos, SEGMENT_DURATIONS[0])
    return retime_clip(clip, SEGMENT_DURATIONS)


class StandPunchTargetPlugin(BasePlugin):
    """Issue a world-fixed strike target + hand track per agent."""

    def __init__(self, target_robots=("robot_a", "robot_b")) -> None:
        self._targets = tuple(target_robots)

    @property
    def name(self) -> str:
        return "stand_punch_target"

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
            hand = str(opts.get(f"{rid}_sp_hand", "left"))
            tgt_local = np.array(
                [float(opts.get(f"{rid}_sp_t{a}", d))
                 for a, d in zip("xyz", (1.05, 0.0, 1.1))],
                dtype=np.float64)
            p5 = point5_from_view(ds[rid], rid, with_vel=False)
            f1 = f1_from_state(p5)
            tgt = p5_to_world(tgt_local, f1)
            side = 1 if hand == "left" else -1
            clip_d = opts.get(f"{rid}_sp_clip")
            if clip_d is None:
                raise ValueError(
                    f"{rid}_sp_clip missing from episode_options — "
                    "stand-punch plans are validated offline in "
                    "build_jobs; there is no reset-time fallback")
            hidx = POINT_INDEX[f"hand_{hand}"]
            clip = timed_stand_clip(P5Clip.from_dict(clip_d), p5.pos)
            track = clip_world_samples(clip, f1, DT)[:, hidx]
            ctx.mutator.set_effector_world_track(
                rid, +1, side, track)
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
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandPunchTargetPlugin":
        return cls(**config)
