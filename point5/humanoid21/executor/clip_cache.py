"""E2: clip submission validation + instruction cache.

Caller-side builder prepends the measured standing pose so the submitted
clip's frame 0 is anchored to the current P5 — a hard caller
requirement, verified again on submission (no auto-fix).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from point5.contracts import (
    P5Clip, Keyframe, Transfer, to_local, to_world)
from point5.mass_model import mass_proxy
from point5.trajectory import FramePose, keyframe_frames, keyframe_world
from .contracts import AnchorError, Submission


def build_stand_punch_clip(current_p5, target_local, hand,
                           punch_cfg=None) -> P5Clip:
    """Caller-side: anchored start -> coil -> swing -> land -> strike."""
    from point5.humanoid21.stand_punch import (
        StandPunchConfig, plan_stand_punch)
    plan = plan_stand_punch(np.asarray(target_local, float), hand,
                            punch_cfg or StandPunchConfig(),
                            initial=current_p5)
    if not plan.reachable:
        raise AnchorError(f"plan_stand_punch rejected: {plan.reason}")
    task = plan.task
    kf0 = task.keyframes[0]
    start = Keyframe(pos=np.asarray(current_p5.pos, dtype=np.float64),
                     label="start", meta=dict(kf0.meta))
    # verify coil keeps frame origin+yaw (dx=dy=dyaw=0 is then exact)
    w = keyframe_world(task, (np.asarray(current_p5.metadata[
        "frame_origin_xy"], np.float64),
        float(current_p5.metadata["frame_yaw_deg"])))
    o0 = (w[0, 3, :2] + w[0, 4, :2]) / 2
    o_cur = np.asarray(current_p5.metadata["frame_origin_xy"],
                       dtype=np.float64)
    if np.linalg.norm(o0 - o_cur) > 1e-6:
        raise AnchorError("coil moves ankle midpoint — nonzero transfer "
                          "required, builder out of date")
    kf1 = Keyframe(pos=kf0.pos, label=kf0.label, meta=dict(kf0.meta),
                   transfer=Transfer(dx=0.0, dy=0.0, dyaw=0.0,
                                     dur=0.1))
    return P5Clip(name=task.name,
                  keyframes=(start, kf1) + task.keyframes[1:],
                  metadata=dict(task.metadata))


def _rel_frame(a: FramePose, b: FramePose) -> Transfer:
    """Pose of frame b expressed in frame a (SE2 inverse-compose)."""
    dx_w = np.asarray(b[0]) - np.asarray(a[0])
    r = np.deg2rad(a[1])
    c, s = np.cos(r), np.sin(r)
    return Transfer(dx=float(c * dx_w[0] + s * dx_w[1]),
                    dy=float(-s * dx_w[0] + c * dx_w[1]),
                    dyaw=float(b[1] - a[1]), dur=0.1)


class InstructionCache:
    """Holds the active clip; maintains internal anchoring as the
    executor tracks it (caller does NOT re-anchor mid-execution)."""

    def __init__(self, anchor_atol: float = 1e-6):
        self.anchor_atol = anchor_atol
        self.sub: Optional[Submission] = None
        self.next_kf = 1              # index of next unconfirmed keyframe
        self.head_frame: FramePose = (np.zeros(2), 0.0)  # internal frame0
        self.head_pos: Optional[np.ndarray] = None

    # -- submission ---------------------------------------------------------
    def submit(self, clip: P5Clip, current_p5) -> Submission:
        """Hard check: clip frame0 local pos == current p5 local pos."""
        dev = np.linalg.norm(np.asarray(clip.keyframes[0].pos)
                             - np.asarray(current_p5.pos), axis=1)
        if float(dev.max()) > self.anchor_atol:
            raise AnchorError(
                f"clip frame0 not anchored to current "
                f"(max dev {dev.max():.4g} m > {self.anchor_atol:g})")
        f1 = (np.asarray(current_p5.metadata["frame_origin_xy"],
                         np.float64),
              float(current_p5.metadata["frame_yaw_deg"]))
        wf = keyframe_frames(clip, f1)
        wpts = keyframe_world(clip, f1)
        com = np.stack([to_world(
            mass_proxy(k.pos).com.reshape(1, 3), *wf[i])[0]
            for i, k in enumerate(clip.keyframes)])
        self.sub = Submission(clip=clip, f1=f1, world_frames=wf,
                              world_pts=wpts, com_proxy=com)
        self.next_kf = 1
        self.head_frame = f1
        self.head_pos = np.asarray(current_p5.pos, np.float64)
        return self.sub

    # -- internal re-anchoring during execution ------------------------------
    def update_head(self, current_p5) -> None:
        """Adopt the actual current P5 as the internal first frame while
        preserving every unconfirmed keyframe's WORLD targets."""
        if self.sub is None:
            return
        self.head_frame = (
            np.asarray(current_p5.metadata["frame_origin_xy"],
                       np.float64),
            float(current_p5.metadata["frame_yaw_deg"]))
        self.head_pos = np.asarray(current_p5.pos, np.float64)

    def confirm_keyframe(self) -> None:
        if self.next_kf < len(self.sub.clip.keyframes):
            self.next_kf += 1

    # -- views ---------------------------------------------------------------
    def remaining_transfer(self) -> Optional[Transfer]:
        """Transfer from the internal head frame to the next keyframe's
        frame — what a virtual clip would carry on that keyframe."""
        if self.sub is None or self.next_kf >= len(
                self.sub.clip.keyframes):
            return None
        return _rel_frame(self.head_frame,
                          self.sub.world_frames[self.next_kf])

    def target_world(self) -> Optional[np.ndarray]:
        if self.sub is None or self.next_kf >= len(
                self.sub.clip.keyframes):
            return None
        return self.sub.world_pts[self.next_kf]

    def clear(self) -> None:
        self.sub = None
        self.next_kf = 0
