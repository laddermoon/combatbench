"""Executor — closed-loop P5 instruction servo."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from point5.contracts import P5Clip
from point5.trajectory import FramePose
from point5.trajectory import keyframe_frames, keyframe_world
from .contracts import (AnchorError, ExecutorConfig, PlanError,
                        RefFrame, StepResult)
from .state import capture, current_state, com_state, foot_contacts
from .reference import plan_reference
from .search import score_state, search_window, _measure


class Executor:
    """Servo: current state + clip cache -> next action each cycle.

    Usage:
        ex = Executor(wsim)         # wsim = working sim for MPC
        ex.submit(current, clip, f1, hand, target)   # validated clip
        r = ex.step(current)        # returns action to apply once
    """

    def __init__(self, wsim, cfg: Optional[ExecutorConfig] = None,
                 robot_id: str = "robot_a"):
        self.cfg = cfg or ExecutorConfig()
        self.wsim = wsim
        self.robot_id = robot_id
        self.other_id = ("robot_b" if robot_id == "robot_a"
                         else "robot_a")
        self.clip = None
        self.world_pts = None
        self.refs: List[RefFrame] = []
        self.key_times = []          # (keyframe_idx, t) in plan time
        self.event_idx = 0           # next keyframe index to confirm
        self.status = "HOLDING"
        self.plan_t = 0.0
        self.replans = 0
        try:
            from envs.humanoid21.meta import Humanoid21Meta
            self.prev_action = np.asarray(
                Humanoid21Meta.INITIAL_POSES["standing"]["action"],
                dtype=np.float32)
        except Exception:
            self.prev_action = np.zeros(21, dtype=np.float32)
        self.prev_seq = None
        self.t_submit = None
        self.events = []
        self.settle_since = None
        self.vlog = []

    # -- instruction cache ---------------------------------------------------
    def submit(self, current: Dict[str, Any], clip: P5Clip,
               f1: FramePose, hand: str = "right",
               target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate + adopt a clip whose first frame anchors current P5.

        Raises AnchorError without touching the existing cache.
        """
        p5 = current["p5"]
        d = np.linalg.norm(np.asarray(clip.keyframes[0].pos)
                           - np.asarray(p5.pos), axis=1)
        if float(d.max()) > self.cfg.anchor_atol:
            raise AnchorError(
                f"clip frame0 not anchored to current "
                f"(max dev {d.max():.4f} m > {self.cfg.anchor_atol})")
        com = com_state(self.wsim_for(current), self.robot_id) \
            if "com_pos" not in current else current["com_pos"]
        com_pos = np.asarray(current.get("com_pos",
                                         com_state(self.wsim_for(current),
                                                   self.robot_id).com_pos))
        com_vel = np.asarray(current.get("com_vel",
                                         com_state(self.wsim_for(current),
                                                   self.robot_id).lin_mom))
        self.clip = clip
        self.world_pts = keyframe_world(clip, f1)
        cst = com_state(self.wsim_for(current), self.robot_id)
        self.refs = plan_reference(
            clip, f1, cst.com_pos,
            np.asarray(cst.lin_mom) / cst.mass, self.cfg)
        # keyframe end times = seg boundaries (s=1 events)
        self.key_times = []
        for fr in self.refs:
            if fr.event:
                if not self.key_times or \
                        self.key_times[-1][1] != fr.t:
                    self.key_times.append((fr.seg + 1, fr.t))
        self.event_idx = 1          # frame 0 = anchor, confirmed
        self.status = "RUNNING"
        self.plan_t = 0.0
        self.replans = 0
        self.prev_seq = None
        self.t_submit = float(current["time"])
        self.events = [{"t": 0.0, "event": "submitted",
                        "label": clip.keyframes[0].label}]
        self.settle_since = None
        return {"accepted": True,
                "duration_s": self.refs[-1].t if self.refs else 0.0}

    def clear(self):
        self.clip = None
        self.refs = []
        self.status = "SETTLING"
        self.settle_since = None

    def wsim_for(self, current):
        return current["sim"] if "sim" in current else self.wsim

    # -- per-cycle step ------------------------------------------------------
    def step(self, current: Dict[str, Any],
             snapshot: Any = None) -> StepResult:
        """Return the action to apply for this control period.

        ``current`` is fresh measured state; ``snapshot`` is a full
        working-sim Snapshot taken at the same instant.
        """
        cfg = self.cfg
        if self.status == "FAILED":
            return StepResult(self.prev_action.copy(), "FAILED")
        meas = _measure(self.wsim, self.robot_id, None, cfg.dt, cfg)

        # advance plan clock (first call after submit counts step 1)
        if self.status != "HOLDING" or self.clip is not None:
            self.plan_t += cfg.dt

        # confirm pending keyframe if reached (positions within tol)
        if self.event_idx < len(self.clip.keyframes) if self.clip \
                else False:
            due_t = [t for k, t in self.key_times
                     if k == self.event_idx]
            tgt = self.world_pts[self.event_idx]
            if due_t:
                err = np.linalg.norm(meas["p5w"] - tgt, axis=1).max()
                if err <= max(self.refs[0].tol_point.max(), 0.12):
                    self.events.append(
                        {"t": self.plan_t, "event": "reached",
                         "idx": self.event_idx,
                         "label": self.clip.keyframes[
                             self.event_idx].label,
                         "err": float(err)})
                    self.event_idx += 1
                elif self.plan_t > due_t[0] + cfg.replan_wait:
                    if self.replans < cfg.max_replans:
                        self.replans += 1
                        self.events.append(
                            {"t": self.plan_t, "event": "replan",
                             "idx": self.event_idx})
                    else:
                        self.status = "FAILED"
                        return StepResult(
                            self.prev_action.copy(), "FAILED",
                            {"reason": "keyframe_missed",
                             "idx": self.event_idx,
                             "err": float(err)})

        # done all keyframes -> settle/hold
        if self.clip is not None and \
                self.event_idx >= len(self.clip.keyframes):
            if self.status == "RUNNING":
                self.status = "SETTLING"
                self.events.append(
                    {"t": self.plan_t, "event": "keyframes_done"})
            spd = float(np.linalg.norm(meas["com_vel"]))
            if self.settle_since is None and spd < cfg.hold_com_vel:
                self.settle_since = self.plan_t
            if self.settle_since is not None and \
                    self.plan_t - self.settle_since > 0.5:
                self.status = "HOLDING"

        # reference window for MPC (after last ref: hold last pose)
        if self.refs:
            t_now = self.plan_t
            win = []
            for h in range(cfg.horizon):
                tt = t_now + (h + 1) * cfg.dt
                j = int(np.searchsorted(
                    [r.t for r in self.refs], tt, side="right") - 1)
                j = int(np.clip(j, 0, len(self.refs) - 1))
                win.append(self.refs[j])
        else:
            # no clip: hold current pose as reference
            r = RefFrame(t=self.plan_t, p5_world=meas["p5w"],
                         p5_vel_world=np.zeros((5, 3)),
                         com_pos=meas["com"], com_vel=np.zeros(3),
                         yaw=meas["yaw"], yaw_rate=0.0,
                         support=np.array([True, True]),
                         tol_point=np.full(5, cfg.tol_default))
            win = [r] * cfg.horizon

        init = self.prev_seq if self.prev_seq is not None \
            else np.tile(self.prev_action, (cfg.horizon, 1))
        snap = snapshot or capture(self.wsim)
        sol = search_window(self.wsim, snap, self.robot_id,
                            self.other_id, win, init,
                            self.prev_action, cfg,
                            seed=cfg.seed + len(self.vlog))
        a = np.clip(sol["action"], -cfg.action_clip, cfg.action_clip)
        seq = sol["seq"]
        self.prev_seq = np.vstack([seq[1:], seq[-1:]])
        self.prev_action = a
        diag = {"score": sol["score"], "plan_t": self.plan_t,
                "event_idx": self.event_idx, "replans": self.replans,
                "tilt": meas["tilt"], "com_vel": meas["com_vel"].tolist()}
        self.vlog.append(diag)
        # safety
        if meas["tilt"] > cfg.tilt_fail or \
                meas["head_z"] < cfg.head_min_z:
            self.status = "FAILED"
            return StepResult(a, "FAILED",
                              {**diag, "reason": "fallen"})
        if self.t_submit is not None and \
                current["time"] - self.t_submit > cfg.task_budget_s \
                and self.event_idx < len(self.clip.keyframes):
            self.status = "FAILED"
            return StepResult(a, "FAILED",
                              {**diag, "reason": "budget"})
        return StepResult(a, self.status, diag)
