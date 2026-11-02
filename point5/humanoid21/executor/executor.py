"""E6: closed-loop Executor — cache + progress + MPC step."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from point5.contracts import P5Clip
from .clip_cache import InstructionCache
from .contracts import (ExecStatus, ExecutorConfig, RefFrame,
                        StepResult)
from .reference import RefPlan, plan_reference, window
from .search import search_window
from .sim_iface import load_interface
from .state import (Current, fall_check, snapshot)
from .contracts import ExecutorError


class Executor:
    """Servo: validated clip + measured current -> next action.

    wsim is the executor's private working simulator used for MPC
    rollouts; it is synced to the caller's sim every step (the caller's
    sim is the ground truth and is never written by the executor).
    """

    def __init__(self, wsim, cfg: Optional[ExecutorConfig] = None,
                 robot_id: str = "robot_a"):
        self.cfg = cfg or ExecutorConfig()
        self.wsim = wsim
        self.robot_id = robot_id
        self.other_id = ("robot_b" if robot_id == "robot_a"
                         else "robot_a")
        self.iface = load_interface(wsim)
        self.cache = InstructionCache(self.cfg.anchor_atol)
        self.status = ExecStatus.HOLDING
        self.plan: Optional[RefPlan] = None
        self.plan_t = 0.0
        self.replans = 0
        self.prev_action = self.iface.standing_action.astype(np.float32)
        self.prev_seq = np.tile(self.prev_action, (self.cfg.horizon, 1))
        self.prev_p5w: Optional[np.ndarray] = None
        self.pose_actions: list = []
        self.t_submit: Optional[float] = None
        self.settle_mark: Optional[float] = None
        self.events: List[Dict[str, Any]] = []
        self.diag_log: List[Dict[str, Any]] = []

    # -- instruction ops -----------------------------------------------------
    def submit(self, clip: P5Clip, cur: Current,
               meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Anchor-check then plan; failure leaves old state untouched."""
        sub = self.cache.submit(clip, cur.p5)      # raises AnchorError
        plan = plan_reference(clip, cur.frame_pose, cur.com_pos,
                              cur.com_vel, self.cfg)
        # IK guide rail: one action per keyframe (None on failure)
        from .scoring import pose_init
        self.pose_actions = [
            pose_init(k.pos, self.iface) for k in clip.keyframes]
        self.plan = plan
        self.plan_t = 0.0
        self.replans = 0
        self.t_submit = cur.time
        self.status = ExecStatus.RUNNING
        self.settle_mark = None
        self.events = [{"t": 0.0, "event": "submitted"}]
        return {"accepted": True, "duration_s": plan.total_s,
                "n_frames": len(plan.frames)}

    def clear(self) -> None:
        self.cache.clear()
        self.plan = None
        self.status = ExecStatus.SETTLING
        self.settle_mark = None

    # -- progress ------------------------------------------------------------
    def _progress(self, cur: Current) -> None:
        """Confirm pending keyframe if truly reached; else maybe replan."""
        kf = self.cache.next_kf
        if self.cache.sub is None or kf >= len(
                self.cache.sub.clip.keyframes):
            return
        due = dict(self.plan.key_times).get(kf)
        tgt = self.cache.sub.world_pts[kf]
        err = float(np.linalg.norm(cur.p5_world - tgt, axis=1).max())
        if err <= self.cfg.kf_reach_tol:
            self.cache.confirm_keyframe()
            self.events.append({"t": self.plan_t, "event": "reached",
                                "idx": kf, "err": err,
                                "label": self.cache.sub.clip
                                .keyframes[kf].label})
        elif due is not None and \
                self.plan_t > due + self.cfg.replan_wait_s:
            if self.replans < self.cfg.max_replans:
                self.replans += 1
                self.events.append({"t": self.plan_t,
                                    "event": "replan", "idx": kf})
            else:
                self.status = ExecStatus.FAILED
                self.events.append({"t": self.plan_t,
                                    "event": "failed",
                                    "reason": "keyframe_missed",
                                    "idx": kf, "err": err})

    def _hold_ref(self, cur: Current) -> List[RefFrame]:
        r = RefFrame(t=self.plan_t, p5_world=cur.p5_world,
                     p5_vel_world=np.zeros((5, 3)),
                     com_pos=cur.com_pos, com_vel=np.zeros(3),
                     yaw=np.deg2rad(cur.frame_pose[1]), yaw_rate=0.0,
                     support=np.array([True, True]),
                     tol_point=np.full(5, self.cfg.tol_default))
        return [r] * self.cfg.horizon

    # -- main step -----------------------------------------------------------
    def step(self, cur: Current) -> StepResult:
        """Compute the action for this control period.

        Caller then applies it once to the REAL sim.
        """
        cfg = self.cfg
        if self.status == ExecStatus.FAILED:
            return StepResult(self.prev_action.copy(),
                              ExecStatus.FAILED)
        reason = fall_check(cur, cfg)
        if reason:
            self.status = ExecStatus.FAILED
            self.events.append({"t": self.plan_t, "event": "failed",
                                "reason": reason})
            return StepResult(self.prev_action.copy(),
                              ExecStatus.FAILED, {"reason": reason})
        if self.status in (ExecStatus.RUNNING, ExecStatus.SETTLING):
            # progress-gated clock: the schedule only advances while the
            # robot is actually near its current reference.  Large lag
            # freezes plan_t instead of letting the ref run away; a
            # stalled plan still hits the task budget -> honest FAIL.
            self.plan_t += cfg.dt
        if self.cache.sub is not None:
            self._progress(cur)
        # task timeout
        if (self.t_submit is not None
                and cur.time - self.t_submit > cfg.task_budget_s
                and self.cache.sub is not None
                and self.cache.next_kf < len(
                    self.cache.sub.clip.keyframes)):
            self.status = ExecStatus.FAILED
            return StepResult(self.prev_action.copy(),
                              ExecStatus.FAILED,
                              {"reason": "budget"})
        # settle/hold transitions
        if (self.cache.sub is None
                or self.cache.next_kf >= len(
                    self.cache.sub.clip.keyframes)):
            if self.status == ExecStatus.RUNNING:
                self.status = ExecStatus.SETTLING
                self.events.append({"t": self.plan_t,
                                    "event": "clip_done"})
            if float(np.linalg.norm(cur.com_vel)) < cfg.hold_com_vel:
                self.settle_mark = (self.settle_mark
                                    if self.settle_mark is not None
                                    else self.plan_t)
                if self.plan_t - self.settle_mark > 0.5:
                    self.status = ExecStatus.HOLDING
            else:
                self.settle_mark = None

        # internal re-anchor: current actual pose drives the remaining
        # world-anchored plan forward (caller does not re-submit)
        self.cache.update_head(cur.p5)

        if self.plan is not None:
            # adaptive horizon: cover the next keyframe + 2-step buffer,
            # clamped to [4, horizon_max]
            due = dict(self.plan.key_times).get(
                self.cache.next_kf, self.plan_t + cfg.horizon * cfg.dt)
            need = int(np.ceil(max(due - self.plan_t, 0.0)
                               / cfg.dt)) + 2
            h = int(np.clip(need, 4, cfg.horizon))
            refs = window(self.plan, self.plan_t, h, cfg)
        else:
            refs = self._hold_ref(cur)

        if cur.snapshot is None:
            raise ExecutorError(
                "step() needs Current captured with snapshot=True")
        snap = cur.snapshot
        # free joints = support-leg ankle/hip + abdomen (balance DOF);
        # both legs when double support, planted leg when single.
        from .reflex import ANKLE_Y, ANKLE_X, HIP_X, HIP_Y, ABDOMEN_X
        sup = refs[0].support if refs else np.array([True, True])
        free = [ABDOMEN_X]
        for side, on in (("right", sup[0]), ("left", sup[1])):
            if on:
                free += [ANKLE_Y[side], ANKLE_X[side],
                         HIP_X[side], HIP_Y[side]]
        # during a mover segment the swinging leg's joints are also
        # free — the foot target cannot be reached if the leg is
        # frozen to an imperfect IK guide
        mv = refs[0].mover if refs else -1
        if mv >= 0:
            from point5.contracts import POINT_INDEX
            mside = ("right" if mv == POINT_INDEX["foot_right"]
                     else "left")
            from .reflex import HIP_X as HX, HIP_Y as HY
            free += [ {"right": 3, "left": 9}[mside],
                      {"right": 4, "left": 10}[mside],   # hip_z
                      {"right": 5, "left": 11}[mside],   # hip_y
                      {"right": 6, "left": 12}[mside],   # knee
                      {"right": 7, "left": 13}[mside],   # ankle_y
                      {"right": 8, "left": 14}[mside] ]  # ankle_x
        free_idx = np.array(sorted(set(free)))
        # guide rail: for each window ref, blend prev action -> the
        # IK action of that ref's target keyframe by segment progress
        guide = None
        if self.pose_actions:
            rows = []
            for r in refs:
                kf = min(r.seg + 1, len(self.pose_actions) - 1) \
                    if r.seg >= 0 else 0
                tgt = self.pose_actions[kf]
                if tgt is None:
                    tgt = self.prev_action
                u = float(np.clip(getattr(r, "s", 0.0) or 0.0, 0, 1))
                rows.append((1 - u) * np.asarray(self.prev_action)
                            + u * np.asarray(tgt))
            guide = np.clip(np.stack(rows), -1, 1)
        sol = search_window(
            self.wsim, snap, refs, self.prev_seq, self.prev_action,
            self.iface.standing_action, self.robot_id, self.other_id,
            cfg, seed=cfg.seed + len(self.diag_log), guide_seq=guide,
            free_idx=free_idx)
        a = np.clip(sol["action"], -cfg.action_clip, cfg.action_clip)
        self.prev_seq = np.vstack([sol["seq"][1:], sol["seq"][-1:]])
        self.prev_action = a
        self.prev_p5w = cur.p5_world
        diag = {"score": sol["score"], "plan_t": self.plan_t,
                "next_kf": self.cache.next_kf,
                "tilt": cur.torso_tilt,
                "com_vel": float(np.linalg.norm(cur.com_vel))}
        self.diag_log.append(diag)
        return StepResult(a, self.status, diag)



