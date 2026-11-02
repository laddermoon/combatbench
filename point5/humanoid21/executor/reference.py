"""E3: big-logic reference planner — support, timing, RefFrames.

Deterministic, closed-form.  Timing comes from displacement vs
capability caps over a quintic blend (max |s'|=1.875, |s''|~3.33);
a crude ground-force check retimes longer up to ``max_retries``.
Fails loud with the limiting cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from point5.contracts import POINT_INDEX, P5Clip, to_world
from point5.mass_model import mass_proxy
from point5.trajectory import FramePose, keyframe_frames
from .contracts import ExecutorConfig, PlanError, RefFrame

FOOT = (POINT_INDEX["foot_right"], POINT_INDEX["foot_left"])
HAND = (POINT_INDEX["hand_right"], POINT_INDEX["hand_left"])
_S1, _S2 = 1.875, 3.333      # max quintic blend derivatives


def smooth(s: float) -> Tuple[float, float, float]:
    return (6 * s**5 - 15 * s**4 + 10 * s**3,
            30 * s**4 - 60 * s**3 + 30 * s**2,
            120 * s**3 - 180 * s**2 + 60 * s)


@dataclass
class Segment:
    i: int                            # clip segment kf[i] -> kf[i+1]
    dur: float
    support: np.ndarray               # (2,) bool right,left
    mover: Optional[int]              # POINT_INDEX of swinging foot
    limiter: str = ""


def segment_support(p0: np.ndarray, p1: np.ndarray
                    ) -> Tuple[np.ndarray, Optional[int]]:
    """A foot with real XY or Z displacement is swinging."""
    d = p1 - p0
    mover = None
    for i in FOOT:
        if np.linalg.norm(d[i]) > 0.03 or abs(d[i, 2]) > 0.02:
            mover = i
    if mover is None:
        return np.array([True, True]), None
    sup = np.array([True, True])          # [right, left]
    if mover == FOOT[0]:
        sup[0] = False
    else:
        sup[1] = False
    return sup, mover


def needed_duration(dp5: np.ndarray, dcom: np.ndarray, dyaw_deg: float,
                    landing: bool, cfg: ExecutorConfig
                    ) -> Tuple[float, str]:
    """Min duration for one segment from capability caps."""
    d = np.linalg.norm(dp5, axis=1)
    com = float(np.linalg.norm(dcom))
    yaw = abs(np.deg2rad(dyaw_deg))
    req = {
        "hand": max(_S1 * max(d[HAND[0]], d[HAND[1]]) / cfg.v_hand,
                    np.sqrt(_S2 * max(d[HAND[0]], d[HAND[1]])
                            / cfg.a_hand)),
        "head": max(_S1 * d[0] / cfg.v_head,
                    np.sqrt(_S2 * d[0] / cfg.a_head)),
        "foot": max(_S1 * max(d[FOOT[0]], d[FOOT[1]]) / cfg.v_foot,
                    np.sqrt(_S2 * max(d[FOOT[0]], d[FOOT[1]])
                            / cfg.a_foot)),
        "com": max(_S1 * com / cfg.v_com,
                   np.sqrt(_S2 * com / cfg.a_com)),
        "yaw": _S1 * yaw / cfg.w_yaw,
        "min": cfg.min_seg_dur,
    }
    lim, T = max(req.items(), key=lambda kv: kv[1])
    if landing:
        T = max(T, cfg.land_settle)
        lim = "land_settle" if T == cfg.land_settle else lim
    return T, lim


@dataclass
class RefPlan:
    frames: List[RefFrame]
    segments: List[Segment]
    key_times: List[Tuple[int, float]]   # (kf_idx, time)
    com_world: np.ndarray                # (K,3) per keyframe
    total_s: float


def plan_reference(clip: P5Clip, f1: FramePose,
                   com0: np.ndarray, vel0: np.ndarray,
                   cfg: ExecutorConfig) -> RefPlan:
    """Timed reference for a submitted clip (already anchored)."""
    K = len(clip.keyframes)
    wf = keyframe_frames(clip, f1)
    wpts = np.stack([to_world(k.pos, *wf[i])
                     for i, k in enumerate(clip.keyframes)])
    coms = np.stack([to_world(
        mass_proxy(k.pos).com.reshape(1, 3), *wf[i])[0]
        for i, k in enumerate(clip.keyframes)])
    coms = coms.copy()
    coms[0] = np.asarray(com0, np.float64)
    tols = np.full((K, 5), cfg.tol_default)
    for i, kf in enumerate(clip.keyframes):
        for name, idx in POINT_INDEX.items():
            spec = (kf.meta or {}).get("specs", {}).get(name)
            if spec and "tol" in spec:
                tols[i, idx] = float(spec["tol"])

    scale, limiter = 1.0, ""
    segs: List[Segment] = []
    for attempt in range(cfg.max_retries + 1):
        segs = []
        feasible = True
        for k in range(K - 1):
            sup, mv = segment_support(wpts[k], wpts[k + 1])
            tr = clip.keyframes[k + 1].transfer
            T, lim = needed_duration(
                wpts[k + 1] - wpts[k], coms[k + 1] - coms[k],
                tr.dyaw, mv is not None, cfg)
            T *= scale
            # crude force check: required peak COM accel vs gravity
            a_req = _S2 * np.linalg.norm(
                coms[k + 1] - coms[k]) / (T * T)
            if a_req > 9.81 * 1.5:
                feasible, limiter = False, "com_accel"
            segs.append(Segment(k, T, sup, mv, lim))
        if feasible:
            break
        scale *= cfg.duration_growth
    if not feasible:
        raise PlanError(f"timing infeasible ({limiter})")
    if sum(s.dur for s in segs) > cfg.task_budget_s:
        raise PlanError(
            f"plan needs {sum(s.dur for s in segs):.2f}s > "
            f"budget {cfg.task_budget_s}s")

    times = np.concatenate([[0.0], np.cumsum([s.dur for s in segs])])
    key_times = [(k + 1, float(times[k + 1])) for k in range(K - 1)]
    frames: List[RefFrame] = []
    n = int(np.ceil((times[-1] - 1e-9) / cfg.dt)) + 1
    for i in range(n):
        t = min(i * cfg.dt, times[-1])
        seg_i = int(min(np.searchsorted(times, t, side="right") - 1,
                        len(segs) - 1))
        seg = segs[seg_i]
        T = max(seg.dur, 1e-9)
        s = min((t - times[seg_i]) / T, 1.0)
        u, du, _ = smooth(s)
        k0, k1 = clip.keyframes[seg_i], clip.keyframes[seg_i + 1]
        tr = k1.transfer
        loc = k0.pos + (k1.pos - k0.pos) * u
        loc_v = (k1.pos - k0.pos) * (du / T)
        o0, y0 = wf[seg_i]
        if tr is None:
            o_s, y_s = np.asarray(o0, dtype=np.float64), float(y0)
            o_dot, y_dot = np.zeros(2), 0.0
        else:
            do = np.array([tr.dx, tr.dy]) * u
            r0 = np.deg2rad(y0)
            c0, s0 = np.cos(r0), np.sin(r0)
            R0 = np.array([[c0, -s0], [s0, c0]])
            o_s = np.asarray(o0, dtype=np.float64) + R0 @ do
            dod = np.array([tr.dx, tr.dy]) * (du / T)
            o_dot = R0 @ dod
            y_s = y0 + tr.dyaw * u
            y_dot = np.deg2rad(tr.dyaw) * (du / T)
        pw = to_world(loc, o_s, y_s)
        r = np.deg2rad(y_s)
        R = np.array([[np.cos(r), -np.sin(r)],
                      [np.sin(r), np.cos(r)]])
        rel = (R @ loc[:, :2].T).T
        pv = np.zeros((5, 3))
        pv[:, :2] = loc_v[:, :2] @ R.T + o_dot \
            + y_dot * np.stack([-rel[:, 1], rel[:, 0]], axis=1)
        pv[:, 2] = loc_v[:, 2]
        cw = coms[seg_i] + (coms[seg_i + 1] - coms[seg_i]) * u
        cv = (coms[seg_i + 1] - coms[seg_i]) * (du / T)
        frames.append(RefFrame(
            t=float(t), p5_world=pw, p5_vel_world=pv,
            com_pos=cw, com_vel=cv,
            yaw=float(np.deg2rad(y_s)),
            yaw_rate=float(y_dot),
            support=seg.support,
            tol_point=tols[min(seg_i + 1, K - 1)],
            seg=seg_i))
    # mark keyframe arrivals on nearest frames
    for kf_i, tk in key_times:
        j = int(np.argmin([abs(fr.t - tk) for fr in frames]))
        frames[j].event = clip.keyframes[kf_i].label
        frames[j].kf_idx = kf_i
    return RefPlan(frames=frames, segments=segs, key_times=key_times,
                   com_world=coms, total_s=float(times[-1]))


def window(plan: RefPlan, t0: float, h: int,
           cfg: ExecutorConfig) -> List[RefFrame]:
    """H references at t0+dt..t0+h*dt; past the end, hold the last."""
    ts = [r.t for r in plan.frames]
    out = []
    for k in range(h):
        tt = t0 + (k + 1) * cfg.dt
        j = int(np.clip(np.searchsorted(ts, tt, side="right") - 1,
                        0, len(plan.frames) - 1))
        out.append(plan.frames[j])
    return out
