"""Big-logic reference planning: support phases, timing, RefFrames."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from point5.contracts import POINT_INDEX, P5Clip, Transfer, Keyframe
from point5.mass_model import mass_proxy
from point5.trajectory import keyframe_frames, keyframe_world, FramePose
from point5.contracts import to_world, to_local, compose_frame
from .contracts import (
    ExecutorConfig, PlanError, RefFrame)

UP = np.array([0.0, 0.0, 1.0])


def build_stand_clip(plan_clip: P5Clip,
                     start_local: np.ndarray) -> P5Clip:
    """Caller-side helper: prepend the measured standing pose as frame 0.

    stand_punch's coil keeps origin+yaw -> identity transfer.
    """
    first = plan_clip.keyframes[0]
    kf0 = Keyframe(pos=np.asarray(start_local, dtype=np.float64),
                   label="start")
    kf1 = Keyframe(pos=first.pos, transfer=Transfer(
        dx=0.0, dy=0.0, dyaw=0.0, dur=0.1),
        label=first.label, meta=dict(first.meta))
    return P5Clip(name=plan_clip.name,
                  keyframes=(kf0, kf1) + plan_clip.keyframes[1:],
                  metadata=dict(plan_clip.metadata))


def _seg_support(p0: np.ndarray, p1: np.ndarray,
                 cfg: ExecutorConfig) -> Tuple[np.ndarray, Optional[str]]:
    """Per-segment support [right,left] + moving foot name or None.

    A foot with noticeable XY displacement or z lift is swinging; the
    other foot must stay planted (else unsupported plan).
    """
    d = p1 - p0
    moving = None
    for i, name in ((POINT_INDEX["foot_right"], "right"),
                    (POINT_INDEX["foot_left"], "left")):
        if np.linalg.norm(d[i]) > 0.03 or abs(d[i, 2]) > 0.02:
            moving = name
    if moving is None:
        return np.array([True, True]), None
    return np.array([moving != "right", moving != "left"]), moving


def _smooth(s: float) -> Tuple[float, float, float]:
    """Quintic blend and first/second derivatives w.r.t. s."""
    return (6 * s**5 - 15 * s**4 + 10 * s**3,
            30 * s**4 - 60 * s**3 + 30 * s**2,
            120 * s**3 - 180 * s**2 + 60 * s)


_MAX_S1 = 1.875          # max smooth'(s)
_MAX_S2 = 3.333          # approx max smooth''(s)


def _needed_dur(dp5: np.ndarray, dc: np.ndarray, dyaw: float,
                landing: bool, cfg: ExecutorConfig) -> Tuple[float, str]:
    """Minimum segment duration from displacement + capability caps."""
    hands = max(np.linalg.norm(dp5[1]), np.linalg.norm(dp5[2]))
    head = float(np.linalg.norm(dp5[0]))
    feet = max(np.linalg.norm(dp5[3]), np.linalg.norm(dp5[4]))
    com = float(np.linalg.norm(dc))
    yaw = abs(np.deg2rad(dyaw))
    req = {
        "hand": max(_MAX_S1 * hands / cfg.v_hand,
                    np.sqrt(_MAX_S2 * hands / cfg.a_hand)),
        "head": max(_MAX_S1 * head / cfg.v_head,
                    np.sqrt(_MAX_S2 * head / cfg.a_head)),
        "foot": max(_MAX_S1 * feet / cfg.v_foot,
                    np.sqrt(_MAX_S2 * feet / cfg.a_foot)),
        "com": max(_MAX_S1 * com / cfg.v_com,
                   np.sqrt(_MAX_S2 * com / cfg.a_com)),
        "yaw": _MAX_S1 * yaw / cfg.w_yaw,
        "min": cfg.min_seg_dur,
    }
    name, T = max(req.items(), key=lambda kv: kv[1])
    if landing:
        T = max(T, cfg.land_settle)
    return T, name


def plan_reference(clip: P5Clip, f1: FramePose,
                   com0: np.ndarray, vel0: np.ndarray,
                   cfg: ExecutorConfig) -> List[RefFrame]:
    """Build the full timed reference for one submitted clip.

    Deterministic; retiming multiplies durations by 1.25 up to
    cfg.max_retries, then raises PlanError with the limiting cause.
    """
    wf = keyframe_frames(clip, f1)
    wpts = np.stack([to_world(k.pos, o, y)
                     for k, (o, y) in zip(clip.keyframes, wf)])
    lcs = np.stack([mass_proxy(k.pos).com for k in clip.keyframes])
    coms = np.stack([to_world(lc.reshape(1, 3), o, y)[0]
                     for lc, (o, y) in zip(lcs, wf)])
    coms = coms.copy()
    coms[0] = np.asarray(com0, dtype=np.float64)
    tols = np.full((len(clip.keyframes), 5), cfg.tol_default)
    for ki, kf in enumerate(clip.keyframes):
        specs = (kf.meta or {}).get("specs") or {}
        for pname, idx in POINT_INDEX.items():
            if pname in specs and "tol" in specs[pname]:
                tols[ki, idx] = float(specs[pname]["tol"])

    scale = 1.0
    limiter = ""
    for attempt in range(cfg.max_retries + 1):
        durs = []
        supports = []
        movers = []
        limiter = ""
        for k in range(len(clip.keyframes) - 1):
            sup, mv = _seg_support(wpts[k], wpts[k + 1], cfg)
            tr = clip.keyframes[k + 1].transfer
            landing = (mv is not None and sup.sum() == 1)
            T, lim = _needed_dur(wpts[k + 1] - wpts[k],
                                 coms[k + 1] - coms[k],
                                 tr.dyaw, landing, cfg)
            durs.append(T * scale)
            supports.append(sup)
            movers.append(mv)
            if lim in ("hand", "foot", "com"):
                limiter = lim
        # crude support-force check at segment peak accelerations
        ok = True
        for k, T in enumerate(durs):
            dc = coms[k + 1] - coms[k]
            a_req = _MAX_S2 * np.linalg.norm(dc) / (T * T)
            fz = cfg.total_mass * (a_req - (-9.81))
            fxy = 0.0
            if fz <= 0.0:
                ok, limiter = False, "support_normal"
        if ok:
            break
        scale *= cfg.max_duration_scale
    else:
        raise PlanError(
            f"timing infeasible after {cfg.max_retries} retries "
            f"(limiter: {limiter})")
    if not ok:
        raise PlanError(f"support check failed ({limiter})")

    times = np.concatenate([[0.0], np.cumsum(durs)])
    frames: List[RefFrame] = []
    n_out = int(np.ceil((times[-1] - 1e-9) / cfg.dt)) + 1
    for i in range(n_out):
        t = float(min(i * cfg.dt, times[-1]))
        seg = int(np.searchsorted(times, t, side="right") - 1)
        seg = min(seg, len(durs) - 1)
        T = durs[seg]
        s = 1.0 if T <= 0 else min((t - times[seg]) / T, 1.0)
        u, du, ddu = _smooth(s)
        k0 = clip.keyframes[seg]
        k1 = clip.keyframes[seg + 1] if seg + 1 < len(clip.keyframes) \
            else clip.keyframes[seg]
        tr = k1.transfer
        loc = k0.pos + (k1.pos - k0.pos) * u
        loc_v = (k1.pos - k0.pos) * (du / max(T, 1e-9))
        o0, y0 = wf[seg]
        if tr is None:
            o_s, y_s, o_dot, y_dot = np.asarray(o0), y0, np.zeros(2), 0.0
        else:
            do = np.array([tr.dx, tr.dy]) * u
            r0 = np.deg2rad(y0)
            c0, s0 = np.cos(r0), np.sin(r0)
            o_s = o0 + np.array([c0 * do[0] - s0 * do[1],
                                 s0 * do[0] + c0 * do[1]])
            dod = np.array([tr.dx, tr.dy]) * (du / max(T, 1e-9))
            o_dot = np.array([c0 * dod[0] - s0 * dod[1],
                              s0 * dod[0] + c0 * dod[1]])
            y_s = y0 + tr.dyaw * u
            y_dot = tr.dyaw * (du / max(T, 1e-9))
        pw = to_world(loc, np.asarray(o_s), y_s)
        # world point velocity: frame motion + local velocity
        r = np.deg2rad(y_s)
        c_, s_ = np.cos(r), np.sin(r)
        R = np.array([[c_, -s_], [s_, c_]])
        pv = np.zeros((5, 3))
        pv[:, :2] = loc_v[:, :2] @ R.T
        pv[:, 2] = loc_v[:, 2]
        rel = (R @ (loc[:, :2]).T).T          # rotated local xy
        pv[:, 0] += -np.deg2rad(y_dot) * rel[:, 1]
        pv[:, 1] += np.deg2rad(y_dot) * rel[:, 0]
        pv[:, :2] += o_dot
        cw = coms[seg] + (coms[seg + 1] - coms[seg]) * u \
            if seg + 1 < len(clip.keyframes) else coms[seg]
        cv = (coms[seg + 1] - coms[seg]) * (du / max(T, 1e-9)) \
            if seg + 1 < len(clip.keyframes) else np.zeros(3)
        ev = ""  # assigned in post-pass below
        frames.append(RefFrame(
            t=t, p5_world=pw, p5_vel_world=pv, com_pos=cw, com_vel=cv,
            yaw=float(np.deg2rad(y_s)),
            yaw_rate=float(np.deg2rad(y_dot)),
            support=supports[seg] if seg < len(supports)
            else supports[-1],
            tol_point=tols[min(seg + 1, len(clip.keyframes) - 1)],
            seg=seg, s=s, event=ev))
    # mark each keyframe arrival on the nearest frame to its boundary
    for k in range(1, len(clip.keyframes)):
        j = int(np.argmin([abs(fr.t - times[k]) for fr in frames]))
        frames[j].event = clip.keyframes[k].label
        frames[j].kf_idx = k
    return frames
