"""Standing punch planner — strike from the standard standing pose.

``plan_stand_punch(target, hand, cfg, initial=None) -> PunchPlan``

Two phases only, always lead-hand (jab) mechanics:

    start(standing) -> load(发力姿态) -> strike(出拳)

  Phase 1 ``load`` — the body coils WITHOUT stepping: head sinks a few
      centimetres, weight loads, the punching hand chambers to the guard
      line (elbow back, fist at chin height on its own side), the torso
      pre-twists slightly AWAY from the punch side (counter-rotation to
      wind up).  Feet never move.
  Phase 2 ``strike`` — the punch hand drives straight to
      ``target + punch_dir * penetration``, the torso snaps the wind-up
      into a small forward twist (jab = modest, not a cross), the head
      leans a touch forward.  Guard hand tucks to the chin.

``initial`` defaults to the standard standing local pose
(:func:`stand_p5`).  ``target`` is expressed in the initial local frame
(torso-facing, ankle-mid origin) — the same convention as every v2
consumer.  Verdicts: ``out_of_sector`` (target far outside the frontal
cone), ``out_of_reach`` (beyond jab reach — no stepping exists here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from ..contracts import (
    POINT_INDEX,
    Keyframe,
    P5Clip,
    Point5State,
    Transfer,
    ankle_mid_xy,
    to_local,
)

_UP = np.array([0.0, 0.0, 1.0])
HAND_POINT = {"left": "hand_left", "right": "hand_right"}
FOOT_POINT = {"left": "foot_left", "right": "foot_right"}


@lru_cache(maxsize=1)
def stand_p5() -> np.ndarray:
    """Standard standing pose as v2 local coordinates (5,3).

    Landmarks of the unloaded skeleton with the root height bisected so
    the ankles rest at their planted height — the same reset pose the
    arena spawns, expressed in the torso-facing local frame (the
    skeleton's own frame already faces +X).
    """
    from .mechanics import Skeleton
    sk = Skeleton()
    lo, hi = 0.2, 1.2
    for _ in range(25):
        m = (lo + hi) / 2
        lm = sk.landmarks([0, 0, m], [1, 0, 0, 0], np.zeros(21))
        if lm[3:, 2].min() < 0.087:
            lo = m
        else:
            hi = m
    return sk.landmarks([0, 0, (lo + hi) / 2], [1, 0, 0, 0],
                        np.zeros(21))


@dataclass(frozen=True)
class StandPunchConfig:
    """Tunables for the standing jab planner."""

    # --- reach / sector ---
    arm_reach: float = 0.60       # fist travel from guard to impact (m)
    sector_cos_min: float = 0.17  # target within ~80° of +X forward
    penetration: float = 0.08     # overshoot past the target (m)

    # --- step (重心转移) for far targets ---
    max_step: float = 0.40        # lead-foot advance cap (m)
    head_step_share: float = 0.6  # head follows this fraction of step
    foot_swing_clearance: float = 0.06
    step_sink_extra: float = 0.05  # extra COM sink driven by the step (m)

    # --- load (发力姿态) ---
    head_sink_load: float = 0.05  # crouch while coiling (m)
    chamber_back: float = 0.10    # fist pulled behind head along -X (m)
    chamber_side: float = 0.12    # lateral offset to the punch side (m)
    chamber_drop: float = 0.18    # below head (m)
    coil_deg: float = 8.0         # counter-rotation wind-up (deg)
    guard_ride: float = 1.0       # guard hand follows the body shift

    # --- strike ---
    jab_twist_deg: float = 15.0   # torso twist at impact (toward punch)
    head_lean_strike: float = 0.05
    head_sink_extra: float = 0.03  # additional sink on impact (m)
    guard_tuck_back: float = 0.05
    guard_tuck_side: float = 0.10
    guard_tuck_drop: float = 0.12

    # --- timing ---
    dur_load: float = 0.22        # slower coil
    dur_strike: float = 0.12      # fast snap

    # --- downstream scoring annotations ---
    w_punch: float = 3.0
    tol_punch: float = 0.03
    w_guard: float = 1.0
    tol_guard: float = 0.08
    w_head: float = 0.5
    tol_head: float = 0.10
    w_feet: float = 1.5
    tol_feet: float = 0.05


@dataclass
class PunchPlan:
    """Planning verdict.  ``task is None`` iff unreachable."""

    task: Optional[P5Clip]
    reachable: bool
    reason: str = ""
    diag: Dict[str, object] = field(default_factory=dict)


def _xy(v):
    v = np.asarray(v, dtype=np.float64).copy()
    v[2] = 0.0
    return v


def _norm(v, eps=1e-9):
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def _rotz(v, deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def plan_stand_punch(target: np.ndarray,
                     hand: str,
                     cfg: StandPunchConfig = StandPunchConfig(),
                     initial: Optional[Point5State] = None,
                     ) -> PunchPlan:
    """Plan a standing jab: standing -> load -> strike."""
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if hand not in HAND_POINT:
        raise ValueError(f"hand must be 'left'/'right', got {hand!r}")
    pos = (initial.pos if initial is not None else stand_p5())
    pos = np.asarray(pos, dtype=np.float64)
    diag: Dict[str, object] = {}

    hp_name = HAND_POINT[hand]
    hg_name = "hand_right" if hand == "left" else "hand_left"
    i_hp, i_hg = POINT_INDEX[hp_name], POINT_INDEX[hg_name]
    i_fr, i_fl = POINT_INDEX["foot_right"], POINT_INDEX["foot_left"]
    head, hp, hg = pos[0], pos[i_hp], pos[i_hg]

    # --- attack axis + frontal sector ----------------------------------
    a = _norm(_xy(target - head))
    if not np.any(a):
        a = np.array([1.0, 0.0, 0.0])
    diag["attack_axis"] = a
    cos_af = float(a[0])                    # local frame X = forward
    diag["cos_attack_vs_fwd"] = cos_af
    if cos_af < cfg.sector_cos_min:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_sector", diag=diag)

    # --- reach -> step length (重心转移: lead foot steps, COM follows) --
    gap = float(np.linalg.norm(target - hp)) - cfg.arm_reach
    diag["reach_gap"] = gap
    step_need = max(gap, 0.0) / cfg.head_step_share
    if step_need > cfg.max_step:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_reach", diag=diag)
    step_len = min(step_need, cfg.max_step)
    diag["step_len"] = step_len

    # lead foot = the one ahead along the attack axis; it steps
    proj = {i_fr: float((pos[i_fr] - head) @ a),
            i_fl: float((pos[i_fl] - head) @ a)}
    move_i = max(proj, key=proj.get)
    anchor_i = i_fl if move_i == i_fr else i_fr
    move_name = ("foot_right" if move_i == i_fr else "foot_left")
    diag["moving_foot"] = move_name
    diag["anchor_foot"] = ("foot_right" if anchor_i == i_fr
                           else "foot_left")
    f_move = pos[move_i].copy()
    f_anchor = pos[anchor_i]
    f_landed = f_move + a * step_len

    sgn = 1.0 if hand == "right" else -1.0  # + = toward the right side
    right = np.array([a[1], -a[0], 0.0])

    # --- phase 1: load (发力姿态: 迈步 + 降重心 + 重心转移) --------------
    body_shift = (a * (cfg.head_step_share * step_len)
                  + np.array([0, 0, -(cfg.head_sink_load
                                     + cfg.step_sink_extra
                                     * (step_len / cfg.max_step))]))
    head_load = head + body_shift
    hp_load = (head_load
               - a * cfg.chamber_back
               + right * (sgn * cfg.chamber_side))
    hp_load[2] = head_load[2] - cfg.chamber_drop
    hg_load = hg + body_shift * cfg.guard_ride
    feet_load = {i_fr: pos[i_fr].copy(), i_fl: pos[i_fl].copy()}
    feet_load[move_i] = f_landed                          # lead foot lands
    feet_load[anchor_i] = f_anchor                        # anchor planted
    t_load = _rotz(np.array([1.0, 0.0, 0.0]),
                   sgn * cfg.coil_deg)                    # counter-rotation

    # --- phase 2: strike ------------------------------------------------
    punch_dir = _norm(target - hp_load)
    if not np.any(punch_dir):
        punch_dir = a
    hp_hit = target + punch_dir * cfg.penetration
    head_hit = (head_load + a * cfg.head_lean_strike
                + np.array([0, 0, -cfg.head_sink_extra]))
    hg_hit = (head_hit - a * cfg.guard_tuck_back
              + right * ((-sgn) * cfg.guard_tuck_side)
              + np.array([0, 0, -cfg.guard_tuck_drop]))
    t_hit = _rotz(np.array([1.0, 0.0, 0.0]),
                  -sgn * cfg.jab_twist_deg)             # snap INTO punch
    diag["strike_twist_deg"] = -sgn * cfg.jab_twist_deg

    # --- keyframes: re-express each in its own local frame --------------
    t0_dir = np.array([1.0, 0.0, 0.0])
    rows = [
        ("start",  hp,       hg,      head,      t0_dir, 0.0),
        ("load",   hp_load,  hg_load, head_load, t_load, cfg.dur_load),
        ("strike", hp_hit,   hg_hit,  head_hit,  t_hit, cfg.dur_strike),
    ]
    kfs, prev_o, prev_yaw = [], None, None
    for label, hp_t, hg_t, head_t, t_dir, dur in rows:
        pos5 = np.empty((5, 3))
        pos5[POINT_INDEX["head"]] = head_t
        pos5[i_hp] = hp_t
        pos5[i_hg] = hg_t
        if label == "start":
            pos5[i_fr], pos5[i_fl] = pos[i_fr], pos[i_fl]
        else:
            pos5[i_fr], pos5[i_fl] = feet_load[i_fr], feet_load[i_fl]
        o_k = ankle_mid_xy(pos5)
        yaw_k = float(np.degrees(np.arctan2(t_dir[1], t_dir[0])))
        tr = None
        if prev_o is not None:
            do = o_k - prev_o
            r0 = np.deg2rad(prev_yaw)
            c0, s0 = np.cos(r0), np.sin(r0)
            tr = Transfer(dx=c0 * do[0] + s0 * do[1],
                          dy=-s0 * do[0] + c0 * do[1],
                          dyaw=yaw_k - prev_yaw, dur=dur)
        kfs.append(Keyframe(
            pos=to_local(pos5, o_k, yaw_k), transfer=tr, label=label,
            meta={"specs": {
                hp_name: {"w": cfg.w_punch, "tol": cfg.tol_punch},
                hg_name: {"w": cfg.w_guard, "tol": cfg.tol_guard},
                "head": {"w": cfg.w_head, "tol": cfg.tol_head},
                "foot_right": {"w": cfg.w_feet, "tol": cfg.tol_feet},
                "foot_left": {"w": cfg.w_feet, "tol": cfg.tol_feet},
            }}))
        prev_o, prev_yaw = o_k, yaw_k

    task = P5Clip(name=f"stand_punch_{hand}", keyframes=tuple(kfs),
                  metadata={"hand": hand, "target": target.tolist(),
                            "attack_axis": a.tolist()})
    return PunchPlan(task=task, reachable=True, diag=diag)
