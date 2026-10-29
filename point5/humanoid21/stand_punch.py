"""Standing punch planner — strike from the standard standing pose.

``plan_stand_punch(target, hand, cfg, initial=None) -> PunchPlan``

Four keyframes — a full-body committed strike (前手拳 lead-hand only):

    coil(降重心) -> swing(抬脚上步) -> land(落地刃架抱架) -> strike(转腰出拳)

  ``coil``   — from standing, COM sinks, legs bend; feet and facing stay.
  ``swing``  — the lead foot lifts mid-step along the attack line (the
      ground line from the ankle-midpoint projection to the target's);
      the torso already blades 30° toward the NON-punching side; both
      hands come up to a chest guard (一拳之隔, slightly above elbow).
  ``land``   — lead foot lands parallel-offset along the line (full step
      derived from the reach gap), torso at the same 30° blade, hands
      chambered at the chest.
  ``strike`` — COM keeps driving toward the target while the fist
      extends to ``target + punch_dir * penetration``; the torso keeps
      rotating the same way — 60° to the line at impact (another 30°).
      Guard hand holds the chamber.

Mirroring: left punch blades to the RIGHT of the line (-30° → -60°);
right punch blades left (+30° → +60°).

``initial`` defaults to the standard standing local pose
(:func:`stand_p5`).  ``target`` is expressed in the initial local frame
(torso-facing, ankle-mid origin) — the same convention as every v2
consumer.  Verdicts: ``out_of_sector`` (target far outside the frontal
cone), ``out_of_reach`` (gap > ``max_step``).
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
    max_step: float = 0.60        # lead-foot advance cap (m)
    head_step_share: float = 0.75  # head follows this fraction of step
    foot_swing_clearance: float = 0.06
    step_sink_extra: float = 0.05  # extra COM sink driven by the step (m)

    # --- coil / land ---
    head_sink_load: float = 0.06  # crouch while coiling (m)

    # --- guard (双手胸前抱架, 与躯干隔一拳, 手略高于肘) ---
    guard_fwd: float = 0.13       # hands ahead of chest (m)
    guard_side: float = 0.12      # lateral offset (m)
    guard_drop: float = 0.32      # below head = chest height (m)

    # --- strike ---
    head_lean_strike: float = 0.08
    head_sink_extra: float = 0.03  # additional sink on impact (m)

    # --- transfer duration is a schema-required placeholder only;
    #     P5 carries no timing semantics, execution timing is downstream.
    dur: float = 0.1

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
    # The punch fires from the chambered guard, which sits *behind* the
    # standing hand — solve the step iteratively so the chamber ends up
    # exactly one arm-reach from the target.
    sgn = 1.0 if hand == "right" else -1.0  # + = toward the right side
    right = np.array([a[1], -a[0], 0.0])
    step_len = 0.0
    for _ in range(3):
        sink = (cfg.head_sink_load
                + cfg.step_sink_extra * (step_len / cfg.max_step))
        hp_est = (head + a * (cfg.head_step_share * step_len)
                  + a * cfg.guard_fwd
                  + right * (sgn * cfg.guard_side)
                  + np.array([0, 0, -(sink + cfg.guard_drop)]))
        gap = float(np.linalg.norm(target - hp_est)) - cfg.arm_reach
        if gap <= 0.0:
            break
        step_len += gap / cfg.head_step_share
        step_len = min(step_len, cfg.max_step)
    diag["reach_gap"] = gap
    diag["step_len"] = step_len
    if gap > 0.0 and step_len >= cfg.max_step:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_reach", diag=diag)

    # stepping foot = the punch-side foot (left punch -> left foot steps)
    move_i = i_fl if hand == "left" else i_fr
    anchor_i = i_fl if move_i == i_fr else i_fr
    move_name = ("foot_right" if move_i == i_fr else "foot_left")
    diag["moving_foot"] = move_name
    diag["anchor_foot"] = ("foot_right" if anchor_i == i_fr
                           else "foot_left")
    f_move = pos[move_i].copy()
    f_anchor = pos[anchor_i]
    f_landed = f_move + a * step_len

    # === choreography (DESIGN: coil -> swing -> land/blade -> strike) ===
    # Left-hand example: blade to the RIGHT of the attack line (CW = -),
    # then keep rotating right through the punch -> -60° at impact.
    # Right hand mirrors (+30° blade, +60° impact).
    blade_deg = sgn * 30.0
    strike_deg = sgn * 60.0
    t_coil = np.array([1.0, 0.0, 0.0])
    t_blade = _rotz(t_coil, blade_deg)
    t_strike = _rotz(t_coil, strike_deg)
    diag["blade_deg"] = blade_deg
    diag["strike_deg"] = strike_deg

    # kf0 coil: sink, legs bend, feet still, facing unchanged
    head_coil = head + np.array([0, 0, -cfg.head_sink_load])

    # kf1 swing: lead foot lifts ~half-step along the line, bladed already
    f_swing = f_move + a * (0.5 * step_len) \
        + _UP * min(cfg.foot_swing_clearance, 0.5 * step_len)
    head_swing = head_coil + a * (cfg.head_step_share * step_len * 0.4)
    guard_z = head_swing[2] - cfg.guard_drop
    hp_swing = (head_swing + a * cfg.guard_fwd
                + right * (sgn * cfg.guard_side))
    hp_swing[2] = guard_z
    hg_swing = (head_swing + a * cfg.guard_fwd
                + right * ((-sgn) * cfg.guard_side))
    hg_swing[2] = guard_z

    # kf2 land/blade: foot lands parallel to the line, full 30° blade,
    # both hands chambered in front of the chest
    body_shift = (a * (cfg.head_step_share * step_len)
                  + np.array([0, 0, -(cfg.head_sink_load
                                     + cfg.step_sink_extra
                                     * (step_len / cfg.max_step))]))
    head_load = head + body_shift
    guard_z_l = head_load[2] - cfg.guard_drop
    hp_load = head_load + a * cfg.guard_fwd \
        + right * (sgn * cfg.guard_side)
    hp_load[2] = guard_z_l
    hg_load = head_load + a * cfg.guard_fwd \
        + right * ((-sgn) * cfg.guard_side)
    hg_load[2] = guard_z_l
    feet_load = {i_fr: pos[i_fr].copy(), i_fl: pos[i_fl].copy()}
    feet_load[move_i] = f_landed
    feet_load[anchor_i] = f_anchor

    # kf3 strike: COM drives forward, fist to target+penetration,
    # torso keeps rotating to 60°; guard hand holds the chamber
    punch_dir = _norm(target - hp_load)
    if not np.any(punch_dir):
        punch_dir = a
    # penetration capped so the fist stays within arm reach (IK-legible)
    hp_hit = hp_load + punch_dir * min(
        float(np.linalg.norm(target - hp_load)) + cfg.penetration,
        cfg.arm_reach)
    head_hit = (head_load + a * cfg.head_lean_strike
                + np.array([0, 0, -cfg.head_sink_extra]))
    hg_hit = head_hit + a * cfg.guard_fwd \
        + right * ((-sgn) * cfg.guard_side)
    hg_hit[2] = head_hit[2] - cfg.guard_drop

    # --- keyframes: re-express each in its own local frame --------------
    feet_swing = {i_fr: pos[i_fr].copy(), i_fl: pos[i_fl].copy()}
    feet_swing[move_i] = f_swing
    feet_swing[anchor_i] = f_anchor
    rows = [
        ("coil",   hp,       hg,      head_coil,  t_coil,
         {i_fr: pos[i_fr], i_fl: pos[i_fl]}),
        ("swing",  hp_swing, hg_swing, head_swing, t_blade, feet_swing),
        ("land",   hp_load,  hg_load, head_load,  t_blade, feet_load),
        ("strike", hp_hit,   hg_hit,  head_hit,   t_strike, feet_load),
    ]
    kfs, prev_o, prev_yaw = [], None, None
    for label, hp_t, hg_t, head_t, t_dir, feet_k in rows:
        pos5 = np.empty((5, 3))
        pos5[POINT_INDEX["head"]] = head_t
        pos5[i_hp] = hp_t
        pos5[i_hg] = hg_t
        pos5[i_fr], pos5[i_fl] = feet_k[i_fr], feet_k[i_fl]
        o_k = ankle_mid_xy(pos5)
        yaw_k = float(np.degrees(np.arctan2(t_dir[1], t_dir[0])))
        tr = None
        if prev_o is not None:
            do = o_k - prev_o
            r0 = np.deg2rad(prev_yaw)
            c0, s0 = np.cos(r0), np.sin(r0)
            tr = Transfer(dx=c0 * do[0] + s0 * do[1],
                          dy=-s0 * do[0] + c0 * do[1],
                          dyaw=yaw_k - prev_yaw, dur=cfg.dur)
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
