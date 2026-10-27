"""Point5 punch planner v2 — two-phase kinetic chain, sim-unaware.

``plan_punch(initial_p5, target, hand, cfg) -> PunchPlan``

Design model (boxing kinetic chain, expressed in the 5-point language):

    start -> load -> strike        (two phases, three keyframes)

  Phase A ``load``   — FOOT + CENTRE OF MASS ONLY.
      The lead foot (ahead along the attack axis) steps forward and
      lands; the head shifts onto the new support and sinks.  BOTH
      hands stay exactly at their measured guard positions — the
      punch never telegraphs, and the step is finished before the arm
      fires (floor -> legs -> hips -> shoulder -> fist).
  Phase B ``strike`` — ARMS ONLY, fast.
      Punch hand drives straight to `target + punch_dir * penetration`.
      Guard hand retracts to the chin (defensive position, it is NOT
      left hanging).  Head adds a small final lean and a depth-scaled
      sink.  Anchor foot never moves in either phase.

  The step serves BOTH reach and power: ``step = clip(max(reach_gap,
  power_step_min), 0, max_step)`` — even an in-range punch gets a
  minimum drive step (``power_step_min``) so weight transfer actually
  happens; set it to 0 to allow pure-arm punches.

  Rear-hand punches (punching hand on the anchor-foot side, e.g. a
  right cross from a left-lead guard) get ``cross_reach_bonus`` extra
  effective reach — hip rotation buys range without moving the anchor.

Verdicts are honest: ``out_of_sector`` (needs a turn) and
``out_of_reach`` (needs a walk-in) return ``task=None``.

v2 deliberately still omits: recovery after impact, pivot/heel-lift on
the anchor foot (P5 positions only), turning, multi-step approach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .contracts import (
    POINT_INDEX,
    PointSpec,
    Keyframe,
    Point5State,
    Point5Task,
    WorldPoint,
)

_UP = np.array([0.0, 0.0, 1.0])

HAND_POINT = {"left": "hand_left", "right": "hand_right"}
FOOT_POINT = {"left": "foot_left", "right": "foot_right"}


@dataclass(frozen=True)
class PunchPlannerConfig:
    """Tunables for the v2 two-phase punch planner."""

    # --- reach / step ---
    arm_reach: float = 0.55       # fist travel from guard to impact (m)
    max_step: float = 0.35        # lead-foot advance cap (m)
    power_step_min: float = 0.12  # min drive step for weight transfer (m)
    sector_cos_min: float = 0.26  # target within ~75° of stance forward
    penetration: float = 0.08     # overshoot past target (m)
    cross_reach_bonus: float = 0.05  # rear-hand rotation reach bonus (m)

    # --- centre-of-mass proxy (head) ---
    head_sink_load: float = 0.04    # loading sink while stepping (m)
    head_step_share: float = 0.6    # head follows this fraction of step
    head_lean_strike: float = 0.04  # extra advance at impact (m)
    head_sink_scale: float = 0.15   # extra sink per m of target drop
    head_sink_max: float = 0.14     # total sink cap (m)

    # --- guard hand (tucks to chin on the strike frame) ---
    guard_tuck_back: float = 0.05   # behind chin along -a (m)
    guard_tuck_side: float = 0.10   # lateral offset to its own side (m)
    guard_tuck_drop: float = 0.12   # below head (m)

    # --- timing (phase A slower, phase B fast) ---
    dur_load: float = 0.16
    dur_extend: float = 0.10

    # --- reward annotations (downstream scorer) ---
    w_strike_hand: float = 3.0
    tol_strike_hand: float = 0.03
    w_feet: float = 1.5
    tol_feet: float = 0.05
    w_head: float = 0.5
    tol_head: float = 0.10
    w_guard: float = 1.0
    tol_guard: float = 0.08
    tol_generic: float = 0.10


@dataclass
class PunchPlan:
    """Planning verdict.  ``task is None`` iff unreachable."""

    task: Optional[Point5Task]
    reachable: bool
    reason: str = ""
    diag: Dict[str, object] = field(default_factory=dict)


def _xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).copy()
    v[2] = 0.0
    return v


def _norm(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def _stance_forward(foot_l: np.ndarray, foot_r: np.ndarray) -> np.ndarray:
    """Horizontal facing estimate from the foot line.

    Facing +X with the left foot on the +Y side gives
    cross(up, foot_R - foot_L) = +X.
    """
    return _norm(_xy(np.cross(_UP, foot_r - foot_l)))


def _spec(target: np.ndarray, w: float, tol: float) -> PointSpec:
    return PointSpec(target=WorldPoint(np.asarray(target, dtype=np.float64)),
                     tolerance=tol, weight=w)


def plan_punch(initial: Point5State,
               target: np.ndarray,
               hand: str,
               cfg: PunchPlannerConfig = PunchPlannerConfig(),
               ) -> PunchPlan:
    """Plan one strike: start -> load -> strike (two phases)."""
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if hand not in HAND_POINT:
        raise ValueError(f"hand must be 'left'/'right', got {hand!r}")

    pos = initial.pos
    hp_name = HAND_POINT[hand]
    hg_name = "hand_right" if hand == "left" else "hand_left"
    fp_l, fp_r = "foot_left", "foot_right"
    hp = pos[POINT_INDEX[hp_name]]
    hg = pos[POINT_INDEX[hg_name]]
    head = pos[POINT_INDEX["head"]]
    feet = {fp_l: pos[POINT_INDEX[fp_l]], fp_r: pos[POINT_INDEX[fp_r]]}
    diag: Dict[str, object] = {}

    # --- attack axis (horizontal) ---------------------------------
    a = _norm(_xy(target - head))
    if not np.any(a):
        a = _stance_forward(feet[fp_l], feet[fp_r])
    diag["attack_axis"] = a

    # --- frontal sector check --------------------------------------
    fwd = _stance_forward(feet[fp_l], feet[fp_r])
    cos_af = float(a @ fwd)
    diag["stance_forward"] = fwd
    diag["cos_attack_vs_stance"] = cos_af
    if np.any(fwd) and cos_af < cfg.sector_cos_min:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_sector", diag=diag)

    # --- lead vs anchor foot ---------------------------------------
    proj = {fp_l: float((feet[fp_l] - head) @ a),
            fp_r: float((feet[fp_r] - head) @ a)}
    move_name = max(proj, key=proj.get)
    anchor_name = fp_r if move_name == fp_l else fp_l
    diag["moving_foot"] = move_name
    diag["anchor_foot"] = anchor_name
    f_move = feet[move_name]
    f_anchor = feet[anchor_name]

    # --- reach -> step length (rear-hand bonus reach) --------------
    rear_hand = (FOOT_POINT[hand] == anchor_name)
    eff_reach = cfg.arm_reach + (cfg.cross_reach_bonus if rear_hand else 0.0)
    gap = float(np.linalg.norm(target - hp)) - eff_reach
    diag["rear_hand"] = rear_hand
    diag["reach_gap"] = gap
    diag["eff_arm_reach"] = eff_reach
    if gap > cfg.max_step:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_reach", diag=diag)
    step_len = min(max(gap, cfg.power_step_min), cfg.max_step)
    diag["step_len"] = step_len
    f_landed = np.array([f_move[0] + a[0] * step_len,
                         f_move[1] + a[1] * step_len,
                         0.0])

    # --- head / COM proxy ------------------------------------------
    depth = max(0.0, float(head[2] - target[2]))
    sink_strike = min(cfg.head_sink_max,
                      cfg.head_sink_load + cfg.head_sink_scale * depth)
    head_load = (head + a * (cfg.head_step_share * step_len)
                 + np.array([0, 0, -cfg.head_sink_load]))
    head_hit = head_load + a * cfg.head_lean_strike + np.array(
        [0, 0, -(sink_strike - cfg.head_sink_load)])

    # --- hands -------------------------------------------------------
    # punch hand: guard position (phase A frozen) -> straight drive
    punch_dir = _norm(target - hp)
    if not np.any(punch_dir):
        punch_dir = a
    hp_hit = target + punch_dir * cfg.penetration
    # guard hand: stays in phase A, tucks to the chin on impact
    side = 1.0 if hg_name == "hand_right" else -1.0
    right = np.array([a[1], -a[0], 0.0])
    hg_hit = (head_hit - a * cfg.guard_tuck_back
              + right * (side * cfg.guard_tuck_side)
              + np.array([0, 0, -cfg.guard_tuck_drop]))

    # --- keyframes ---------------------------------------------------
    def kf(label: str, hp_t, hg_t, head_t, fm_t) -> Keyframe:
        specs = {
            hp_name: _spec(hp_t, cfg.w_strike_hand,
                           cfg.tol_strike_hand if label == "strike"
                           else cfg.tol_generic),
            hg_name: _spec(hg_t, cfg.w_guard, cfg.tol_guard),
            "head": _spec(head_t, cfg.w_head, cfg.tol_head),
            move_name: _spec(fm_t, cfg.w_feet, cfg.tol_feet),
            anchor_name: _spec(f_anchor, cfg.w_feet, cfg.tol_feet),
        }
        return Keyframe(specs=specs, label=label)

    kf0 = kf("start", hp, hg, head, f_move)
    kf1 = kf("load", hp, hg, head_load, f_landed)
    kf2 = kf("strike", hp_hit, hg_hit, head_hit, f_landed)

    task = Point5Task(
        name=f"punch_{hand}",
        keyframes=(kf0, kf1, kf2),
        durations=(cfg.dur_load, cfg.dur_extend),
        metadata={
            "hand": hand,
            "target": target.tolist(),
            "attack_axis": a.tolist(),
            "moving_foot": move_name,
            "anchor_foot": anchor_name,
            "rear_hand": rear_hand,
            "step_len": step_len,
        },
    )
    return PunchPlan(task=task, reachable=True, diag=diag)
