"""Point5 punch planner v1 — geometric, sim-unaware, hand-coded.

Plans a single strike from an initial 5-point pose to a world target:

    plan_punch(initial_p5, target, hand, cfg) -> PunchPlan

The task model (``Point5Task``: keyframes + durations) forces every
keyframe to specify ALL five points, so the planner's real job is
synthesising a whole-body punch shape — the striking hand's endpoint
is the easy part.  The v1 model:

  * ONE foot may move (the other is a hard anchor — identical target in
    every keyframe).  WHICH foot is analysed per plan, not fixed:
    the foot already ahead along the attack axis is the "lead" and does
    the stepping; the rear foot anchors the support polygon.
  * Stepping covers a reach gap up to ``max_step``; beyond that the
    plan is honestly ``unreachable`` (a walk-in is a different problem).
  * The target must lie inside the frontal sector of the stance
    (cos(angle) >= ``sector_cos_min``) — turning is out of scope.
  * Centre-of-mass shift is expressed through the only upper-body point
    we have — the head: a small sink at prepare, then advance with the
    step plus extra sink scaled by target depth.

Keyframes:  start -> prepare -> strike (impact).

  start    exact measured pose (guard whatever it is).
  prepare  striking hand chambers slightly back/down, head sinks —
           feet unchanged (the step happens WITH the extension).
  strike   hand = target + punch_dir * penetration (hit-through),
           moving foot landed along the attack axis (z -> ground),
           head advanced with the step and sunk by target depth,
           guard hand + anchor foot untouched.

Everything is a ``WorldPoint`` anchor — positions are already absolute,
no resolver dependence.  ``weight``/``tolerance`` annotations are
written for the downstream DeepMimic-style scorer to consume.

v1 deliberately omits: recovery after impact, turn/pivot, cross-step,
target tracking (plan-time snapshot only), velocity specs.
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
    """Tunables for the v1 geometric punch planner."""

    # --- reach / step ---
    arm_reach: float = 0.55      # fist travel from chamber to impact (m)
    max_step: float = 0.35       # lead-foot advance cap (m)
    sector_cos_min: float = 0.26  # target must sit within ~75° of stance fwd
    penetration: float = 0.08    # overshoot past target along punch dir (m)

    # --- hand path ---
    chamber_back: float = 0.05   # prepare: pull hand back along -a (m)
    chamber_drop: float = 0.02   # prepare: drop hand slightly (m)

    # --- centre-of-mass proxy (head) ---
    head_sink_prepare: float = 0.02   # loading sink at prepare (m)
    head_lean: float = 0.05           # advance along attack axis (m)
    head_step_share: float = 0.5      # head follows this fraction of step
    head_sink_scale: float = 0.15     # extra sink per m of target drop
    head_sink_max: float = 0.12       # total sink cap (m)

    # --- timing ---
    dur_prepare: float = 0.12
    dur_extend: float = 0.15

    # --- reward annotations (for downstream scorer) ---
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
    """Plan one strike: start -> prepare -> strike.

    Returns ``PunchPlan(task=None, reachable=False, reason=...)`` when
    the target is outside the frontal sector or beyond one step's reach.
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if hand not in HAND_POINT:
        raise ValueError(f"hand must be 'left'/'right', got {hand!r}")

    pos = initial.pos
    hp_name = HAND_POINT[hand]
    fp_l, fp_r = "foot_left", "foot_right"
    hp = pos[POINT_INDEX[hp_name]]
    hg = pos[POINT_INDEX["hand_left" if hand == "right" else "hand_right"]]
    head = pos[POINT_INDEX["head"]]
    feet = {fp_l: pos[POINT_INDEX[fp_l]], fp_r: pos[POINT_INDEX[fp_r]]}
    diag: Dict[str, object] = {}

    # --- attack axis (horizontal) -------------------------------
    a = _norm(_xy(target - head))
    if not np.any(a):  # target straight above/below head — use stance fwd
        a = _stance_forward(feet[fp_l], feet[fp_r])
    diag["attack_axis"] = a

    # --- frontal sector check ------------------------------------
    fwd = _stance_forward(feet[fp_l], feet[fp_r])
    cos_af = float(a @ fwd)
    diag["stance_forward"] = fwd
    diag["cos_attack_vs_stance"] = cos_af
    if np.any(fwd) and cos_af < cfg.sector_cos_min:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_sector", diag=diag)

    # --- reach gap -> step length --------------------------------
    gap = float(np.linalg.norm(target - hp)) - cfg.arm_reach
    diag["reach_gap"] = gap
    if gap > cfg.max_step:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_reach", diag=diag)
    step_len = max(0.0, gap)
    diag["step_len"] = step_len

    # --- lead vs anchor foot (ahead along attack axis steps) -----
    proj = {fp_l: float((feet[fp_l] - head) @ a),
            fp_r: float((feet[fp_r] - head) @ a)}
    move_name = max(proj, key=proj.get)
    anchor_name = fp_r if move_name == fp_l else fp_l
    diag["moving_foot"] = move_name
    diag["anchor_foot"] = anchor_name
    f_move = feet[move_name]
    f_anchor = feet[anchor_name]
    # Stepping foot lands on the ground; when no step is needed the foot
    # target is left exactly as measured (a no-op, not a re-placement).
    f_landed = np.array([f_move[0] + a[0] * step_len,
                         f_move[1] + a[1] * step_len,
                         0.0 if step_len > 0 else f_move[2]])

    # --- hand path ------------------------------------------------
    hp_prep = hp - a * cfg.chamber_back + np.array([0, 0, -cfg.chamber_drop])
    punch_dir = _norm(target - hp_prep)
    if not np.any(punch_dir):
        punch_dir = np.array([a[0], a[1], 0.0])
    hp_hit = target + punch_dir * cfg.penetration

    # --- head / COM proxy -----------------------------------------
    depth = max(0.0, float(head[2] - target[2]))
    sink_strike = min(cfg.head_sink_max,
                      cfg.head_sink_prepare + cfg.head_sink_scale * depth)
    head_prep = head + np.array([0, 0, -cfg.head_sink_prepare])
    head_hit = (head
                + a * (cfg.head_lean + cfg.head_step_share * step_len)
                + np.array([0, 0, -sink_strike]))

    # --- keyframes ------------------------------------------------
    def kf(label: str, hp_t, hg_t, head_t, fm_t, fa_t) -> Keyframe:
        guard_name = "hand_right" if hand == "left" else "hand_left"
        specs = {
            hp_name: _spec(hp_t, cfg.w_strike_hand,
                           cfg.tol_strike_hand if label == "strike"
                           else cfg.tol_generic),
            guard_name: _spec(hg_t, cfg.w_guard, cfg.tol_guard),
            "head": _spec(head_t, cfg.w_head, cfg.tol_head),
            move_name: _spec(fm_t, cfg.w_feet, cfg.tol_feet),
            anchor_name: _spec(fa_t, cfg.w_feet, cfg.tol_feet),
        }
        return Keyframe(specs=specs, label=label)

    kf0 = kf("start", hp, hg, head, f_move, f_anchor)
    kf1 = kf("prepare", hp_prep, hg, head_prep, f_move, f_anchor)
    kf2 = kf("strike", hp_hit, hg, head_hit, f_landed, f_anchor)

    task = Point5Task(
        name=f"punch_{hand}",
        keyframes=(kf0, kf1, kf2),
        durations=(cfg.dur_prepare, cfg.dur_extend),
        metadata={
            "hand": hand,
            "target": target.tolist(),
            "attack_axis": a.tolist(),
            "moving_foot": move_name,
            "anchor_foot": anchor_name,
            "step_len": step_len,
        },
    )
    return PunchPlan(task=task, reachable=True, diag=diag)
