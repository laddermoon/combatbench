"""Point5 punch planner v2 — two-phase kinetic chain, sim-unaware.

``plan_punch(initial_p5, target, hand, cfg) -> PunchPlan``

Design model (boxing kinetic chain, expressed in the 5-point language):

    start -> swing -> load -> extend -> strike

  Phase A ``swing -> load`` — FOOT + CENTRE OF MASS (and the guard rides
      WITH the torso, it is not left behind).  The lead foot (ahead
      along the attack axis) steps forward and lands; the head shifts
      onto the new support and sinks.  Both hands translate by the
      SAME body shift — the guard keeps its offset from the torso,
      so relative to the body nothing moved (no telegraphing), and
      the step still finishes before the arm fires.
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
  Their HAND PATH also differs: a rear hand starts behind and is driven
  through by the shoulder/hip, so the ``extend`` midpoint keeps it
  later (~35 % extension, biased to its own side) — the elbow leads
  before the fist extends, instead of a straight teleport-to-target.

Verdicts are honest: ``out_of_sector`` (needs a turn) and
``out_of_reach`` (needs a walk-in) return ``task=None``.

Keyframes:  start -> swing -> load -> extend -> strike.
v2 deliberately still omits: recovery after impact, pivot/heel-lift on
the anchor foot (P5 positions only), turning, multi-step approach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..contracts import (
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
    foot_swing_clearance: float = 0.07  # ankle rises before landing (m)
    head_lean_strike: float = 0.04  # extra advance at impact (m)
    head_sink_scale: float = 0.15   # extra sink per m of target drop
    head_sink_max: float = 0.14     # total sink cap (m)

    # --- guard hand (tucks to chin on the strike frame) ---
    guard_tuck_back: float = 0.05   # behind chin along -a (m)
    guard_tuck_side: float = 0.10   # lateral offset to its own side (m)
    guard_tuck_drop: float = 0.12   # below head (m)

    # --- punch path shaping (extend midpoint between load & strike) ---
    mid_frac_lead: float = 0.50   # lead hand: halfway along the line
    mid_frac_rear: float = 0.35   # rear hand stays back longer (rotation)
    mid_side_off: float = 0.04    # rear hand: bias to its own side (m)

    # --- torso twist (torso_yaw, deg; sign = toward punching-hand side) ---
    jab_twist_deg: float = 15.0    # lead-hand jab: small shoulder turn
    cross_twist_deg: float = 40.0  # rear-hand cross: hip drives shoulder
    twist_extend_share: float = 0.6  # fraction applied by the extend frame

    # --- timing (phase A slower, phase B fast) ---
    # Total 0.40 s (8 action steps @50 ms).  The earlier 0.28 s plan
    # demanded ~4-7 m/s fist speed — beyond what this robot learns
    # (reference strike clips that trained successfully ran 0.65-1.1 s),
    # so every episode stalled in the shared load phase.
    dur_load: float = 0.20
    dur_reach: float = 0.10       # load -> extend midpoint
    dur_snap: float = 0.10        # extend -> impact

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


def _yaw_of(t_des: np.ndarray, foot_r: np.ndarray,
            foot_l: np.ndarray) -> Optional[float]:
    """torso_yaw (deg, DESIGN.md 4.6) for a desired torso-forward world
    direction, expressed against THIS keyframe's ankle-line basis."""
    b = _xy(foot_r - foot_l)
    nb = np.linalg.norm(b)
    if nb < 1e-3:
        return None
    b /= nb
    n = np.array([-b[1], b[0]])
    t = _xy(t_des)[:2]
    if np.linalg.norm(t) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(t @ b[:2], t @ n)))


def _yaw_dir(yaw_deg: float, foot_r: np.ndarray,
             foot_l: np.ndarray) -> np.ndarray:
    """Inverse of _yaw_of: world torso-forward direction for a yaw."""
    b = _norm(_xy(foot_r - foot_l))
    n = np.array([-b[1], b[0], 0.0])
    r = np.deg2rad(yaw_deg)
    return np.cos(r) * n + np.sin(r) * b


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
    # The body follows only a share of the step, so the reach bought
    # by stepping is share*step, not step — solve for the step needed.
    rear_hand = (FOOT_POINT[hand] == anchor_name)
    eff_reach = cfg.arm_reach + (cfg.cross_reach_bonus if rear_hand else 0.0)
    gap = float(np.linalg.norm(target - hp)) - eff_reach
    diag["rear_hand"] = rear_hand
    diag["reach_gap"] = gap
    diag["eff_arm_reach"] = eff_reach
    step_need = gap / cfg.head_step_share
    if step_need > cfg.max_step:
        return PunchPlan(task=None, reachable=False,
                         reason="out_of_reach", diag=diag)
    step_len = min(max(step_need, cfg.power_step_min), cfg.max_step)
    diag["step_len"] = step_len
    # P5 feet are ankle centers: their measured planted z is ~0.09m,
    # not ground z=0. A straight ground-level slide asks the robot to
    # penetrate the floor; lift the foot then land at its initial z.
    f_swing = f_move + a * (0.4 * step_len) + _UP * cfg.foot_swing_clearance
    f_landed = f_move + a * step_len

    # --- body shift & head / COM proxy -------------------------------
    body_shift = (a * (cfg.head_step_share * step_len)
                  + np.array([0, 0, -cfg.head_sink_load]))
    head_load = head + body_shift
    depth = max(0.0, float(head[2] - target[2]))
    sink_strike = min(cfg.head_sink_max,
                      cfg.head_sink_load + cfg.head_sink_scale * depth)
    head_hit = head_load + a * cfg.head_lean_strike + np.array(
        [0, 0, -(sink_strike - cfg.head_sink_load)])

    # --- hands -------------------------------------------------------
    # The guard is carried by the torso: during the step both hands
    # translate by the same body shift (fixed offset from the body),
    # then the punch drives from the LOADED guard position.
    hp_load = hp + body_shift
    hg_load = hg + body_shift
    punch_dir = _norm(target - hp_load)
    if not np.any(punch_dir):
        punch_dir = a
    hp_hit = target + punch_dir * cfg.penetration
    mid_frac = cfg.mid_frac_rear if rear_hand else cfg.mid_frac_lead
    punch_side = 1.0 if hp_name == "hand_right" else -1.0
    right = np.array([a[1], -a[0], 0.0])
    hp_mid = (hp_load + (hp_hit - hp_load) * mid_frac
              + right * (punch_side * cfg.mid_side_off if rear_hand
                         else 0.0))
    # guard hand rides in phase A, tucks to the chin on impact
    side = 1.0 if hg_name == "hand_right" else -1.0
    hg_hit = (head_hit - a * cfg.guard_tuck_back
              + right * (side * cfg.guard_tuck_side)
              + np.array([0, 0, -cfg.guard_tuck_drop]))
    hg_mid = hg_load + (hg_hit - hg_load) * 0.5
    head_mid = head_load + (head_hit - head_load) * 0.5

    # --- torso yaw schedule -------------------------------------------
    # Phase A keeps the initial world torso direction (no telegraphing);
    # the twist fires through extend->strike toward the punching side.
    # yaw is re-expressed per keyframe against THAT frame's ankle line —
    # the baseline rotates when the lead foot steps.
    yaw0 = initial.torso_yaw if initial.torso_yaw is not None else 0.0
    t0_dir = _yaw_dir(yaw0, feet[fp_r], feet[fp_l])
    twist_deg = cfg.cross_twist_deg if rear_hand else cfg.jab_twist_deg
    # +yaw twists toward the right foot: the (b,n) basis is left-handed,
    # so a positive yaw delta is a CLOCKWISE world rotation.
    sgn = 1.0 if hand == "right" else -1.0
    twist_rad = np.deg2rad(-sgn * twist_deg)
    c, s = np.cos(twist_rad), np.sin(twist_rad)
    t_hit = np.array([c * t0_dir[0] - s * t0_dir[1],
                      s * t0_dir[0] + c * t0_dir[1], 0.0])
    t_ext = t0_dir + (t_hit - t0_dir) * cfg.twist_extend_share
    diag["strike_yaw_twist_deg"] = sgn * twist_deg

    # --- keyframes ---------------------------------------------------
    def kf(label: str, hp_t, hg_t, head_t, fm_t,
           t_dir: np.ndarray) -> Keyframe:
        feet_k = {move_name: fm_t, anchor_name: f_anchor}
        specs = {
            hp_name: _spec(hp_t, cfg.w_strike_hand,
                           cfg.tol_strike_hand if label == "strike"
                           else cfg.tol_generic),
            hg_name: _spec(hg_t, cfg.w_guard, cfg.tol_guard),
            "head": _spec(head_t, cfg.w_head, cfg.tol_head),
            move_name: _spec(fm_t, cfg.w_feet, cfg.tol_feet),
            anchor_name: _spec(f_anchor, cfg.w_feet, cfg.tol_feet),
        }
        return Keyframe(specs=specs, label=label,
                        torso_yaw=_yaw_of(t_dir, feet_k[fp_r],
                                          feet_k[fp_l]))

    kf0 = kf("start", hp, hg, head, f_move, t0_dir)
    kf1 = kf("swing", hp + body_shift * 0.4, hg + body_shift * 0.4,
             head + body_shift * 0.4, f_swing, t0_dir)
    kf2 = kf("load", hp_load, hg_load, head_load, f_landed, t0_dir)
    kf3 = kf("extend", hp_mid, hg_mid, head_mid, f_landed, t_ext)
    kf4 = kf("strike", hp_hit, hg_hit, head_hit, f_landed, t_hit)

    task = Point5Task(
        name=f"punch_{hand}",
        keyframes=(kf0, kf1, kf2, kf3, kf4),
        durations=(cfg.dur_load * 0.5, cfg.dur_load * 0.5,
                   cfg.dur_reach, cfg.dur_snap),
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
