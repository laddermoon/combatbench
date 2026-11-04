"""Support planning (algorithm-1 branch).

Two layers:

1. ``support_transitions`` — which segments need weight transfer
   (foot displacement between consecutive keyframes in a shared
   frame).  Detection, not planning.

2. ``plan_footsteps`` — the support-planning sub-algorithm per
   bootstrip: from the start LandMark + clip emit an ordered
   Footstep list.  Physics rule: if COM momentum points away from
   where the next support must go, insert a corrective step
   (垫步) on the support side to brake/catch the momentum before
   the intended mover lifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from point5.contracts import compose_frame, to_world

from .centroidal import CTRL_DT
from .cop_solver import FLAT_Z as LIFT_Z

_FEET = ("right", "left")
_FIDX = {"right": 3, "left": 4}
_SETTLE_T = 0.30
_STEP_MAX = 0.35          # prior: nominal max single-step length (m)
_V_AWAY = 0.10            # m/s: momentum-away threshold for correction


# ---------- detection -------------------------------------------------------
@dataclass
class SupportTransition:
    """One segment that needs support planning."""

    seg: int
    mover: str
    support: str
    mover_dxy: float
    mover_dz: float


def support_transitions(clip, xy_eps: float = 0.05,
                        lift_eps: float = 0.06
                        ) -> Optional[List[SupportTransition]]:
    """Segments requiring support planning, or None."""
    out = []
    kfs = clip.keyframes
    for k in range(1, len(kfs)):
        prev, cur = kfs[k - 1], kfs[k]
        tr = cur.transfer
        cur_in_prev = to_world(cur.pos, [tr.dx, tr.dy], tr.dyaw)
        movers = []
        for side, fi in (("right", 3), ("left", 4)):
            d = cur_in_prev[fi] - prev.pos[fi]
            if (np.linalg.norm(d[:2]) > xy_eps or d[2] > lift_eps):
                movers.append((side, float(np.linalg.norm(d[:2])),
                               float(d[2])))
        if movers:
            movers.sort(key=lambda t: -t[1])
            m = movers[0][0]
            out.append(SupportTransition(
                seg=k, mover=m,
                support="left" if m == "right" else "right",
                mover_dxy=movers[0][1], mover_dz=movers[0][2]))
    return out or None


# ---------- per-segment phases (kept from before) ---------------------------
@dataclass
class SupportPhase:
    kind: str                # 'hold'|'shift'|'move'|'settle'
    support: Tuple[str, ...]
    dur: float
    mover: Optional[str] = None


@dataclass
class SupportSegment:
    seg: int
    mover: Optional[str]
    phases: List[SupportPhase] = field(default_factory=list)


@dataclass
class SupportPlan:
    keyframe_supports: List[Tuple[str, ...]]
    segments: List[SupportSegment]

    @property
    def n_transition_segments(self) -> int:
        return sum(1 for s in self.segments if s.mover)


def _support_at(p5_pos: np.ndarray) -> Tuple[str, ...]:
    return tuple(s for s in _FEET if p5_pos[_FIDX[s], 2] <= LIFT_Z)


def build_support_plan(clip) -> SupportPlan:
    """P5Clip -> per-keyframe support + per-segment phase plan."""
    from .capability_solver import capability
    kfs = clip.keyframes
    kf_supports = [_support_at(np.asarray(kf.pos)) for kf in kfs]
    trans = {t.seg: t for t in support_transitions(clip) or []}
    segments = []
    for k in range(1, len(kfs)):
        prev, cur = kfs[k - 1], kfs[k]
        dur = cur.transfer.dur
        t = trans.get(k)
        if t is None:
            segments.append(SupportSegment(
                seg=k, mover=None,
                phases=[SupportPhase("hold", kf_supports[k - 1],
                                     dur)]))
            continue
        sup, mov = t.support, t.mover
        landed = cur.pos[_FIDX[mov], 2] <= LIFT_Z
        cap = capability(np.asarray(prev.pos))
        fi = _FIDX[sup]
        d = float(np.linalg.norm(
            cap.com_local[:2] - prev.pos[fi, :2])) - 0.05
        v = max(cap.v_max([1, 0]), 0.3)
        sh = float(np.clip(d / v, CTRL_DT, max(dur * 0.6, CTRL_DT)))
        move = max(dur - sh, CTRL_DT)
        phases = [SupportPhase("shift", (sup,), sh, mov),
                  SupportPhase("move", (sup,), move, mov)]
        if landed and len(kf_supports[k]) == 2:
            phases.append(SupportPhase(
                "settle", kf_supports[k], _SETTLE_T))
        segments.append(SupportSegment(seg=k, mover=mov,
                                       phases=phases))
    return SupportPlan(keyframe_supports=kf_supports,
                       segments=segments)


# ---------- footstep sub-algorithm (bootstrip) ------------------------------
@dataclass
class Footstep:
    """One planned step: which foot goes to which xy point."""

    foot: str
    target: np.ndarray                  # (2,) local frame xy
    kind: str = "move"                  # 'move' | 'corrective'
    reason: str = ""


def plan_footsteps(start_landmark, clip,
                   v_away: float = _V_AWAY,
                   step_max: float = _STEP_MAX) -> List[Footstep]:
    """Start LandMark + clip -> ordered footsteps incl. correction.

    For each transition segment the mover foot steps to that
    keyframe's foot xy (expressed in frame0).  If COM momentum
    points away from the support foot (i.e. the required weight
    shift would start backward), insert a corrective step on the
    support foot in the momentum direction — a short 垫步 that
    catches the drift before the mover lifts.
    """
    steps: List[Footstep] = []
    trans = support_transitions(clip) or []
    if not trans:
        return steps

    # accumulate frame poses for world-frame foot targets
    poses = [(np.zeros(2), 0.0)]
    for k in range(1, len(clip.keyframes)):
        poses.append(compose_frame(*poses[-1],
                                   clip.keyframes[k].transfer))

    com = np.asarray(start_landmark.com)
    vel = (np.asarray(start_landmark.vel)
           if start_landmark.vel is not None else np.zeros(3))

    for t in trans:
        sup = t.support
        kf = clip.keyframes[t.seg]
        tgt_xy = to_world(
            kf.pos[_FIDX[t.mover]:_FIDX[t.mover] + 1],
            *poses[t.seg])[0, :2]
        sup_xy = np.asarray(
            start_landmark.feet[sup].foot_pos
            if start_landmark.feet[sup].foot_pos is not None
            else kf.pos[_FIDX[sup], :2])

        # corrective check: momentum pointing away from support foot
        to_sup = sup_xy - com[:2]
        away = np.dot(vel[:2], to_sup / (np.linalg.norm(to_sup)
                                       + 1e-9))
        if away < -v_away and np.linalg.norm(vel[:2]) > v_away:
            # step the support foot in the momentum direction to
            # re-establish a polygon the COM can settle over
            step = vel[:2] / (np.linalg.norm(vel[:2]) + 1e-9)
            n = np.linalg.norm(step)
            step = step / n * min(step_max, 0.25 + 0.3 * n)
            steps.append(Footstep(
                foot=sup, target=sup_xy + step[:2],
                kind="corrective",
                reason=(f"com momentum {away:.2f} m/s away from "
                        f"{sup} foot -> brace step")))
        steps.append(Footstep(foot=t.mover, target=tgt_xy,
                              kind="move",
                              reason=f"seg{t.seg} mover"))
    return steps
