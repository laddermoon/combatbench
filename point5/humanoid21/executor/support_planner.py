"""Support planning branch of algorithm 1.

Extract the support structure implied by a P5Clip and COMPLETE it
into an executable phase plan:

  - per keyframe: which feet support (from foot height)
  - per transition segment (tool7): decompose into
        shift -> move -> [land] -> [settle]
    'shift'    COM moves onto the support foot before the mover lifts
    'move'     the mover foot travels
    'settle'   post-landing stabilisation when both feet are down

Phase durations come from the capability envelope (tool4) — the
clip's transfer.dur is intent, not a hard budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from point5.contracts import to_world

from .capability_solver import capability
from .centroidal import CTRL_DT
from .cop_solver import FLAT_Z as LIFT_Z
from .support_solver import SupportTransition, support_transitions

_FEET = ("right", "left")
_FIDX = {"right": 3, "left": 4}
_SHIFT_FRAC = 0.35         # nominal fraction of a transition seg
_SETTLE_T = 0.30           # post-landing stabilisation (s)


@dataclass
class SupportPhase:
    """One ordered phase inside a segment."""

    kind: str                # 'hold'|'shift'|'move'|'settle'
    support: Tuple[str, ...]           # grounded feet this phase
    dur: float
    mover: Optional[str] = None


@dataclass
class SupportSegment:
    """Segment k (kf_{k-1} -> kf_k) broken into support phases."""

    seg: int
    mover: Optional[str]
    phases: List[SupportPhase] = field(default_factory=list)


@dataclass
class SupportPlan:
    """Whole-clip support plan."""

    keyframe_supports: List[Tuple[str, ...]]   # per keyframe
    segments: List[SupportSegment]

    @property
    def n_transition_segments(self) -> int:
        return sum(1 for s in self.segments if s.mover)


def _support_at(p5_pos: np.ndarray) -> Tuple[str, ...]:
    return tuple(s for s in _FEET if p5_pos[_FIDX[s], 2] <= LIFT_Z)


def _shift_dur(p5_prev: np.ndarray, p5_cur: np.ndarray,
               support: str, seg_dur: float) -> float:
    """Time to move COM onto the support foot's CoP patch."""
    cap = capability(p5_prev)
    fi = _FIDX[support]
    d = float(np.linalg.norm(
        cap.com_local[:2] - p5_prev[fi, :2])) - 0.05
    v = max(cap.v_max([1, 0]), 0.3)
    return float(np.clip(d / v, 0.5 * CTRL_DT,
                         max(seg_dur * 0.6, CTRL_DT)))


def build_support_plan(clip) -> SupportPlan:
    """P5Clip -> per-keyframe support + per-segment phase plan."""
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
        sh = _shift_dur(np.asarray(prev.pos), np.asarray(cur.pos),
                        sup, dur)
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
