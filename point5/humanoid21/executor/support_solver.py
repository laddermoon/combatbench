"""Tool 7: P5Clip -> segments that need support (weight-transfer) planning.

For each consecutive keyframe pair, the next frame's feet are
expressed in the previous frame's coordinates via its Transfer.  A
foot whose position changes beyond threshold is the mover — that
segment requires support planning.  Returns segment descriptors or
None when nothing moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from point5.contracts import to_world

FOOT = {"right": 3, "left": 4}
_XY_EPS = 0.05            # m
_LIFT_EPS = 0.06          # m


@dataclass
class SupportTransition:
    """One segment that needs support planning."""

    seg: int               # segment index k (kf_{k-1} -> kf_k)
    mover: str             # 'right' | 'left'
    support: str           # the other foot
    mover_dxy: float       # foot horizontal displacement (m)
    mover_dz: float        # foot lift (m)


def support_transitions(clip, xy_eps: float = _XY_EPS,
                        lift_eps: float = _LIFT_EPS
                        ) -> Optional[List[SupportTransition]]:
    """Segments requiring support planning, or None."""
    out = []
    kfs = clip.keyframes
    for k in range(1, len(kfs)):
        prev, cur = kfs[k - 1], kfs[k]
        tr = cur.transfer
        cur_in_prev = to_world(cur.pos, [tr.dx, tr.dy], tr.dyaw)
        movers = []
        for side, fi in FOOT.items():
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
