"""Tool 6: P5 foot point -> CoP (center of pressure) range.

If the foot is grounded, its distributed pressure can be equivalently
summarized at one point on the ground — the CoP — which must stay
inside the foot's contact patch.  Returns that patch as a polygon in
the P5 local frame (around the ankle landmark).  An airborne foot
has no CoP -> ``None``.

Foot patch dims from the arena foot geoms: each foot spans
x in [-0.07, +0.14], y in [-0.057, +0.057] around its landmark.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

LIFT_Z = 0.12            # ankle landmark airborne threshold (m)
_FOOT_XMIN, _FOOT_XMAX = -0.07, 0.14
_FOOT_Y = 0.057


def foot_cop_range(foot_pos: np.ndarray) -> Optional[np.ndarray]:
    """CoP feasible polygon (4,2) in local frame, or None if airborne."""
    f = np.asarray(foot_pos, np.float64)
    if f[2] > LIFT_Z:
        return None
    xy = f[:2]
    return np.array([
        xy + [_FOOT_XMIN, -_FOOT_Y],
        xy + [_FOOT_XMAX, -_FOOT_Y],
        xy + [_FOOT_XMAX, _FOOT_Y],
        xy + [_FOOT_XMIN, _FOOT_Y]])


def cop_ranges(local_p5: np.ndarray) -> list:
    """[right, left] -> Optional[polygon] per foot."""
    p5 = np.asarray(local_p5, np.float64)
    return [foot_cop_range(p5[3]), foot_cop_range(p5[4])]
