"""Point5 landmark definitions for Humanoid21.

Each business point is a fixed offset in a named body's local frame:
``world = body_xpos + body_xmat @ offset``.

- head / hands: geom (sphere) centers coincide with the body origin.
- feet: ankle-center = midpoint of the ankle_y (local z=0.08) and
  ankle_x (local z=0.04) joint anchors on the foot body -> [0,0,0.06].

Shared by the derived-view adapter and the pose-completion FK/solver.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from mocap.point5.contracts import POINT_NAMES

# point_name -> (body_name_without_suffix, local_offset_in_body_frame)
LANDMARKS: Dict[str, Tuple[str, np.ndarray]] = {
    "head": ("head", np.zeros(3)),
    "hand_right": ("hand_right", np.zeros(3)),
    "hand_left": ("hand_left", np.zeros(3)),
    "foot_right": ("foot_right", np.array([0.0, 0.0, 0.06])),
    "foot_left": ("foot_left", np.array([0.0, 0.0, 0.06])),
}

FOOT_NAMES = ("foot_right", "foot_left")
