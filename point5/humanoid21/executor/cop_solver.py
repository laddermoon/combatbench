"""Tool 6: P5 foot point -> CoP (center of pressure) feasible set.

The CoP must lie inside whatever part of the foot still touches the
ground.  The ankle landmark height tells us which configuration is
possible — foot geoms from the arena XML: patch x in [-0.07, +0.14],
half-width 0.057 about the landmark.

- flat contact (z <= FLAT_Z): the whole patch polygon
- tilted (翘起来): pitching about an edge raises the ankle by
  edge_distance * sin(pitch).  The CoP degenerates to the edge(s)
  still in contact — direction of tilt unknown, so return every
  edge whose pivot can explain the current height
  (pitch_max ~50 deg):
      toe edge   reach z <= 0.06 + 0.14*sin50 = 0.167
      heel edge  reach z <= 0.06 + 0.07*sin50 = 0.114
      side edges reach z <= 0.06 + 0.057*sin50 = 0.104
- airborne: no edge reaches -> None

Return value: (N,2) list of polygon vertices, or a FootCoP for the
edge-level detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

FLAT_Z = 0.10                    # ankle landmark resting height (~0.087)
PITCH_MAX = np.radians(50.0)
_XMIN, _XMAX = -0.07, 0.14       # heel / toe reach (local x)
_HALF_Y = 0.057                  # side reach (local y)


@dataclass
class FootCoP:
    """CoP feasible set for one foot."""

    contact: bool                # any contact possible
    flat: bool                   # full sole on ground
    polygons: List[np.ndarray] = field(default_factory=list)
    edges: List[np.ndarray] = field(default_factory=list)

    @property
    def region(self) -> Optional[np.ndarray]:
        """Single polygon covering the feasible CoP locus:
        flat -> patch; tilted -> convex hull of feasible edges."""
        if self.flat and self.polygons:
            return self.polygons[0]
        if not self.edges:
            return None
        return np.vstack(self.edges)


def _patch(xy) -> np.ndarray:
    return np.array([xy + [_XMIN, -_HALF_Y], xy + [_XMAX, -_HALF_Y],
                     xy + [_XMAX, _HALF_Y], xy + [_XMIN, _HALF_Y]])


def foot_cop(foot_pos: np.ndarray) -> FootCoP:
    """Full result: flat patch, tilted edges, or airborne."""
    f = np.asarray(foot_pos, np.float64)
    xy, z = f[:2], f[2]
    if z <= FLAT_Z:
        return FootCoP(contact=True, flat=True,
                       polygons=[_patch(xy)])
    # tilted: edges whose pivot distance can produce this height
    reach = {"toe": _XMAX, "heel": -_XMIN, "side": _HALF_Y}
    edge_map = {
        "toe": [xy + [_XMAX, -_HALF_Y], xy + [_XMAX, _HALF_Y]],
        "heel": [xy + [_XMIN, -_HALF_Y], xy + [_XMIN, _HALF_Y]],
        "side_r": [xy + [_XMIN, -_HALF_Y], xy + [_XMAX, -_HALF_Y]],
        "side_l": [xy + [_XMIN, _HALF_Y], xy + [_XMAX, _HALF_Y]],
    }
    edges = []
    rise = z - FLAT_Z
    if rise <= _XMAX * np.sin(PITCH_MAX):
        edges.append(np.asarray(edge_map["toe"]))
    if rise <= -_XMIN * np.sin(PITCH_MAX):
        edges.append(np.asarray(edge_map["heel"]))
    if rise <= _HALF_Y * np.sin(PITCH_MAX):
        edges.append(np.asarray(edge_map["side_r"]))
        edges.append(np.asarray(edge_map["side_l"]))
    if not edges:
        return FootCoP(contact=False, flat=False)
    return FootCoP(contact=True, flat=False, edges=edges)


def foot_cop_range(foot_pos: np.ndarray) -> Optional[np.ndarray]:
    """CoP feasible polygon/edges, or None if airborne."""
    return foot_cop(foot_pos).region


def cop_ranges(local_p5: np.ndarray) -> list:
    """[right, left] -> Optional[polygon] per foot."""
    p5 = np.asarray(local_p5, np.float64)
    return [foot_cop_range(p5[3]), foot_cop_range(p5[4])]
