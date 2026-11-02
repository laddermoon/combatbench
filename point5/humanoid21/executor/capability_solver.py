"""Tool 4: P5 -> COM-change capability expectation.

Physics-grounded empirical model (LIP/capture-point logic):

- Support polygon = contact patches of grounded feet (local frame).
- Horizontal accel authority along direction d:
      a(d) = g * margin(d) / z0        capped at mu*g
  where margin(d) = distance from COM xy projection to polygon
  boundary along d.  Negative margin => statically unstable.
- Velocity cap along d (capture-point): v(d) = sqrt(g/z0) * margin(d).
- Vertical authority from leg-extension reserve (head height proxy).
- Single support shrinks the polygon to one patch => low authority.

Rough, ~90% information — used by algorithm 1 for segment timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .centroidal import MASS_KG  # noqa: F401  (mass for absolute force)
from .com_solver import com_estimate

G = 9.81
MU = 0.6                      # friction-cone accel cap: mu*g
_LIFT_Z = 0.12                # airborne if ankle-point z above this
_V_CAP = 2.5                  # empirical velocity ceiling (m/s)
_A_Z_CAP = 12.0               # empirical leg accel ceiling (m/s^2)
_STAND_HEAD_Z = 1.65
_CROUCH_HEAD_Z = 0.90         # ~deep squat

# foot contact patch around the ankle landmark (local axes):
#   +x forward: toe reach, -x heel, ±y half-width
_PATCH = ((-0.05, -0.05), (0.14, -0.05), (0.14, 0.05), (-0.05, 0.05))


@dataclass
class Capability:
    """COM-change envelope for a posture (P5 local frame)."""

    com_local: np.ndarray                  # (3,) tool1 estimate
    support_polygon: np.ndarray            # (N,2) grounded-foot hull
    margin_min: float                      # min edge distance of COM (m)
    single_support: bool
    static_ok: bool                        # COM inside polygon?
    v_xy_cap: float = _V_CAP
    a_xy_cap: float = MU * G
    a_z: float = 0.0                       # vertical accel authority
    v_z: float = 0.0

    def margin(self, direction: np.ndarray) -> float:
        """Distance from COM xy to polygon boundary along direction."""
        d = np.asarray(direction, np.float64)[:2]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return 0.0
        d = d / n
        c = self.com_local[:2]
        if not len(self.support_polygon):
            return -1.0
        # max t s.t. c + t*d stays inside polygon (convex half-planes)
        poly = self.support_polygon
        t_max, out_depth = np.inf, np.inf
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            e = b - a
            nrm = np.array([e[1], -e[0]])          # inward if CCW
            if np.dot(nrm, poly.mean(axis=0) - a) < 0:
                nrm = -nrm
            inside = float(np.dot(nrm, c - a))     # >0 = inside edge
            den = float(np.dot(nrm, d))
            if inside < 0:
                out_depth = min(out_depth, -inside)
            if den < -1e-12 and inside >= 0:
                t_max = min(t_max, -inside / den)
        if out_depth < np.inf:
            return -out_depth                      # already outside
        return float(t_max) if np.isfinite(t_max) else 0.0

    def a_max(self, direction: np.ndarray) -> float:
        """Max horizontal COM acceleration along direction (m/s^2)."""
        mg = self.margin(direction)
        if mg <= 0:
            return 0.0
        z0 = max(self.com_local[2], 0.1)
        return float(min(G * mg / z0, self.a_xy_cap))

    def v_max(self, direction: np.ndarray) -> float:
        """Max horizontal COM speed sustainable along direction."""
        mg = max(self.margin(direction), 0.0)
        z0 = max(self.com_local[2], 0.1)
        return float(min(np.sqrt(G / z0) * mg, self.v_xy_cap))


def _hull2d(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of (N,2) points."""
    pts = np.unique(np.asarray(pts, np.float64), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(tuple(p))
    for p in pts[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(tuple(p))
    return np.array(lo[:-1] + up[:-1])


def capability(local_p5: np.ndarray) -> Capability:
    """Empirical capability for a P5 local pose."""
    p5 = np.asarray(local_p5, np.float64)
    com = com_estimate(p5)

    poly_pts = []
    grounded = []
    for fi in (3, 4):                            # foot_right, foot_left
        if p5[fi, 2] <= _LIFT_Z:
            grounded.append(fi)
            for dx, dy in _PATCH:
                poly_pts.append(p5[fi, :2] + np.array([dx, dy]))
    single = len(grounded) == 1
    poly = (_hull2d(np.array(poly_pts)) if poly_pts
            else np.empty((0, 2)))

    cap = Capability(com_local=com, support_polygon=poly,
                     margin_min=0.0, single_support=single,
                     static_ok=True)
    if len(poly):
        # min distance from COM to any polygon edge
        c = com[:2]
        m = len(poly)
        margins = []
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            e = b - a
            t = np.clip(np.dot(c - a, e) / max(np.dot(e, e), 1e-12),
                        0, 1)
            margins.append(np.linalg.norm(c - (a + t * e)))
        # inside => positive; outside => negative
        inside = all(np.dot(np.array([(poly[(i + 1) % m] - poly[i])[1],
                                      -(poly[(i + 1) % m] - poly[i])[0]]),
                            c - poly[i]) <= 1e-9
                     for i in range(m)) if m >= 3 else True
        cap.margin_min = float(min(margins) * (1.0 if inside else -1.0))
        cap.static_ok = inside or min(margins) < 0.01

    # vertical authority: leg-extension reserve from crouch depth
    crouch = np.clip((_STAND_HEAD_Z - p5[0, 2])
                     / (_STAND_HEAD_Z - _CROUCH_HEAD_Z), 0.0, 1.0)
    cap.a_z = _A_Z_CAP * (0.25 + 0.75 * crouch)
    cap.v_z = 1.2 * (0.3 + 0.7 * crouch)
    return cap
