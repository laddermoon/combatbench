"""Model-free mass distribution proxy for a P5 pose.

Big-logic, simulator-unaware: from the five points alone, reconstruct a
skeleton-shaped *virtual body* and assign it a generic humanoid mass
table.  Outputs a COM proxy and a mass-normalized inertia tensor — no
joint angles, no IK solver, no optimizer.

Geometry (closed form, deterministic):

  - pelvis P: a column over the ankle midpoint, ``torso_len`` below the
    head, optionally leaning a small share of the head's XY offset
    (``pelvis_lean``) — the head-foot *axis* is NOT used for pelvis
    height (bent knees drop the pelvis while the head stays up).
  - torso axis n = (H - P)/|H-P|; shoulder line S = H - shoulder_drop*n;
    hips/shoulders offset laterally by hip_spread / shoulder_spread.
  - legs are straight chords hip->ankle (``knee_bend`` scales the
    two-link bow; fitted value 0.0 — the bend direction disperses over
    the pose distribution, so the mean estimator is the chord).
  - arms get a two-link elbow with ``elbow_bend`` scaling, prior
    direction down + slightly back (``elbow_back`` share of fwd).

Mass elements (14, fractions fitted offline against the real
equilibrium-solved poses of a 40.8 kg humanoid — generic defaults):

    pelvis_pt, waist_pt(P+0.17n), chest_pt(P+0.38n), head_pt,
    uarm/larm x2, thigh/shin x2, foot stubs x2.

Evaluation vs real COM on equilibrium-solved poses
(``humanoid21/eval_com_model.py``): RMSE ~4 cm natural / ~2 cm
stand-punch keyframes.  It is a reference-grade proxy, not a precise
measurement — consume it with tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .contracts import POINT_INDEX, TaskValidationError

SEG_NAMES: Tuple[str, ...] = (
    "pelvis", "waist", "chest", "head",
    "upper_arm_left", "lower_arm_left",
    "upper_arm_right", "lower_arm_right",
    "thigh_left", "shin_left", "thigh_right", "shin_right",
    "foot_left", "foot_right",
)

# default segment mass fractions (sum = 1)
DEFAULT_MASSES: Tuple[float, ...] = (
    0.162, 0.055, 0.143, 0.075,
    0.041, 0.030, 0.041, 0.030,
    0.116, 0.068, 0.116, 0.068,
    0.028, 0.028,
)


@dataclass(frozen=True)
class MassModelConfig:
    """Geometry + bend-prior tunables for the virtual body."""

    torso_len: float = 0.58       # head_z - pelvis_z (m)
    pelvis_lean: float = 0.20     # pelvis follows this share of head XY
    shoulder_drop: float = 0.13   # shoulder line below head along axis
    hip_spread: float = 0.10      # hip anchor lateral offset (m)
    shoulder_spread: float = 0.17  # shoulder anchor lateral offset (m)
    leg_upper: float = 0.42       # hip -> knee bone (m)
    leg_lower: float = 0.37       # knee -> ankle bone (m)
    arm_upper: float = 0.31       # shoulder -> elbow bone (m)
    arm_lower: float = 0.31       # elbow -> hand bone (m)
    knee_bend: float = 0.0        # expected knee bow fraction in [0,1]
    elbow_bend: float = 1.0       # expected elbow bow fraction in [0,1]
    elbow_back: float = 0.5       # elbow prior = -up - elbow_back*fwd
    waist_up: float = 0.17        # waist mass point along axis (m)
    chest_up: float = 0.38        # chest mass point along axis (m)
    toe_len: float = 0.07         # ankle -> toe stub (m)
    masses: Tuple[float, ...] = DEFAULT_MASSES


@dataclass(frozen=True)
class MassProxy:
    """Result: COM proxy + mass-normalized inertia about it.

    ``inertia_per_mass`` units are m^2 (tensor J = I/M); multiply by the
    real total mass to get an absolute inertia estimate.
    """

    com: np.ndarray                       # (3,)
    inertia_per_mass: np.ndarray          # (3,3), about ``com``
    segments: np.ndarray                  # (14,3) segment mass centres

    @property
    def gyration_z(self) -> float:
        """Radius of gyration about the vertical axis through com (m)."""
        return float(np.sqrt(max(self.inertia_per_mass[2, 2], 0.0)))


def _perp_unit(v: np.ndarray, along: np.ndarray,
               fallback: np.ndarray) -> np.ndarray:
    e = np.asarray(v, dtype=np.float64)
    e = e - float(e @ along) * along
    n = float(np.linalg.norm(e))
    if n < 1e-9:
        e = np.asarray(fallback, dtype=np.float64)
        e = e - float(e @ along) * along
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            return np.zeros(3)
    return e / n


def _two_link_mid(A: np.ndarray, T: np.ndarray, l1: float, l2: float,
                  bend_dir: np.ndarray, fallback: np.ndarray,
                  rho: float) -> np.ndarray:
    """Expected mid-joint position between anchor A and target T.

    rho=1 puts the joint on the exact two-link circle toward bend_dir;
    rho<1 shrinks toward the chord (a mean estimator when the real bend
    direction disperses).
    """
    d = T - A
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return A.copy()
    along = d / dist
    dc = min(dist, l1 + l2 - 1e-6)
    x = (l1 * l1 - l2 * l2 + dc * dc) / (2.0 * dc)
    h = float(np.sqrt(max(l1 * l1 - x * x, 0.0)))
    e = _perp_unit(bend_dir, along, fallback)
    return A + along * x + e * h * rho


def virtual_body(p5: np.ndarray,
                 cfg: MassModelConfig = MassModelConfig()):
    """Reconstruct the virtual body from a local P5 pose.

    Returns (centers, lines): ``centers`` (14,3) mass-element positions
    matching ``SEG_NAMES``/``cfg.masses``; ``lines`` (14,) of Optional
    bone vectors (endpoint difference) for line elements, None for
    point masses.
    """
    pos = np.asarray(p5, dtype=np.float64)
    if pos.shape != (5, 3) or not np.all(np.isfinite(pos)):
        raise TaskValidationError(
            f"p5 must be a finite (5,3) array, got {pos.shape}")
    h = pos[POINT_INDEX["head"]]
    ar = pos[POINT_INDEX["hand_right"]]
    al = pos[POINT_INDEX["hand_left"]]
    fr = pos[POINT_INDEX["foot_right"]]
    fl = pos[POINT_INDEX["foot_left"]]
    F = 0.5 * (fr + fl)
    up = np.array([0.0, 0.0, 1.0])

    P = np.array([F[0] + cfg.pelvis_lean * (h[0] - F[0]),
                  F[1] + cfg.pelvis_lean * (h[1] - F[1]),
                  h[2] - cfg.torso_len])
    d = h - P
    u = float(np.linalg.norm(d))
    n = d / u if u > 1e-9 else up
    left = _perp_unit(np.array([0.0, 1.0, 0.0]), n,
                      np.array([1.0, 0.0, 0.0]))
    fwd = _perp_unit(np.array([1.0, 0.0, 0.0]), n, up)

    S = h - cfg.shoulder_drop * n
    hip_l = P + left * cfg.hip_spread
    hip_r = P - left * cfg.hip_spread
    sho_l = S + left * cfg.shoulder_spread
    sho_r = S - left * cfg.shoulder_spread

    knee_l = _two_link_mid(hip_l, fl, cfg.leg_upper, cfg.leg_lower,
                           fwd, up, cfg.knee_bend)
    knee_r = _two_link_mid(hip_r, fr, cfg.leg_upper, cfg.leg_lower,
                           fwd, up, cfg.knee_bend)
    e_l = -up - cfg.elbow_back * fwd
    e_r = -up - cfg.elbow_back * fwd
    elb_l = _two_link_mid(sho_l, al, cfg.arm_upper, cfg.arm_lower,
                          e_l, fwd, cfg.elbow_bend)
    elb_r = _two_link_mid(sho_r, ar, cfg.arm_upper, cfg.arm_lower,
                          e_r, fwd, cfg.elbow_bend)

    toe_l = fl + fwd * cfg.toe_len
    toe_r = fr + fwd * cfg.toe_len

    centers = np.stack([
        P, P + cfg.waist_up * n, P + cfg.chest_up * n, h,
        0.5 * (sho_l + elb_l), 0.5 * (elb_l + al),
        0.5 * (sho_r + elb_r), 0.5 * (elb_r + ar),
        0.5 * (hip_l + knee_l), 0.5 * (knee_l + fl),
        0.5 * (hip_r + knee_r), 0.5 * (knee_r + fr),
        0.5 * (fl + toe_l), 0.5 * (fr + toe_r),
    ])
    lines = (
        None, None, None, None,
        elb_l - sho_l, al - elb_l,
        elb_r - sho_r, ar - elb_r,
        knee_l - hip_l, fl - knee_l,
        knee_r - hip_r, fr - knee_r,
        toe_l - fl, toe_r - fr,
    )
    return centers, lines


def mass_proxy(p5: np.ndarray,
               cfg: MassModelConfig = MassModelConfig()) -> MassProxy:
    """P5 (5,3) local pose -> COM proxy + mass-normalized inertia."""
    centers, lines = virtual_body(p5, cfg)
    mu = np.asarray(cfg.masses, dtype=np.float64)
    if mu.shape != (len(SEG_NAMES),):
        raise TaskValidationError("masses must have 14 entries")
    mu = mu / mu.sum()
    com = np.einsum("k,kc->c", mu, centers)
    J = np.zeros((3, 3))
    for i, (c, dv) in enumerate(zip(centers, lines)):
        r = c - com
        J += mu[i] * (float(r @ r) * np.eye(3) - np.outer(r, r))
        if dv is not None:
            dvv = np.asarray(dv, dtype=np.float64)
            J += (mu[i] / 12.0) * (float(dvv @ dvv) * np.eye(3)
                                   - np.outer(dvv, dvv))
    return MassProxy(com=com, inertia_per_mass=J, segments=centers)
