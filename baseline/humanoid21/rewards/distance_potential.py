"""Distance potential reward for fight curriculum.

Linear potential function Φ(d)::

    Φ(d) = clamp((d_max - d) / (d_max - d_strike), 0, 1)

where ``d`` is the 2D horizontal distance between the agent and the opponent
(root xy positions from the ``approach_velocity`` observer).

Φ = 1 when ``d <= d_strike`` (at striking range) and decreases linearly
to 0 at ``d = d_max``.

Trajectory smoothing (reference: ``follow_opponent.py``):
    Both self and opponent xy trajectories are smoothed with a centered
    moving average before computing distance, to filter out gait oscillation
    noise (left-right sway during walking).  The window width is chosen to
    cover roughly one gait cycle (~0.85s at dt=0.05s → window=17).

Dense reward mode (reference: ``standing_balance_4stage_dense``)::

    r[t] = (1 - γ) · Φ(d_t)

Every step at striking range gets a constant positive reward ``(1-γ)·1.0``;
far away gets ≈ 0.  No terminal special-casing needed.
"""

from __future__ import annotations

import numpy as np

# Trajectory smoothing window (action-step count).
# dt = 1/CONTROL_FREQUENCY = 0.05s, N=17 ≈ 0.85s ≈ one gait cycle.
DIST_SMOOTH_WINDOW = 17


def _centered_moving_average(xy: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average for (T, 2) trajectory; boundary window shrinks."""
    arr = np.asarray(xy, dtype=np.float64)
    T = arr.shape[0]
    if window <= 1 or T == 0:
        return arr.copy()
    half = int(window) // 2
    csum = np.cumsum(arr, axis=0)
    out = np.empty_like(arr)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T - 1, t + half)
        s = csum[hi] - (csum[lo - 1] if lo > 0 else 0.0)
        out[t] = s / (hi - lo + 1)
    return out


def compute_dense_distance_reward(
    self_xy: np.ndarray,
    opp_xy: np.ndarray,
    *,
    d_strike: float = 0.7,
    d_max: float = 8.0,
    gamma: float = 0.99,
    smooth_window: int = DIST_SMOOTH_WINDOW,
) -> np.ndarray:
    """Compute Dense distance reward from per-step xy positions.

    Smooths both trajectories with centered moving average to remove gait
    oscillation, then computes 2D distance and applies a linear potential.

    Args:
        self_xy: ``(T, 2)`` array of agent root xy positions.
        opp_xy:  ``(T, 2)`` array of opponent root xy positions.
        d_strike: striking distance (m) where Φ = 1.
        d_max: distance (m) where Φ = 0.
        gamma: discount factor for Dense scaling.
        smooth_window: centered moving average window width (action steps).

    Returns:
        ``(T,)`` float32 array of Dense rewards: ``(1-γ) · Φ(d_t)``.
    """
    self_xy = np.asarray(self_xy, dtype=np.float64)
    opp_xy = np.asarray(opp_xy, dtype=np.float64)
    T = self_xy.shape[0]

    if T == 0:
        return np.zeros(0, dtype=np.float32)

    # Smooth both trajectories to filter gait oscillation
    sm_self = _centered_moving_average(self_xy, smooth_window)
    sm_opp = _centered_moving_average(opp_xy, smooth_window)

    # 2D horizontal distance from smoothed positions
    dist = np.linalg.norm(sm_opp - sm_self, axis=1)

    # Linear potential: Φ(d) = clamp((d_max - d) / (d_max - d_strike), 0, 1)
    phi = np.clip((d_max - dist) / (d_max - d_strike), 0.0, 1.0)

    # Dense: r[t] = (1 - γ) · Φ(d_t)
    r = ((1.0 - gamma) * phi).astype(np.float32)

    return r
