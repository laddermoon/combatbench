"""Potential-Based Reward Shaping (PBRS) for distance to opponent.

The potential function Φ(s) is a smooth bump centred at the striking distance
``d_strike``::

    Φ(s) = exp(-k * (d - d_strike)²)

where ``d`` is the 2D horizontal distance between the agent and the opponent.
Φ = 1 when ``d = d_strike`` (at striking range) and decays smoothly as the
agent moves away from or overshoots the striking distance.

The PBRS reward is the standard difference form::

    r[t] = γ · Φ(s_{t+1}) − Φ(s_t)

This is **policy-invariant** (Ng et al., 1999): the shaping does not change
the optimal policy, only speeds up learning by providing a dense gradient
towards the striking distance.

At episode termination (including sub-episode boundaries), Φ(terminal) = 0
so the last step receives ``r[T-1] = -Φ(s_{T-1})``, charging the policy for
any remaining distance potential.
"""

from __future__ import annotations

import numpy as np


def compute_pbrs_distance_reward(
    self_xy: np.ndarray,
    opp_xy: np.ndarray,
    *,
    d_strike: float = 0.7,
    k: float = 3.0,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute PBRS distance reward from per-step xy positions.

    Args:
        self_xy: ``(T, 2)`` array of agent root xy positions.
        opp_xy:  ``(T, 2)`` array of opponent root xy positions.
        d_strike: striking distance (m) where Φ = 1.
        k: sharpness of the bump.  Higher = more focused on d_strike.
        gamma: discount factor for the PBRS difference.

    Returns:
        ``(T,)`` float32 array of PBRS rewards.
    """
    self_xy = np.asarray(self_xy, dtype=np.float64)
    opp_xy = np.asarray(opp_xy, dtype=np.float64)
    T = self_xy.shape[0]

    if T == 0:
        return np.zeros(0, dtype=np.float32)

    # 2D horizontal distance per step
    diff = opp_xy - self_xy
    dist = np.linalg.norm(diff, axis=1)

    # Potential Φ(s) = exp(-k * (d - d_strike)²)
    phi = np.exp(-k * (dist - d_strike) ** 2)

    # PBRS: r[t] = γ · Φ(t+1) − Φ(t)
    # Terminal state has Φ = 0, so r[T-1] = -Φ(T-1)
    r = np.zeros(T, dtype=np.float32)
    if T > 1:
        r[:-1] = (gamma * phi[1:] - phi[:-1]).astype(np.float32)
    r[-1] = float(-phi[-1])  # terminal: Φ(terminal) = 0

    return r
