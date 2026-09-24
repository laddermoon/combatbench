"""Advantage / return estimators.

Two building blocks (see ``baseline/DESIGN.md`` §3.6):

  * :func:`compute_gae` — Generalized Advantage Estimation
    (Schulman et al. 2016). Backward recursion on a *single episode*
    given step-wise rewards, value estimates, and a bootstrap value
    for the post-final-step state. Returns ``(advantages, returns)``
    arrays of shape ``(T,)``.

  * :func:`compute_returns_to_go` — pure discounted-cumulative-reward;
    the V-target / actor target used by REINFORCE-style baselines and
    value-network warmup.

P1-6: Removed ``compute_grpo_advantages`` (GRPO is unrelated to this
framework's multi-channel PPO design).

All functions are pure-numpy and operate on already-collected
trajectories, so they slot in between :class:`RolloutCollector` and
:class:`RolloutSampler` (the standard PPO pipeline).

Termination semantics
---------------------
GAE is sensitive to the difference between *terminated* and *truncated*:

  * **terminated** = the episode reached a true MDP terminal state →
    bootstrap value is ``0.0`` (no future return).
  * **truncated**  = the episode hit a step / time limit but the
    underlying MDP would have continued → bootstrap value is the
    critic's estimate of the value of the post-final-step state
    ``V(final_obs)``. Computing that value is the **caller**'s
    responsibility — the function takes ``last_value`` as an explicit
    float argument so this choice is honest and visible.

Mismatching these silently bakes survivor bias into PPO and is a
classic source of "training looks fine then plateaus weirdly". The
GAE function never tries to guess.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import erfinv as _sp_erfinv
from scipy.stats import rankdata as _sp_rankdata


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation for a single episode.

    Parameters
    ----------
    rewards: shape ``(T,)``.
    values:  shape ``(T,)`` — the critic's estimate at each of the T
             observations *that produced an action* (i.e. ``obs[:-1]``
             from a :class:`RolloutBatch`).
    last_value: bootstrap V-value for the post-final-step state.
                Set to ``0.0`` when the episode terminated (including
                sub-episode boundaries where a gating policy switched
                control to a fallback — the post-boundary state is
                out-of-distribution for the active policy's critic).
                Set to ``critic(final_obs)`` when truncated (e.g.
                episode cut off by max_steps).
    gamma: discount factor in [0, 1].
    lam:   GAE λ in [0, 1]. ``λ=1`` recovers Monte-Carlo returns,
           ``λ=0`` recovers the TD(0) advantage.

    Returns
    -------
    (advantages, returns)
        Both shape ``(T,)``. ``returns = advantages + values`` is the
        target for the value head; ``advantages`` is what PPO multiplies
        the importance ratio by.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    if rewards_arr.shape != values_arr.shape:
        raise ValueError(
            f"rewards shape {rewards_arr.shape} != values shape {values_arr.shape}."
        )
    if rewards_arr.ndim != 1:
        raise ValueError(
            f"GAE expects 1-D rewards / values; got shape {rewards_arr.shape}."
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1]; got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1]; got {lam}")

    # ------------------------------------------------------------------
    # GAE math reminder
    # ------------------------------------------------------------------
    # TD-error (1-step):
    #     δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
    #
    # GAE advantage (infinite-horizon series):
    #     A_t = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
    #
    # Recursive form (used here, computed backward in time):
    #     A_t = δ_t + γλ · A_{t+1}
    #
    # Return target for the critic:
    #     R_t = A_t + V(s_t)
    # When λ=1 the (γλ)^k δ series telescopes to Σ γ^k r_{t+k} - V(s_t),
    # so R_t collapses to the Monte-Carlo discounted return Σ γ^k r_{t+k}.
    # When λ<1, R_t is a λ-weighted blend of n-step bootstrapped returns —
    # this is the bias / variance knob (λ→0: low-variance, biased TD(0);
    # λ→1: unbiased MC with high variance).
    # ------------------------------------------------------------------
    t = rewards_arr.shape[0]
    advantages = np.zeros(t, dtype=np.float32)
    # Walking backward, `next_value` holds V(s_{i+1}) and
    # `next_advantage` holds A_{i+1} from the previous iteration.
    # For the last step, V(s_T) is the user-supplied `last_value`
    # (0 for terminal episodes, critic(final_obs) for truncated ones),
    # and A_T is defined as 0 (nothing beyond the rollout to advantage over).
    next_value = float(last_value)
    next_advantage = 0.0
    for i in range(t - 1, -1, -1):
        # δ_i = r_i + γ · V(s_{i+1}) - V(s_i)
        delta = rewards_arr[i] + gamma * next_value - values_arr[i]
        # A_i = δ_i + γλ · A_{i+1}
        next_advantage = delta + gamma * lam * next_advantage
        advantages[i] = next_advantage
        # Shift: for the next (earlier) timestep, V(s_{i}) becomes V(s_{next+1}).
        next_value = float(values_arr[i])
    # R_t = A_t + V(s_t); this is what the critic regresses to.
    returns = advantages + values_arr
    return advantages, returns


# ---------------------------------------------------------------------------
# Returns-to-go
# ---------------------------------------------------------------------------
def compute_returns_to_go(
    rewards: np.ndarray,
    *,
    last_value: float = 0.0,
    gamma: float = 0.99,
) -> np.ndarray:
    """Discounted reward-to-go (V-target for vanilla policy gradient).

    ``returns[t] = sum_{k>=t} gamma^(k-t) * r_k + gamma^(T-t) * last_value``.

    ``last_value`` follows the same terminated / truncated convention
    as :func:`compute_gae`.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    if rewards_arr.ndim != 1:
        raise ValueError(
            f"compute_returns_to_go expects 1-D rewards; got {rewards_arr.shape}."
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1]; got {gamma}")
    t = rewards_arr.shape[0]
    returns = np.zeros(t, dtype=np.float32)
    running = float(last_value)
    for i in range(t - 1, -1, -1):
        running = float(rewards_arr[i]) + gamma * running
        returns[i] = running
    return returns


# ---------------------------------------------------------------------------
# Advantage normalization
# ---------------------------------------------------------------------------
def normalize_advantages(
    adv: np.ndarray, mask: np.ndarray, method: str = "zscore",
) -> np.ndarray:
    """Advantage normalization on active frames.  Inactive frames get zero.

    Normalization is per-channel and per-update: each channel's advantages
    are independently scaled. This means the *absolute scale* of rewards
    across channels is irrelevant — only the *relative pattern* within
    each channel matters. The experiment controls cross-channel
    importance via actor_weight, not via reward magnitudes.

    Methods:
    - ``"zscore"`` — (A − mean)/std.  Batch-mean centering; frames near
      the batch mean can flip sign, which also flips that frame's
      surrogate gradient direction.
    - ``"std"`` — A/std.  Scale-only normalization; preserves the raw
      advantage sign of every frame.
    - ``"gauss_rank"`` — Gaussian quantile transform: ranks → uniform
      quantiles → Φ⁻¹.  Output is exactly N(0,1)-shaped: order is
      preserved, magnitudes are replaced by the expected normal order
      statistics.  Implicitly bounds the tail at ±Φ⁻¹(1−1/2n) (≈±4.9σ
      for n=204800) with no threshold parameter — an outlier trajectory
      cannot dominate the surrogate coefficient the way a −13σ z-score
      outlier can.

    Edge cases:
    - No active frames → all zeros (channel contributes nothing).
    - Zero variance (all advantages equal) → all zeros.  For
      ``gauss_rank`` this falls out naturally: tied ranks all map to
      the median quantile → 0.
    """
    active = adv[mask]
    if active.size == 0:
        return np.zeros_like(adv, dtype=np.float32)
    result = np.zeros_like(adv, dtype=np.float32)
    if method == "gauss_rank":
        # rankdata 'average' gives tied values the same (mean) rank, so
        # equal advantages map to equal quantiles.  Φ⁻¹(q) = √2·erfinv(2q−1).
        ranks = _sp_rankdata(active, method="average")
        q = (ranks - 0.5) / active.size
        result[mask] = (
            math.sqrt(2.0) * _sp_erfinv(2.0 * q - 1.0)
        ).astype(np.float32)
        return result
    mean = float(active.mean()) if method == "zscore" else 0.0
    std = float(active.std())
    if std < 1e-8:
        return np.zeros_like(adv, dtype=np.float32)
    result[mask] = ((active - mean) / std).astype(np.float32)
    return result
