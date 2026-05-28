"""Advantage / return estimators.

Three building blocks (see ``baseline/DESIGN.md`` §3.6):

  * :func:`compute_gae` — Generalized Advantage Estimation
    (Schulman et al. 2016). Backward recursion on a *single episode*
    given step-wise rewards, value estimates, and a bootstrap value
    for the post-final-step state. Returns ``(advantages, returns)``
    arrays of shape ``(T,)``.

  * :func:`compute_returns_to_go` — pure discounted-cumulative-reward;
    the V-target / actor target used by REINFORCE-style baselines and
    value-network warmup.

  * :func:`compute_grpo_advantages` — group-relative advantages used by
    GRPO (DeepSeek 2024). Centers and scales the return of each
    trajectory in a *group* by the group mean / std. No critic needed.

All functions are pure-numpy and operate on already-collected
trajectories, so they slot in between :class:`RolloutCollector` and
:class:`RolloutSampler` (the standard PPO pipeline) or after a batch of
group rollouts (the GRPO pipeline).

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

from typing import List, Sequence, Tuple

import numpy as np


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
                Set to ``0.0`` when the episode terminated; set to
                ``critic(final_obs)`` when truncated.
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
# GRPO group-relative advantages
# ---------------------------------------------------------------------------
def compute_grpo_advantages(
    rewards_per_episode: Sequence[np.ndarray],
    *,
    gamma: float = 1.0,
    eps: float = 1e-8,
    broadcast_to_steps: bool = True,
) -> List[np.ndarray]:
    """Group-relative advantages (DeepSeek GRPO).

    For each trajectory in a *group* (``rewards_per_episode``), compute
    a single scalar return ``R_i = sum gamma^t r_t``, then produce per-
    step advantages by centering and scaling within the group:

        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    With ``broadcast_to_steps=True`` (default) the returned advantage
    arrays match each episode's length ``T_i`` (the same scalar
    repeated along time) — that's the shape PPO's loss expects, so the
    GRPO pipeline can drop straight into :func:`ppo_loss` without a
    critic. Set to ``False`` to get a list of scalar floats instead
    (when the caller wants raw episode-level advantages).

    ``gamma=1.0`` (default) is the canonical GRPO setup — the group
    is a set of completions sharing a prompt, so the time-axis discount
    is meaningless. Other values are accepted for experiments.
    """
    if not rewards_per_episode:
        raise ValueError("rewards_per_episode is empty.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1]; got {gamma}")

    returns: List[float] = []
    for rewards in rewards_per_episode:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.ndim != 1:
            raise ValueError(
                f"GRPO expects 1-D rewards per episode; got {rewards_arr.shape}."
            )
        if gamma == 1.0:
            ret = float(rewards_arr.sum())
        else:
            discount = gamma ** np.arange(rewards_arr.shape[0], dtype=np.float64)
            ret = float((rewards_arr * discount).sum())
        returns.append(ret)
    returns_np = np.asarray(returns, dtype=np.float64)
    mean = float(returns_np.mean())
    std = float(returns_np.std()) + float(eps)
    centered = (returns_np - mean) / std

    if not broadcast_to_steps:
        return [float(a) for a in centered]
    out: List[np.ndarray] = []
    for i, rewards in enumerate(rewards_per_episode):
        t = int(np.asarray(rewards).shape[0])
        out.append(np.full(t, float(centered[i]), dtype=np.float32))
    return out
