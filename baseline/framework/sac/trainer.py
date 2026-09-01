"""SAC V2 update — per-channel n-step TD, auto-alpha, action-gradient normalization.

Implements the core SAC gradient step with:
- Per-channel n-step Bellman targets with per-channel gamma.
- Clipped double-Q (twin critics per channel).
- Automatic entropy temperature (alpha) tuning.
- Action-gradient normalization for actor loss (the primary mechanism
  for balancing per-channel influence on the policy).
- Soft target network updates.

See PLAN.md §1.3 for the action-gradient normalization rationale and
DECISIONS.md N7 for implementation decisions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experiment import SACParams, SACRewardChannel
from .networks import MultiHeadQCritic


# ---------------------------------------------------------------------------
# Running gradient scale statistics for action-gradient normalization
# ---------------------------------------------------------------------------

class GradNormStats:
    """Running RMS of per-channel action-gradient norms.

    Tracks ``ŝ_c = EMA(||∂Q_c/∂a||)`` for each channel. Used to
    normalize the actor loss so that ``actor_weight`` becomes a
    measurable gradient share rather than an uncalibrated scalar.

    See PLAN.md §1.3 for the mathematical rationale.
    """

    def __init__(
        self,
        channel_names: Tuple[str, ...],
        ema_decay: float = 0.99,
        eps: float = 1e-6,
    ):
        self.channel_names = channel_names
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.sq_norm: Dict[str, float] = {ch: 0.0 for ch in channel_names}
        self.initialized = False

    def update(self, grad_norms: Dict[str, float]) -> None:
        """Update running stats from a batch of gradient norms."""
        for ch in self.channel_names:
            gn = grad_norms.get(ch, 0.0)
            if not self.initialized:
                self.sq_norm[ch] = gn * gn
            else:
                self.sq_norm[ch] = (
                    self.ema_decay * self.sq_norm[ch]
                    + (1.0 - self.ema_decay) * (gn * gn)
                )
        self.initialized = True

    def scale(self, channel: str) -> float:
        """Running scale = sqrt(E[||g||²]), floored at eps."""
        return max(self.sq_norm[channel], 0.0) ** 0.5 + self.eps

    def scales(self) -> Dict[str, float]:
        return {ch: self.scale(ch) for ch in self.channel_names}


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------

def sac_update_v2(
    actor: nn.Module,
    critic: MultiHeadQCritic,
    actor_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    batch: Dict[str, Any],
    channels: Tuple[SACRewardChannel, ...],
    sp: SACParams,
    grad_clip_norm: float,
    device: torch.device,
    grad_norm_stats: Optional[GradNormStats] = None,
    grad_norm_step: int = 0,
) -> Dict[str, float]:
    """One SAC gradient step on a minibatch.

    Pipeline:
    1. Compute per-channel n-step TD targets using target Q networks.
    2. Update Q1, Q2 critics (per-channel MSE, masked by actor_weight).
    3. Update actor:
       - If ``use_grad_norm``: normalize Q by running gradient scale.
       - Else: naive weighted Q sum.
    4. Update alpha (auto-temperature).
    5. Soft-update target networks.

    Args:
        actor: Actor with ``sample_action`` and ``evaluate_actions``.
        critic: MultiHeadQCritic managing all trunk groups.
        actor_optimizer: Actor optimizer.
        log_alpha: Log entropy temperature (requires_grad).
        alpha_optimizer: Alpha optimizer (None if auto_alpha=False).
        batch: Minibatch from ``TaggedReplay.sample_nstep``.
        channels: SACRewardChannel configs.
        sp: SAC hyperparameters.
        grad_clip_norm: Max gradient norm.
        device: Torch device.
        grad_norm_stats: Running gradient scale stats (required if
            ``sp.use_grad_norm`` is True).
        grad_norm_step: Global gradient step count (for periodic
            re-estimation of gradient scale).

    Returns:
        Stats dict for logging.
    """
    channel_names = tuple(ch.name for ch in channels)
    gammas = {ch.name: ch.gamma for ch in channels}
    n_steps = {ch.name: ch.n_step for ch in channels}

    obs = batch["obs"]
    actions = batch["actions"]
    sample_weights = batch.get(
        "sample_weights", torch.ones(obs.shape[0], device=device),
    )
    alpha = log_alpha.exp().detach()

    # Normalize sample weights
    sw_mean = sample_weights.mean() + 1e-8
    batch_weights = sample_weights / sw_mean

    # ------------------------------------------------------------------
    # 1. Compute per-channel n-step TD targets
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Sample next actions from current policy
        next_obs_for_target: Dict[str, torch.Tensor] = {}
        next_actions: Optional[torch.Tensor] = None
        next_log_probs: Optional[torch.Tensor] = None

        # We need next_obs per channel (may differ if n_step differs)
        # But the next action is sampled from the same policy, so we
        # only need one set of next_actions per unique next_obs.
        # For simplicity, sample actions for each channel's next_obs.
        # Optimization: if all channels have the same n_step, they share
        # the same next_obs, so we only sample once.

        unique_next_obs: Dict[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = {}

        q_targets: Dict[str, torch.Tensor] = {}
        for ch in channel_names:
            n = n_steps[ch]
            gamma = gammas[ch]

            rewards_n = batch[f"rewards_{ch}"]  # (B, n)
            dones_n = batch[f"dones_{ch}"]      # (B, n)
            next_obs_ch = batch[f"next_obs_{ch}"]  # (B, obs_dim)
            valid_steps = batch[f"valid_steps_{ch}"]  # (B,)

            # Sample next actions for this channel's next_obs
            # Check if we've already computed for this next_obs tensor
            found = False
            for cached_obs, (cached_act, cached_lp) in unique_next_obs.items():
                if cached_obs is next_obs_ch:
                    next_actions_ch = cached_act
                    next_log_probs_ch = cached_lp
                    found = True
                    break

            if not found:
                next_actions_ch, next_log_probs_ch = actor.sample_action(
                    next_obs_ch,
                )
                unique_next_obs[next_obs_ch] = (next_actions_ch, next_log_probs_ch)

            # Compute target Q: min(Q1_target, Q2_target) - alpha * log_prob
            q1_next = critic.q1_target_forward(next_obs_ch, next_actions_ch, ch)
            q2_next = critic.q2_target_forward(next_obs_ch, next_actions_ch, ch)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs_ch

            # n-step discounted reward
            # R_n = sum_{k=0}^{n-1} gamma^k * r_k (with done truncation)
            discount_factors = torch.tensor(
                [gamma ** k for k in range(max(n_steps.values()))],
                dtype=torch.float32, device=device,
            )[:rewards_n.shape[1]]

            n_step_reward = (rewards_n * discount_factors.unsqueeze(0)).sum(dim=-1)
            n_step_reward = n_step_reward * sp.reward_scale

            # Bootstrap: gamma^n_valid * Q_next
            # If the episode terminated within the n-step window (any
            # done=1), don't bootstrap.
            any_done = (dones_n.sum(dim=-1) > 0.5).float()
            gamma_n = gamma ** valid_steps
            bootstrap = (1.0 - any_done) * gamma_n * q_next

            q_targets[ch] = n_step_reward + bootstrap

    # ------------------------------------------------------------------
    # 2. Update Q critics
    # ------------------------------------------------------------------
    q_losses: Dict[str, float] = {}
    q_means: Dict[str, float] = {}
    q_grads: Dict[str, float] = {}

    critic.zero_grad_all()

    for ch in channel_names:
        aw = batch[f"actor_weights_{ch}"]
        target = q_targets[ch]

        # Q1
        q1_pred = critic.q1_forward(obs, actions, ch)
        q1_loss = ((F.mse_loss(q1_pred, target, reduction="none") * aw * batch_weights).sum()
                   / (aw.sum() + 1e-8))
        q1_loss.backward(retain_graph=True)

        # Q2
        q2_pred = critic.q2_forward(obs, actions, ch)
        q2_loss = ((F.mse_loss(q2_pred, target, reduction="none") * aw * batch_weights).sum()
                   / (aw.sum() + 1e-8))
        q2_loss.backward(retain_graph=True)

        q_losses[f"q1_loss_{ch}"] = float(q1_loss.item())
        q_losses[f"q2_loss_{ch}"] = float(q2_loss.item())
        q_means[f"q1_mean_{ch}"] = float(q1_pred.mean().item())
        q_means[f"q2_mean_{ch}"] = float(q2_pred.mean().item())

    # Clip gradients and step all Q optimizers
    for grp in critic.groups.values():
        torch.nn.utils.clip_grad_norm_(grp.q1.parameters(), grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(grp.q2.parameters(), grad_clip_norm)
    critic.step_all()

    # ------------------------------------------------------------------
    # 3. Update actor
    # ------------------------------------------------------------------
    new_actions, new_log_probs = actor.sample_action(obs)

    # Get per-channel Q values for the newly sampled actions
    q1_all = critic.q1_forward_all(obs, new_actions)
    q2_all = critic.q2_forward_all(obs, new_actions)

    # Compute per-channel actor weights (normalized)
    aw_values = {}
    for ch in channel_names:
        aw_batch = batch[f"actor_weights_{ch}"]
        aw_values[ch] = aw_batch

    if sp.use_grad_norm and grad_norm_stats is not None:
        # --- Action-gradient normalization ---
        # Every K steps, re-estimate ||∂Q_c/∂a|| and update running stats.
        if grad_norm_step % sp.grad_norm_est_interval == 0 or not grad_norm_stats.initialized:
            grad_norms: Dict[str, float] = {}
            for ch in channel_names:
                if not any(ch_in_share for ch_in_share in [True]):
                    pass
                q_min_ch = torch.min(q1_all[ch], q2_all[ch])
                # Compute ||∂Q_c/∂a|| via autograd
                grad_ch = torch.autograd.grad(
                    outputs=q_min_ch.sum(),
                    inputs=new_actions,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                grad_norms[ch] = float(grad_ch.norm(dim=-1).mean().item())
            grad_norm_stats.update(grad_norms)

        # Build actor loss with gradient normalization
        # actor_loss = α·logπ - Σ_c w_c(s) · Q_c(s,a) / ŝ_c
        # where ŝ_c is the running gradient scale.
        scales = grad_norm_stats.scales()

        combined_q = torch.zeros_like(new_log_probs)
        total_weight = torch.zeros_like(new_log_probs)

        for ch in channel_names:
            q_min_ch = torch.min(q1_all[ch], q2_all[ch])
            s_ch = scales[ch]
            w_ch = aw_values[ch]
            combined_q = combined_q + w_ch * q_min_ch / s_ch
            total_weight = total_weight + w_ch.abs()

        # Normalize by total weight so the effective Q scale is constant
        combined_q = combined_q / (total_weight + 1e-8)

        # Per-channel gradient share diagnostic
        grad_shares: Dict[str, float] = {}
        total_sq = sum(
            (aw_values[ch].mean().item() / scales[ch]) ** 2
            for ch in channel_names
        ) + 1e-8
        for ch in channel_names:
            share = (aw_values[ch].mean().item() / scales[ch]) ** 2 / total_sq
            grad_shares[f"grad_share_{ch}"] = float(share)

    else:
        # --- Naive weighted Q sum (fallback) ---
        combined_q = torch.zeros_like(new_log_probs)
        total_weight = torch.zeros_like(new_log_probs)

        for ch in channel_names:
            q_min_ch = torch.min(q1_all[ch], q2_all[ch])
            w_ch = aw_values[ch]
            combined_q = combined_q + w_ch * q_min_ch
            total_weight = total_weight + w_ch.abs()

        combined_q = combined_q / (total_weight + 1e-8)
        grad_shares = {}

    actor_loss = ((alpha * new_log_probs - combined_q) * batch_weights).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), grad_clip_norm,
    )
    actor_optimizer.step()

    # ------------------------------------------------------------------
    # 4. Update alpha (auto-temperature)
    # ------------------------------------------------------------------
    alpha_loss_val = 0.0
    if alpha_optimizer is not None:
        target_entropy = sp.target_entropy
        if target_entropy is None:
            target_entropy = -float(actor.action_dim)

        alpha_loss = -(log_alpha * (new_log_probs.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            log_alpha.clamp_(sp.log_alpha_min, sp.log_alpha_max)
        alpha_loss_val = float(alpha_loss.item())

    # ------------------------------------------------------------------
    # 5. Soft target update
    # ------------------------------------------------------------------
    critic.soft_update(sp.tau)

    # ------------------------------------------------------------------
    # Aggregate diagnostics
    # ------------------------------------------------------------------
    result: Dict[str, float] = {
        "actor_loss": float(actor_loss.item()),
        "alpha_loss": alpha_loss_val,
        "alpha": float(alpha.item()),
        "log_prob_mean": float(new_log_probs.mean().item()),
        "grad_norm_actor": float(actor_grad),
    }
    result.update(q_losses)
    result.update(q_means)
    result.update(grad_shares)

    # Summary stats
    result["q1_loss"] = float(np.mean([q_losses[f"q1_loss_{ch}"] for ch in channel_names]))
    result["q2_loss"] = float(np.mean([q_losses[f"q2_loss_{ch}"] for ch in channel_names]))
    result["q1_mean"] = float(np.mean([q_means[f"q1_mean_{ch}"] for ch in channel_names]))

    # Gradient scale diagnostics
    if grad_norm_stats is not None and grad_norm_stats.initialized:
        for ch in channel_names:
            result[f"grad_scale_{ch}"] = grad_norm_stats.scale(ch)

    return result
