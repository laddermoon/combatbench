"""SAC trainer — ReplayBuffer, QCriticMLP, and sac_update.

Implements Soft Actor-Critic (Haarnoja et al., 2018) with:
- Twin Q-critics (clipped double-Q)
- Automatic entropy temperature tuning
- Soft target network updates
- Replay buffer with episode-based insertion
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.common.rollout import Episode

from .experiment import CommonParams, Experiment, SACParams, TrainablePolicy


# ---------------------------------------------------------------------------
# Q-Critic network: Q(s, a) -> scalar
# ---------------------------------------------------------------------------

class QCriticMLP(nn.Module):
    """Q(s, a) critic for SAC.

    Concatenates observation and action, then passes through two hidden
    Tanh layers to produce a scalar Q-value.

    Args:
        obs_dim: Length of the flat observation vector.
        action_dim: Length of the action vector.
        hidden_dim: Width of both hidden layers.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q(s, a) as a (batch,) tensor."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity circular replay buffer storing (s, a, r_k, s', done, w).

    Rewards are stored **per component** (dict of arrays keyed by reward_key)
    to support multi-critic SAC.  Sample weights from
    ``prepare_training_segments`` are stored per transition and used to
    weight losses, matching PPO's sample-weight handling.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, reward_keys: Tuple[str, ...]):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.reward_keys = reward_keys
        self.size = 0
        self.ptr = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.sample_weights = np.ones(capacity, dtype=np.float32)
        self.rewards = {
            k: np.zeros(capacity, dtype=np.float32) for k in reward_keys
        }

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        rewards: Dict[str, float],
        next_obs: np.ndarray,
        done: bool,
        sample_weight: float = 1.0,
    ) -> None:
        """Add a single transition with per-component rewards and sample weight."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        for k in self.reward_keys:
            self.rewards[k][self.ptr] = rewards.get(k, 0.0)
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.sample_weights[self.ptr] = sample_weight
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(
        self,
        episode: Episode,
        experiment: Experiment,
        stage_weights: Tuple[float, ...],
        common_params: CommonParams,
    ) -> int:
        """Extract transitions from an Episode and add them to the buffer.

        Uses ``experiment.prepare_training_segments`` to split the episode
        into weighted sub-segments, matching PPO's segment handling.
        Per-component rewards are stored independently (no combining).

        Returns the number of transitions added.
        """
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        obs = episode.observations.get(ep_target)
        acts = episode.actions.get(ep_target)
        fin = episode.final_observation.get(ep_target)
        if obs is None or acts is None or fin is None:
            return 0

        T_full = int(acts.shape[0])
        if T_full == 0:
            return 0

        # Extract per-component rewards for the full episode
        reward_dict = experiment.extract_rewards(episode)

        # Use prepare_training_segments to get weighted sub-segments
        seg_weights = experiment.prepare_training_segments(episode)
        if not seg_weights:
            return 0

        is_terminated = bool(episode.is_terminated)
        n_added = 0
        for start, end, weight in seg_weights:
            T_seg = end - start
            if T_seg == 0:
                continue
            for t in range(start, end):
                if t < T_full - 1:
                    next_o = obs[t + 1]
                    done = False
                else:
                    next_o = fin
                    done = is_terminated

                step_rewards = {
                    k: float(reward_dict.get(k, np.zeros(T_full))[t])
                    for k in self.reward_keys
                }
                self.add(
                    obs=obs[t].astype(np.float32),
                    action=acts[t].astype(np.float32),
                    rewards=step_rewards,
                    next_obs=next_o.astype(np.float32),
                    done=done,
                    sample_weight=weight,
                )
                n_added += 1

        return n_added

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """Sample a random minibatch and return as GPU tensors.

        Returns dict with keys: obs, actions, next_obs, dones,
        sample_weights, and rewards_<key> for each reward component.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        batch: Dict[str, Any] = {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
            "sample_weights": torch.as_tensor(
                self.sample_weights[idx], dtype=torch.float32, device=device,
            ),
        }
        for k in self.reward_keys:
            batch[f"rewards_{k}"] = torch.as_tensor(
                self.rewards[k][idx], dtype=torch.float32, device=device,
            )
        return batch


# ---------------------------------------------------------------------------
# Running Q statistics for stable normalization
# ---------------------------------------------------------------------------

class QRunningStats:
    """Per-component running statistics of the Bellman Q-TARGETS.

    This is the stable normalizer for PPO-aligned multi-objective SAC.  We
    track an EMA of the mean and second moment of each component's Q-target
    distribution (Q-targets are far more stable than the online Q since they
    do not back-propagate and are smoothed by the target networks).

    The actor loss divides each component's Q by ``std(key)`` so that every
    reward component contributes an action-gradient of comparable scale --
    the direct analog of PPO's per-component advantage normalization
    ``(A - mean) / std``.  Unlike the earlier (buggy) version, the std is
    floored at a *small* epsilon (not 1.0): flooring at 1.0 silently
    disabled the scaling for small-variance components and let the entropy
    term dominate the actor loss, collapsing the policy.

    A short warmup is enforced so we do not amplify noise before the stats
    have seen enough batches.
    """

    def __init__(
        self,
        reward_keys: Tuple[str, ...],
        ema_decay: float = 0.99,
        eps: float = 1e-2,
        warmup_updates: int = 200,
    ):
        self.reward_keys = reward_keys
        self.ema_decay = ema_decay
        self.eps = eps
        self.warmup_updates = warmup_updates
        self.mean: Dict[str, float] = {k: 0.0 for k in reward_keys}
        self.sq_mean: Dict[str, float] = {k: 0.0 for k in reward_keys}  # EMA of Q^2
        self.initialized = False
        self.n_updates = 0

    def update(self, q_values: Dict[str, torch.Tensor]) -> None:
        """Update running stats from a batch of Q-targets (detached)."""
        decay = self.ema_decay
        for key in self.reward_keys:
            q = q_values[key].detach()
            b_mean = float(q.mean().item())
            b_sq_mean = float((q * q).mean().item())
            if not self.initialized:
                self.mean[key] = b_mean
                self.sq_mean[key] = b_sq_mean
            else:
                self.mean[key] = decay * self.mean[key] + (1 - decay) * b_mean
                self.sq_mean[key] = decay * self.sq_mean[key] + (1 - decay) * b_sq_mean
        self.initialized = True
        self.n_updates += 1

    @property
    def ready(self) -> bool:
        """True once stats are warm enough to normalize with."""
        return self.initialized and self.n_updates >= self.warmup_updates

    def std(self, key: str) -> float:
        """Running std = sqrt(E[Q^2] - E[Q]^2), floored at eps."""
        var = self.sq_mean[key] - self.mean[key] ** 2
        return max(var, 0.0) ** 0.5 + self.eps


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------

def sac_update(
    actor: TrainablePolicy,
    q1s: Dict[str, QCriticMLP],
    q2s: Dict[str, QCriticMLP],
    q1_targets: Dict[str, QCriticMLP],
    q2_targets: Dict[str, QCriticMLP],
    log_alpha: torch.Tensor,
    target_entropy: float,
    actor_optimizer: torch.optim.Optimizer,
    q1_optimizers: Dict[str, torch.optim.Optimizer],
    q2_optimizers: Dict[str, torch.optim.Optimizer],
    alpha_optimizer: Optional[torch.optim.Optimizer],
    batch: Dict[str, Any],
    gammas: Dict[str, float],
    reward_keys: Tuple[str, ...],
    stage_weights: Tuple[float, ...],
    tau: float,
    grad_clip_norm: float,
    reward_scale: float = 1.0,
    experiment: Optional[Experiment] = None,
    q_running_stats: Optional[QRunningStats] = None,
) -> Dict[str, float]:
    """Multi-critic SAC gradient step on a minibatch.

    Each reward component has its own pair of twin Q-networks.  Q-targets
    are computed per component with per-component gamma.

    **PPO-aligned actor loss**: each component's Q is normalized by the
    running std of its Q-TARGETS, then combined with ``stage_weights``.  This
    is the direct analog of PPO's per-component advantage normalization
    ``(A - mean) / std``: it puts every component's action-gradient on a
    comparable scale so ``stage_weights`` control relative importance and a
    strong component (e.g. r_fall) no longer drowns out a weak one
    (e.g. r_cross).

    Because SAC's actor loss balances Q against the entropy term
    ``alpha * logpi``, normalizing Q shrinks its scale relative to a *fixed*
    alpha and would let entropy dominate.  This is why **auto-alpha must be
    enabled**: the temperature self-adjusts to hold the policy entropy at
    ``target_entropy`` regardless of the normalized Q scale, decoupling the
    exploration/exploitation balance from the Q magnitude.

    Sample weights from ``prepare_training_segments`` are applied to both
    Q-losses and actor loss, matching PPO's sample-weight handling.

    Args:
        reward_scale: Factor to multiply rewards by before computing Q-targets.
        stage_weights: Normalized weights for combining Q-values in actor loss.
        experiment: Optional experiment for normalize_advantages hook.
        q_running_stats: Running Q-target stats for stable normalization.

    Returns a dict of scalar diagnostics.
    """
    obs = batch["obs"]
    actions = batch["actions"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]
    sample_weights = batch.get("sample_weights", torch.ones_like(dones))

    # Normalize sample weights (matching PPO's batch_weights normalization)
    sw_mean = sample_weights.mean() + 1e-8
    batch_weights = sample_weights / sw_mean

    alpha = log_alpha.exp().detach()

    # Normalize stage weights for actor loss
    w_total = sum(stage_weights)
    if w_total <= 0:
        norm_w = [1.0 / len(stage_weights)] * len(stage_weights)
    else:
        norm_w = [w / w_total for w in stage_weights]

    # ------------------------------------------------------------------
    # 1. Compute per-component target Q-values
    # ------------------------------------------------------------------
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample_action(next_obs)

    q_targets: Dict[str, torch.Tensor] = {}
    for key in reward_keys:
        rewards_k = batch[f"rewards_{key}"] * reward_scale
        gamma_k = gammas[key]
        with torch.no_grad():
            q1_next = q1_targets[key](next_obs, next_actions)
            q2_next = q2_targets[key](next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_targets[key] = rewards_k + (1.0 - dones) * gamma_k * q_next

    # Update running Q-target statistics (stable normalizer for actor loss)
    if q_running_stats is not None:
        q_running_stats.update(q_targets)

    # ------------------------------------------------------------------
    # 2. Update Q-critics (per component, with sample weights)
    # ------------------------------------------------------------------
    q_losses: Dict[str, float] = {}
    q_grads: Dict[str, float] = {}
    q_means: Dict[str, float] = {}
    for key in reward_keys:
        q1 = q1s[key]
        q2 = q2s[key]
        q1_pred = q1(obs, actions)
        q1_loss = ((F.mse_loss(q1_pred, q_targets[key], reduction="none") * batch_weights).mean())
        q1_optimizers[key].zero_grad()
        q1_loss.backward(retain_graph=True)
        q1_grad = torch.nn.utils.clip_grad_norm_(q1.parameters(), grad_clip_norm)
        q1_optimizers[key].step()

        q2_pred = q2(obs, actions)
        q2_loss = ((F.mse_loss(q2_pred, q_targets[key], reduction="none") * batch_weights).mean())
        q2_optimizers[key].zero_grad()
        q2_loss.backward(retain_graph=True)
        q2_grad = torch.nn.utils.clip_grad_norm_(q2.parameters(), grad_clip_norm)
        q2_optimizers[key].step()

        q_losses[f"q1_loss_{key}"] = float(q1_loss.item())
        q_losses[f"q2_loss_{key}"] = float(q2_loss.item())
        q_grads[f"grad_norm_q1_{key}"] = float(q1_grad)
        q_grads[f"grad_norm_q2_{key}"] = float(q2_grad)
        q_means[f"q1_mean_{key}"] = float(q1_pred.mean().item())
        q_means[f"q2_mean_{key}"] = float(q2_pred.mean().item())

    # ------------------------------------------------------------------
    # 3. Update actor (PPO-aligned: per-component Q normalization, then combine)
    # ------------------------------------------------------------------
    new_actions, new_log_probs = actor.sample_action(obs)

    # Per-component normalization, analogous to PPO's (A - mean) / std.
    # We divide each component's Q by the running std of its Q-targets so
    # that every component contributes an action-gradient of comparable
    # scale; stage_weights then control the relative importance.  The mean
    # subtraction is a no-op for the gradient (constant) but keeps
    # combined_q centered for interpretability.  Before warmup we fall back
    # to raw Q (std=1, mean=0) to avoid amplifying untrained-critic noise.
    use_norm = q_running_stats is not None and q_running_stats.ready
    combined_q = torch.zeros_like(new_log_probs)
    norm_q_std: Dict[str, float] = {}
    for w, key in zip(norm_w, reward_keys):
        if w == 0.0:
            continue
        q1_new = q1s[key](obs, new_actions)
        q2_new = q2s[key](obs, new_actions)
        q_min = torch.min(q1_new, q2_new)

        if use_norm:
            r_mean = q_running_stats.mean[key]
            r_std = q_running_stats.std(key)
            q_norm = (q_min - r_mean) / r_std
        else:
            q_norm = q_min

        norm_q_std[f"norm_q_std_{key}"] = float(q_norm.std().item())
        combined_q = combined_q + w * q_norm

    actor_loss = ((alpha * new_log_probs - combined_q) * batch_weights).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
    actor_optimizer.step()

    # ------------------------------------------------------------------
    # 4. Update temperature (alpha)
    # ------------------------------------------------------------------
    alpha_loss_val = 0.0
    if alpha_optimizer is not None:
        alpha_loss = -(log_alpha * (new_log_probs.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            log_alpha.clamp_(-5.0, 2.0)
        alpha_loss_val = float(alpha_loss.item())

    # ------------------------------------------------------------------
    # 5. Soft target update (all components)
    # ------------------------------------------------------------------
    with torch.no_grad():
        for key in reward_keys:
            for p, p_tgt in zip(q1s[key].parameters(), q1_targets[key].parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
            for p, p_tgt in zip(q2s[key].parameters(), q2_targets[key].parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # Aggregate diagnostics
    result: Dict[str, float] = {
        "actor_loss": float(actor_loss.item()),
        "alpha_loss": alpha_loss_val,
        "alpha": float(alpha.item()),
        "log_prob_mean": float(new_log_probs.mean().item()),
        "grad_norm_actor": float(actor_grad),
    }
    result.update(q_losses)
    result.update(q_grads)
    result.update(q_means)
    result.update(norm_q_std)
    result["q_norm_enabled"] = 1.0 if use_norm else 0.0
    if q_running_stats is not None:
        for key in reward_keys:
            result[f"qtgt_std_{key}"] = float(q_running_stats.std(key))
    # Summary stats (averaged across components)
    result["q1_loss"] = float(np.mean([q_losses[f"q1_loss_{k}"] for k in reward_keys]))
    result["q2_loss"] = float(np.mean([q_losses[f"q2_loss_{k}"] for k in reward_keys]))
    result["q_target_mean"] = float(np.mean([q_means[f"q1_mean_{k}"] for k in reward_keys]))
    return result


def soft_copy(source: nn.Module, target: nn.Module) -> None:
    """Hard copy: target = source (for initialization)."""
    target.load_state_dict(source.state_dict())
