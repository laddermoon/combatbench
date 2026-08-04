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

from .experiment import CommonParams, Experiment, SACParams, Segment, TrainablePolicy


# ---------------------------------------------------------------------------
# Segment resolution — v2 API (prepare_segments) with v1 fallback
# ---------------------------------------------------------------------------

def _tuples_to_segments(
    raw: List,
) -> List[Segment]:
    """Convert old-style tuples to Segment objects."""
    result = []
    for seg in raw:
        if isinstance(seg, Segment):
            result.append(seg)
        elif len(seg) == 3:
            result.append(Segment(start=seg[0], end=seg[1], weight=seg[2]))
        elif len(seg) == 4:
            result.append(Segment(start=seg[0], end=seg[1], weight=seg[2], mode=seg[3]))
        else:
            raise ValueError(f"Invalid segment tuple: {seg}")
    return result


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
        key_active: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Add a single transition with per-component rewards and sample weight.

        Args:
            key_active: Per-key active flags for this transition.  If a key
                is inactive (False), its reward is stored as 0 and a mask
                entry is recorded.  None = all keys active (backward compat).
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        for k in self.reward_keys:
            is_active = True if key_active is None else key_active.get(k, True)
            self.rewards[k][self.ptr] = rewards.get(k, 0.0) if is_active else 0.0
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

        # Resolve segments: try v2 API first, fall back to v1.
        segs = experiment.prepare_segments(episode)
        if segs is None:
            raw = experiment.prepare_training_segments(episode)
            segs = _tuples_to_segments(raw)
        if not segs:
            return 0

        is_terminated = bool(episode.is_terminated)
        n_added = 0
        for seg in segs:
            start, end, weight = seg.start, seg.end, seg.weight
            T_seg = end - start
            if T_seg == 0:
                continue

            # Per-key active flags for this segment.
            if seg.key_weights is not None:
                seg_key_active = {
                    k: seg.key_weights.get(k, 0.0) > 0.0
                    for k in self.reward_keys
                }
            else:
                seg_key_active = None

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
                    key_active=seg_key_active,
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
# Running reward statistics for source-level normalization (Route B)
# ---------------------------------------------------------------------------

class RewardRunningStats:
    """Per-component running std of raw rewards (source-level normalization).

    Tracks an EMA of ``E[r^2]`` for each reward component.  Rewards are
    divided by ``sqrt(E[r^2])`` before entering the Bellman target, putting
    all components on a comparable scale so that ``stage_weights`` control
    relative importance — the same goal as PPO's per-component advantage
    normalization, but applied at the reward source rather than at the
    Q-value or gradient level.

    **Only scales, never subtracts mean**: subtracting a constant from
    rewards in an episodic MDP with terminal states distorts the
    survival-vs-termination trade-off (classic survival-bias problem).
    Pure scaling by a positive constant preserves the optimal policy
    (argmax is invariant to positive scaling of the reward signal).

    **Initialization**: SAC collects ``warmup_steps`` transitions (default
    10,000) before any network update.  We exploit this by computing exact
    statistics from the full buffer at that point, so normalization is
    active from the very first gradient step — no blind warmup period.
    Afterwards, EMA tracks distribution drift as the policy evolves.
    """

    def __init__(
        self,
        reward_keys: Tuple[str, ...],
        ema_decay: float = 0.99,
        eps: float = 1e-6,
        warmup_updates: int = 0,
    ):
        self.reward_keys = reward_keys
        self.ema_decay = ema_decay
        self.eps = eps
        self.warmup_updates = warmup_updates
        self.sq_mean: Dict[str, float] = {k: 0.0 for k in reward_keys}
        self.initialized = False
        self.n_updates = 0

    def initialize_from_buffer(self, buffer: "ReplayBuffer") -> None:
        """Compute exact E[r^2] from all buffered rewards as initial stats.

        Called once after SAC warmup completes, before the first network
        update.  This gives us precise statistics from ~10k real
        transitions, eliminating the need for a gradual warmup.
        """
        for key in self.reward_keys:
            r = buffer.rewards[key][:buffer.size]
            self.sq_mean[key] = float(np.mean(r ** 2))
        self.initialized = True
        self.n_updates = self.warmup_updates

    def update(self, rewards_batch: Dict[str, torch.Tensor]) -> None:
        """Update running stats from a batch of raw rewards (before scaling)."""
        decay = self.ema_decay
        for key in self.reward_keys:
            r = rewards_batch[key].detach()
            b_sq_mean = float((r * r).mean().item())
            if not self.initialized:
                self.sq_mean[key] = b_sq_mean
            else:
                self.sq_mean[key] = decay * self.sq_mean[key] + (1 - decay) * b_sq_mean
        self.initialized = True
        self.n_updates += 1

    @property
    def ready(self) -> bool:
        """True once stats are initialized and warm enough."""
        return self.initialized and self.n_updates >= self.warmup_updates

    def scale(self, key: str) -> float:
        """Running scale = sqrt(E[r^2]), floored at eps via max()."""
        return max(self.sq_mean[key], 0.0) ** 0.5 + self.eps


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
    reward_running_stats: Optional[RewardRunningStats] = None,
) -> Dict[str, float]:
    """Multi-critic SAC gradient step on a minibatch.

    Each reward component has its own pair of twin Q-networks.  Q-targets
    are computed per component with per-component gamma.

    **Source-level reward normalization (Route B)**: each component's raw
    reward is divided by its running ``sqrt(E[r^2])`` before entering the
    Bellman target.  This puts all reward components on a comparable scale
    so that ``stage_weights`` control relative importance — the same goal
    as PPO's per-component advantage normalization, but applied at the
    reward source.  Only scaling (no mean subtraction) to avoid
    survival-bias in episodic MDPs with terminal states.

    After source-level scaling, the actor loss is standard SAC:
    ``combined_q = sum(w_k * Q_k)``, ``actor_loss = (alpha * logpi -
    combined_q).mean()``.  Auto-alpha adjusts the temperature to hold
    policy entropy at ``target_entropy``.

    Sample weights from ``prepare_training_segments`` are applied to both
    Q-losses and actor loss, matching PPO's sample-weight handling.

    Args:
        reward_scale: Factor to multiply rewards by before computing Q-targets.
        stage_weights: Normalized weights for combining Q-values in actor loss.
        experiment: Optional experiment for normalize_advantages hook.
        reward_running_stats: Running reward stats for source-level normalization.

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
    # 1. Compute per-component target Q-values (with source-level reward scaling)
    # ------------------------------------------------------------------
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample_action(next_obs)

    # Update reward running stats from raw rewards (before scaling)
    if reward_running_stats is not None:
        raw_rewards = {key: batch[f"rewards_{key}"] for key in reward_keys}
        reward_running_stats.update(raw_rewards)

    use_rnorm = reward_running_stats is not None and reward_running_stats.ready

    q_targets: Dict[str, torch.Tensor] = {}
    for key in reward_keys:
        raw_r = batch[f"rewards_{key}"]
        if use_rnorm:
            r_scale = reward_running_stats.scale(key)
            rewards_k = (raw_r / r_scale) * reward_scale
        else:
            rewards_k = raw_r * reward_scale
        gamma_k = gammas[key]
        with torch.no_grad():
            q1_next = q1_targets[key](next_obs, next_actions)
            q2_next = q2_targets[key](next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_targets[key] = rewards_k + (1.0 - dones) * gamma_k * q_next

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
    # 3. Update actor (standard SAC: weighted sum of Q-values)
    # ------------------------------------------------------------------
    new_actions, new_log_probs = actor.sample_action(obs)

    # Standard SAC actor loss: weighted sum of per-component Q-values.
    # Reward components are already scale-normalized at the source (step 1),
    # so a simple weighted sum suffices — stage_weights control relative
    # importance, analogous to PPO's weighted advantage sum after normalization.
    combined_q = torch.zeros_like(new_log_probs)
    for w, key in zip(norm_w, reward_keys):
        if w == 0.0:
            continue
        q1_new = q1s[key](obs, new_actions)
        q2_new = q2s[key](obs, new_actions)
        q_min = torch.min(q1_new, q2_new)
        combined_q = combined_q + w * q_min

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
    # Reward normalization diagnostics
    if reward_running_stats is not None:
        for key in reward_keys:
            result[f"reward_scale_{key}"] = float(reward_running_stats.scale(key))
        result["reward_norm_enabled"] = 1.0 if use_rnorm else 0.0
    # Summary stats (averaged across components)
    result["q1_loss"] = float(np.mean([q_losses[f"q1_loss_{k}"] for k in reward_keys]))
    result["q2_loss"] = float(np.mean([q_losses[f"q2_loss_{k}"] for k in reward_keys]))
    result["q_target_mean"] = float(np.mean([q_means[f"q1_mean_{k}"] for k in reward_keys]))
    return result


def soft_copy(source: nn.Module, target: nn.Module) -> None:
    """Hard copy: target = source (for initialization)."""
    target.load_state_dict(source.state_dict())
