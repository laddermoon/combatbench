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
    """Fixed-capacity circular replay buffer storing (s, a, r, s', done).

    Rewards are stored as a single combined scalar (weighted sum of reward
    components, using the current stage weights at insertion time).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.size = 0
        self.ptr = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
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

        Rewards are combined into a single scalar using stage_weights.

        Returns the number of transitions added.
        """
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        obs = episode.observations.get(ep_target)
        acts = episode.actions.get(ep_target)
        fin = episode.final_observation.get(ep_target)
        if obs is None or acts is None or fin is None:
            return 0

        T = int(acts.shape[0])
        if T == 0:
            return 0

        # Extract rewards and combine
        reward_dict = experiment.extract_rewards(episode)
        reward_keys = common_params.reward_keys

        # Normalize weights
        w_total = sum(stage_weights)
        if w_total <= 0:
            norm_w = tuple(1.0 / len(stage_weights) for _ in stage_weights)
        else:
            norm_w = tuple(w / w_total for w in stage_weights)

        # Combine rewards into single scalar per step
        combined_rewards = np.zeros(T, dtype=np.float32)
        for w, key in zip(norm_w, reward_keys):
            if w == 0.0:
                continue
            r = reward_dict.get(key, np.zeros(T, dtype=np.float32))
            combined_rewards += float(w) * r

        # Add transitions
        is_terminated = bool(episode.is_terminated)
        for t in range(T):
            if t < T - 1:
                next_o = obs[t + 1]
                done = False
            else:
                next_o = fin
                done = is_terminated

            self.add(
                obs=obs[t].astype(np.float32),
                action=acts[t].astype(np.float32),
                reward=float(combined_rewards[t]),
                next_obs=next_o.astype(np.float32),
                done=done,
            )

        return T

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample a random minibatch and return as GPU tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        }


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------

def sac_update(
    actor: TrainablePolicy,
    q1: QCriticMLP,
    q2: QCriticMLP,
    q1_target: QCriticMLP,
    q2_target: QCriticMLP,
    log_alpha: torch.Tensor,
    target_entropy: float,
    actor_optimizer: torch.optim.Optimizer,
    q1_optimizer: torch.optim.Optimizer,
    q2_optimizer: torch.optim.Optimizer,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    batch: Dict[str, torch.Tensor],
    gamma: float,
    tau: float,
    grad_clip_norm: float,
    reward_scale: float = 1.0,
) -> Dict[str, float]:
    """Single SAC gradient step on a minibatch.

    Args:
        reward_scale: Factor to multiply rewards by before computing Q-targets.
            Use < 1.0 to stabilize training when reward magnitudes are large.

    Returns a dict of scalar diagnostics.
    """
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"] * reward_scale
    next_obs = batch["next_obs"]
    dones = batch["dones"]

    alpha = log_alpha.exp().detach()

    # ------------------------------------------------------------------
    # 1. Compute target Q-value
    # ------------------------------------------------------------------
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample_action(next_obs)
        q1_next = q1_target(next_obs, next_actions)
        q2_next = q2_target(next_obs, next_actions)
        q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
        q_target = rewards + (1.0 - dones) * gamma * q_next

    # ------------------------------------------------------------------
    # 2. Update Q-critics
    # ------------------------------------------------------------------
    q1_pred = q1(obs, actions)
    q1_loss = F.mse_loss(q1_pred, q_target)
    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_grad = torch.nn.utils.clip_grad_norm_(q1.parameters(), grad_clip_norm)
    q1_optimizer.step()

    q2_pred = q2(obs, actions)
    q2_loss = F.mse_loss(q2_pred, q_target)
    q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_grad = torch.nn.utils.clip_grad_norm_(q2.parameters(), grad_clip_norm)
    q2_optimizer.step()

    # ------------------------------------------------------------------
    # 3. Update actor
    # ------------------------------------------------------------------
    new_actions, new_log_probs = actor.sample_action(obs)
    q1_new = q1(obs, new_actions)
    q2_new = q2(obs, new_actions)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha * new_log_probs - q_new).mean()

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
        # Clamp log_alpha to prevent collapse / explosion
        with torch.no_grad():
            log_alpha.clamp_(-5.0, 2.0)
        alpha_loss_val = float(alpha_loss.item())

    # ------------------------------------------------------------------
    # 5. Soft target update
    # ------------------------------------------------------------------
    with torch.no_grad():
        for p, p_tgt in zip(q1.parameters(), q1_target.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for p, p_tgt in zip(q2.parameters(), q2_target.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    return {
        "q1_loss": float(q1_loss.item()),
        "q2_loss": float(q2_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "alpha_loss": alpha_loss_val,
        "alpha": float(alpha.item()),
        "q1_mean": float(q1_pred.mean().item()),
        "q2_mean": float(q2_pred.mean().item()),
        "q_target_mean": float(q_target.mean().item()),
        "log_prob_mean": float(new_log_probs.mean().item()),
        "grad_norm_actor": float(actor_grad),
        "grad_norm_q1": float(q1_grad),
        "grad_norm_q2": float(q2_grad),
    }


def soft_copy(source: nn.Module, target: nn.Module) -> None:
    """Hard copy: target = source (for initialization)."""
    target.load_state_dict(source.state_dict())
