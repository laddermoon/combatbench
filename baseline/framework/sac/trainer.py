"""Standard Shannon-entropy SAC update over ``sac_replay_v1`` batches.

P2-TRAIN-1 deliberately implements only the audited 1-step contract:
per-channel twin Q targets, timeout bootstrap, actor entropy loss, optional
automatic temperature, and soft target updates.  Unsupported extensions are
rejected rather than silently approximated.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experiment import SACParams, SACRewardChannel
from .networks import MultiHeadQCritic


class SACTrainerError(RuntimeError):
    """Raised when a SAC update input or configuration is unsupported."""


def _require_tensor(batch: Dict[str, Any], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise SACTrainerError(f"SAC batch missing tensor {name!r}")
    if not torch.isfinite(value).all():
        raise SACTrainerError(f"SAC batch tensor {name!r} contains NaN/Inf")
    return value


def _validate_batch(
    batch: Dict[str, Any],
    channels: Tuple[SACRewardChannel, ...],
) -> None:
    required = (
        "obs", "actions", "next_obs", "rewards", "channel_valid",
        "terminated", "truncated", "bootstrap", "actor_gate",
        "actor_weight", "sample_weight", "sample_ids", "source_keys",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise SACTrainerError(f"SAC batch missing required fields {missing}")
    obs = _require_tensor(batch, "obs")
    actions = _require_tensor(batch, "actions")
    next_obs = _require_tensor(batch, "next_obs")
    rewards = _require_tensor(batch, "rewards")
    channel_valid = batch["channel_valid"]
    terminated = batch["terminated"]
    truncated = batch["truncated"]
    bootstrap = _require_tensor(batch, "bootstrap")
    actor_gate = _require_tensor(batch, "actor_gate")
    actor_weight = _require_tensor(batch, "actor_weight")
    sample_weight = _require_tensor(batch, "sample_weight")

    B = obs.shape[0]
    C = len(channels)
    if obs.ndim != 2 or actions.ndim != 2 or next_obs.shape != obs.shape:
        raise SACTrainerError("invalid obs/actions/next_obs batch shapes")
    if rewards.shape != (B, C):
        raise SACTrainerError(f"rewards shape {tuple(rewards.shape)} != {(B, C)}")
    if channel_valid.shape != (B, C) or channel_valid.dtype != torch.bool:
        raise SACTrainerError("channel_valid must be bool with shape (B,C)")
    if actor_gate.shape != (B, C) or actor_weight.shape != (B, C):
        raise SACTrainerError("actor_gate/actor_weight must have shape (B,C)")
    for name, value in (
        ("terminated", terminated), ("truncated", truncated),
        ("bootstrap", bootstrap), ("sample_weight", sample_weight),
    ):
        if value.shape != (B,):
            raise SACTrainerError(f"{name} must have shape {(B,)}")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise SACTrainerError("terminated/truncated must be bool")
    if torch.logical_and(terminated, truncated).any():
        raise SACTrainerError("terminated and truncated cannot both be true")
    if (terminated & (bootstrap != 0.0)).any():
        raise SACTrainerError("terminated transitions must have bootstrap=0")
    if (truncated & (bootstrap != 1.0)).any():
        raise SACTrainerError("truncated transitions must bootstrap")
    if (actor_gate < 0).any() or (actor_weight < 0).any():
        raise SACTrainerError("actor gate/weight cannot be negative")
    if (sample_weight <= 0).any():
        raise SACTrainerError("sample_weight must be positive")
    if len(batch["source_keys"]) != B:
        raise SACTrainerError("source_keys length does not match batch")
    sample_ids = batch["sample_ids"]
    if len(torch.unique(sample_ids)) != B:
        raise SACTrainerError("SAC batch contains duplicate sample_id")


def validate_sac_channels(channels: Tuple[SACRewardChannel, ...]) -> None:
    for ch in channels:
        if int(ch.n_step) != 1:
            raise SACTrainerError(
                f"channel {ch.name!r} uses n_step={ch.n_step}; "
                "P2-TRAIN-1 supports only n_step=1"
            )
        if int(ch.n_critics) != 2 or int(ch.in_target_min) != 2:
            raise SACTrainerError(
                f"channel {ch.name!r} requests unsupported critic ensemble "
                "configuration; first version supports exactly twin Q"
            )


def compute_critic_targets(
    actor: nn.Module,
    critic: MultiHeadQCritic,
    batch: Dict[str, Any],
    channels: Tuple[SACRewardChannel, ...],
    *,
    alpha: torch.Tensor,
    reward_scale: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute 1-step per-channel Bellman targets without mutating state."""
    next_obs = _require_tensor(batch, "next_obs").to(device)
    rewards = _require_tensor(batch, "rewards").to(device)
    bootstrap = _require_tensor(batch, "bootstrap").to(device)
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample_action(next_obs)
        targets: Dict[str, torch.Tensor] = {}
        for c, ch in enumerate(channels):
            q1_next = critic.q1_target_forward(next_obs, next_actions, ch.name)
            q2_next = critic.q2_target_forward(next_obs, next_actions, ch.name)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            targets[ch.name] = (
                rewards[:, c] * reward_scale
                + ch.gamma * bootstrap * q_next
            )
        return targets


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
    grad_norm_stats: Optional[Any] = None,
    grad_norm_step: int = 0,
) -> Dict[str, float]:
    """Run one standard 1-step Shannon SAC update."""
    del grad_norm_stats, grad_norm_step
    if sp.use_grad_norm:
        raise SACTrainerError(
            "SAC action-gradient normalization is not part of the approved "
            "first-version trainer; set use_grad_norm=False"
        )
    validate_sac_channels(channels)
    _validate_batch(batch, channels)

    obs = _require_tensor(batch, "obs").to(device)
    actions = _require_tensor(batch, "actions").to(device)
    next_obs = _require_tensor(batch, "next_obs").to(device)
    rewards = _require_tensor(batch, "rewards").to(device)
    channel_valid = batch["channel_valid"].to(device)
    bootstrap = _require_tensor(batch, "bootstrap").to(device)
    actor_weight = _require_tensor(batch, "actor_weight").to(device).detach()
    sample_weight = _require_tensor(batch, "sample_weight").to(device).detach()
    sample_weight = sample_weight / (sample_weight.mean() + 1e-8)

    alpha = log_alpha.exp().detach()
    stats: Dict[str, float] = {}
    q_targets = compute_critic_targets(
        actor,
        critic,
        batch,
        channels,
        alpha=alpha,
        reward_scale=sp.reward_scale,
        device=device,
    )

    critic.zero_grad_all()
    total_critic_loss = torch.zeros((), device=device)
    for c, ch in enumerate(channels):
        mask = channel_valid[:, c].float() * sample_weight
        mask_sum = mask.sum()
        if not torch.isfinite(mask_sum) or mask_sum.item() <= 0:
            raise SACTrainerError(
                f"channel {ch.name!r} has no valid critic samples in batch"
            )
        q1_pred = critic.q1_forward(obs, actions, ch.name)
        q2_pred = critic.q2_forward(obs, actions, ch.name)
        target = q_targets[ch.name]
        q1_loss = (mask * (q1_pred - target).pow(2)).sum() / mask_sum
        q2_loss = (mask * (q2_pred - target).pow(2)).sum() / mask_sum
        (q1_loss + q2_loss).backward(retain_graph=True)
        total_critic_loss = total_critic_loss + q1_loss.detach() + q2_loss.detach()
        stats[f"q1_loss_{ch.name}"] = float(q1_loss.item())
        stats[f"q2_loss_{ch.name}"] = float(q2_loss.item())
        stats[f"q1_mean_{ch.name}"] = float(q1_pred.mean().item())
        stats[f"q2_mean_{ch.name}"] = float(q2_pred.mean().item())
        stats[f"td_abs_mean_{ch.name}"] = float(
            (q1_pred.detach() - target).abs().mean().item()
        )

    torch.nn.utils.clip_grad_norm_(list(critic.all_parameters()), grad_clip_norm)
    critic.step_all()

    new_actions, new_log_probs = actor.sample_action(obs)
    q1_all = critic.q1_forward_all(obs, new_actions)
    q2_all = critic.q2_forward_all(obs, new_actions)
    weighted_q = torch.zeros_like(new_log_probs)
    for c, ch in enumerate(channels):
        q_min = torch.min(q1_all[ch.name], q2_all[ch.name])
        weighted_q = weighted_q + actor_weight[:, c] * q_min
        stats[f"actor_weight_mean_{ch.name}"] = float(actor_weight[:, c].mean().item())
    actor_loss = (
        sample_weight * (alpha * new_log_probs - weighted_q)
    ).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), grad_clip_norm,
    )
    actor_optimizer.step()

    alpha_loss_value = 0.0
    if alpha_optimizer is not None:
        target_entropy = (
            -float(actor.action_dim)
            if sp.target_entropy is None else float(sp.target_entropy)
        )
        alpha_loss = -(
            log_alpha * (new_log_probs.detach() + target_entropy)
        ).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            log_alpha.clamp_(sp.log_alpha_min, sp.log_alpha_max)
        alpha_loss_value = float(alpha_loss.item())

    critic.soft_update(float(sp.tau))

    stats.update({
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(total_critic_loss.item()),
        "alpha_loss": alpha_loss_value,
        "alpha": float(alpha.item()),
        "log_prob_mean": float(new_log_probs.mean().item()),
        "entropy_proxy_mean": float((-new_log_probs).mean().item()),
        "grad_norm_actor": float(actor_grad_norm),
    })
    return stats


sac_update = sac_update_v2


def trainer_state_dict(
    actor: nn.Module,
    critic: MultiHeadQCritic,
    actor_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, Any]:
    """Serialize all train/optimizer state needed for full resume."""
    return {
        "schema": "sac_trainer_v1",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "alpha_optimizer_state_dict": (
            alpha_optimizer.state_dict() if alpha_optimizer is not None else None
        ),
    }


def load_model_state(
    state: Dict[str, Any],
    *,
    actor: nn.Module,
    critic: MultiHeadQCritic,
) -> None:
    """Restore only network weights for an explicit warm start."""
    if state.get("schema") != "sac_trainer_v1":
        raise SACTrainerError(
            f"unsupported trainer schema {state.get('schema')!r}"
        )
    actor.load_state_dict(state["actor_state_dict"])
    critic_state = state["critic_state_dict"]
    if set(critic_state.keys()) != set(critic.groups.keys()):
        raise SACTrainerError(
            f"critic groups mismatch: checkpoint={sorted(critic_state)}, "
            f"current={sorted(critic.groups)}"
        )
    for name, group in critic.groups.items():
        group_state = critic_state[name]
        for key in ("q1", "q2", "q1_target", "q2_target"):
            getattr(group, key).load_state_dict(group_state[key])


def load_trainer_state(
    state: Dict[str, Any],
    *,
    actor: nn.Module,
    critic: MultiHeadQCritic,
    actor_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
) -> None:
    """Strictly restore ``sac_trainer_v1`` state."""
    if state.get("schema") != "sac_trainer_v1":
        raise SACTrainerError(
            f"unsupported trainer schema {state.get('schema')!r}"
        )
    actor.load_state_dict(state["actor_state_dict"])
    critic_state = state["critic_state_dict"]
    if set(critic_state.keys()) != set(critic.groups.keys()):
        raise SACTrainerError(
            f"critic groups mismatch: checkpoint={sorted(critic_state)}, "
            f"current={sorted(critic.groups)}"
        )
    critic.load_state_dict(critic_state)
    actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
    log_alpha.data.copy_(state["log_alpha"].to(log_alpha.device))
    saved_alpha_opt = state.get("alpha_optimizer_state_dict")
    if (alpha_optimizer is None) != (saved_alpha_opt is None):
        raise SACTrainerError("alpha optimizer presence does not match checkpoint")
    if alpha_optimizer is not None:
        alpha_optimizer.load_state_dict(saved_alpha_opt)


__all__ = [
    "SACTrainerError",
    "compute_critic_targets",
    "load_model_state",
    "load_trainer_state",
    "sac_update",
    "sac_update_v2",
    "trainer_state_dict",
    "validate_sac_channels",
]
