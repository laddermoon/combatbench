"""Standard Shannon-entropy SAC update over ``sac_replay_v2`` batches.

P2-TRAIN-1 deliberately implements only the audited 1-step contract:
per-channel twin Q targets, timeout bootstrap, actor entropy loss, optional
automatic temperature, and soft target updates.  Unsupported extensions are
rejected rather than silently approximated.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        "actor_weight", "actor_gate_next", "actor_weight_next",
        "sample_weight", "sample_ids", "source_keys",
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
    actor_gate_next = _require_tensor(batch, "actor_gate_next")
    actor_weight_next = _require_tensor(batch, "actor_weight_next")
    sample_weight = _require_tensor(batch, "sample_weight")

    B = obs.shape[0]
    C = len(channels)
    if obs.ndim != 2 or actions.ndim != 2 or next_obs.shape != obs.shape:
        raise SACTrainerError("invalid obs/actions/next_obs batch shapes")
    if rewards.shape != (B, C):
        raise SACTrainerError(f"rewards shape {tuple(rewards.shape)} != {(B, C)}")
    if channel_valid.shape != (B, C) or channel_valid.dtype != torch.bool:
        raise SACTrainerError("channel_valid must be bool with shape (B,C)")
    if (
        actor_gate.shape != (B, C)
        or actor_weight.shape != (B, C)
        or actor_gate_next.shape != (B, C)
        or actor_weight_next.shape != (B, C)
    ):
        raise SACTrainerError("actor gate/weight fields must have shape (B,C)")
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
    if (
        (actor_gate < 0).any()
        or (actor_weight < 0).any()
        or (actor_gate_next < 0).any()
        or (actor_weight_next < 0).any()
    ):
        raise SACTrainerError("actor gate/weight fields cannot be negative")
    for gate, weight, label in (
        (actor_gate, actor_weight, "actor"),
        (actor_gate_next, actor_weight_next, "actor next"),
    ):
        gate_sum = gate.sum(dim=1)
        if (gate_sum <= 0).any():
            raise SACTrainerError(f"{label} gate row sums must be positive")
        if not torch.allclose(
            weight, gate / gate_sum[:, None], atol=1e-5, rtol=1e-5,
        ):
            raise SACTrainerError(f"{label} weight must equal normalized gate")
    if (sample_weight <= 0).any():
        raise SACTrainerError("sample_weight must be positive")
    if len(batch["source_keys"]) != B:
        raise SACTrainerError("source_keys length does not match batch")
    sample_ids = batch["sample_ids"]
    if len(torch.unique(sample_ids)) != B:
        raise SACTrainerError("SAC batch contains duplicate sample_id")


def validate_sac_channels(channels: Tuple[SACRewardChannel, ...]) -> None:
    if not channels:
        raise SACTrainerError("SAC requires at least one reward channel")
    gamma = float(channels[0].gamma)
    for ch in channels:
        if float(ch.gamma) != gamma:
            raise SACTrainerError(
                "phase-three SAC supports one common cohort gamma; "
                f"channel {ch.name!r} uses {ch.gamma}, expected {gamma}"
            )
        if int(ch.n_step) != 1:
            raise SACTrainerError(
                f"channel {ch.name!r} uses n_step={ch.n_step}; "
                "SAC supports only n_step=1"
            )
        if int(ch.n_critics) != 2 or int(ch.in_target_min) != 2:
            raise SACTrainerError(
                f"channel {ch.name!r} requests unsupported critic ensemble "
                "configuration; first version supports exactly twin Q"
            )


def _require_expectation_actor(actor: nn.Module) -> int:
    """Return K and fail loud if the actor lacks the A4.3 contract."""
    fn = getattr(actor, "expectation_samples", None)
    if not callable(fn):
        raise SACTrainerError(
            f"actor {type(actor).__name__} does not implement "
            "expectation_samples (A4.3); legacy_tanh actors are "
            "warm-start only and cannot train under sac_update_v3"
        )
    return int(getattr(actor, "num_components", 1))


@dataclass(frozen=True)
class _RegularizerArgs:
    regularizer_mode: str
    reg_lambda: float
    u_floor: float
    u_kind: str


def _resolve_u_kind(actor: nn.Module, u_kind: str) -> str:
    if u_kind == "native":
        return "l2" if bool(getattr(actor, "is_mixture", False)) else "peak"
    if u_kind in ("peak", "l2"):
        return u_kind
    raise SACTrainerError(
        f"u_kind must be 'native', 'peak' or 'l2', got {u_kind!r}"
    )


def _state_regularizer(
    actor: nn.Module, obs: torch.Tensor, args: _RegularizerArgs,
) -> torch.Tensor:
    """Bθ(s): per-state regularizer evaluated with the CURRENT π.

    u_bonus → λ·U(s); u_floor → −λ·relu(f−U(s))².  Differentiable when
    called outside no_grad (actor side); callers wrap for the detached
    target side.
    """
    u = actor.uncertainty(obs, _resolve_u_kind(actor, args.u_kind))
    lam = float(args.reg_lambda)
    if args.regularizer_mode == "u_bonus":
        return lam * u
    if args.regularizer_mode == "u_floor":
        return -lam * torch.relu(float(args.u_floor) - u).pow(2)
    raise SACTrainerError(
        f"_state_regularizer called with mode {args.regularizer_mode!r}"
    )


def _validate_regularizer(
    actor: nn.Module, sp: SACParams,
    alpha_optimizer: Optional[torch.optim.Optimizer],
) -> None:
    mode = sp.regularizer_mode
    if mode not in ("shannon", "u_bonus", "u_floor"):
        raise SACTrainerError(
            f"regularizer_mode must be shannon/u_bonus/u_floor, got {mode!r}"
        )
    if mode == "shannon":
        return
    if alpha_optimizer is not None:
        raise SACTrainerError(
            f"regularizer_mode={mode!r} does not accept an alpha "
            "optimizer; no implicit Shannon+temperature stacking (A4.6)"
        )
    if not callable(getattr(actor, "uncertainty", None)):
        raise SACTrainerError(
            f"actor {type(actor).__name__} has no uncertainty(); "
            f"regularizer_mode={mode!r} requires the A4.3 actor contract"
        )
    if not (sp.reg_lambda >= 0.0):
        raise SACTrainerError(
            f"reg_lambda must be >= 0 for mode {mode!r}, got {sp.reg_lambda}"
        )
    if mode == "u_floor" and not (0.0 <= sp.u_floor <= 1.0):
        raise SACTrainerError(
            f"u_floor f must be in [0,1] for mode 'u_floor', got {sp.u_floor}"
        )
    _resolve_u_kind(actor, sp.u_kind)


def _draw_u(
    rng: torch.Generator, shape: Tuple[int, ...], device: torch.device,
) -> torch.Tensor:
    return torch.rand(shape, generator=rng, dtype=torch.float32).to(device)


def _forward_all_candidates(
    forward_fn,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Run a [B,...]-shaped critic forward over [B,K,M,D] candidates.

    Returns [B,K,M].
    """
    B, Kc, M, D = actions.shape
    obs_rep = (
        obs[:, None, None, :]
        .expand(B, Kc, M, obs.shape[-1])
        .reshape(B * Kc * M, obs.shape[-1])
    )
    flat = forward_fn(obs_rep, actions.reshape(B * Kc * M, D))
    return flat.view(B, Kc, M)


def compute_critic_targets(
    actor: nn.Module,
    critic: MultiHeadQCritic,
    batch: Dict[str, Any],
    channels: Tuple[SACRewardChannel, ...],
    *,
    alpha: torch.Tensor,
    reward_scale: float,
    device: torch.device,
    n_expectation_samples: int = 1,
    expectation_rng: Optional[torch.Generator] = None,
    regularizer_mode: str = "shannon",
    reg_lambda: float = 0.0,
    u_floor: float = 0.0,
    u_kind: str = "native",
    capture: Optional[Dict[str, Any]] = None,
    target_info: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute A3 1-step targets over enumerated π candidates (A4.4).

    For every candidate a_km the joint twin pair j* is selected from
    ``F_j = Σ_c w_next_c · Q_{c,j}`` first, then the per-candidate values
    are integrated with ``integration_weights`` — twin selection happens
    BEFORE the candidate expectation; the order is not interchangeable.
    """
    num_components = _require_expectation_actor(actor)
    next_obs = _require_tensor(batch, "next_obs").to(device)
    rewards = _require_tensor(batch, "rewards").to(device)
    bootstrap = _require_tensor(batch, "bootstrap").to(device)
    actor_weight_next = _require_tensor(batch, "actor_weight_next").to(device)
    B = next_obs.shape[0]
    M = int(n_expectation_samples)
    if M < 1:
        raise SACTrainerError(
            f"expectation_samples must be >= 1, got {M}"
        )
    if expectation_rng is None:
        expectation_rng = torch.Generator()
        expectation_rng.manual_seed(0)
    u_next = _draw_u(
        expectation_rng,
        (B, num_components, M, actor.action_dim),
        device,
    )
    with torch.no_grad():
        next_actions, next_log_probs, integ_w = actor.expectation_samples(
            next_obs, u_next,
        )
        Kc, M = next_actions.shape[1], next_actions.shape[2]
        C = len(channels)
        q1_next = torch.stack(
            [
                _forward_all_candidates(
                    lambda o, a: critic.q1_target_forward(o, a, ch.name),
                    next_obs, next_actions,
                )
                for ch in channels
            ],
            dim=-1,
        )                                                  # [B,K,M,C]
        q2_next = torch.stack(
            [
                _forward_all_candidates(
                    lambda o, a: critic.q2_target_forward(o, a, ch.name),
                    next_obs, next_actions,
                )
                for ch in channels
            ],
            dim=-1,
        )
        q_next_pairs = torch.stack((q1_next, q2_next), dim=-1)  # [B,K,M,C,2]
        w_next = actor_weight_next[:, None, None, :, None]      # [B,1,1,C,1]
        f_next = (q_next_pairs * w_next).sum(dim=3)             # [B,K,M,2]
        pair_next = f_next.argmin(dim=-1)                       # [B,K,M]
        selected_next = q_next_pairs.gather(
            -1,
            pair_next[..., None, None].expand(-1, -1, -1, C, 1),
        ).squeeze(-1)                                           # [B,K,M,C]
        if capture is not None:
            capture["u_next"] = u_next.detach().cpu()
            capture["next_actions"] = next_actions.detach().cpu()
            capture["next_log_probs"] = next_log_probs.detach().cpu()
            capture["next_integration_weights"] = integ_w.detach().cpu()
            capture["target_pair_index"] = pair_next.detach().cpu()
        if target_info is not None:
            target_info["pair_index"] = pair_next
        # Continuation regularizer B(s'): shannon uses the per-candidate
        # entropy term; U modes add the current-π state regularizer once
        # (whole target stays detached).
        if regularizer_mode == "shannon":
            b_next = -alpha * next_log_probs                    # [B,K,M]
            b_state = None
        else:
            b_next = torch.zeros_like(next_log_probs)
            b_state = _state_regularizer(
                actor, next_obs,
                _RegularizerArgs(
                    regularizer_mode=regularizer_mode,
                    reg_lambda=reg_lambda,
                    u_floor=u_floor,
                    u_kind=u_kind,
                ),
            )
            if capture is not None:
                capture["reg_next"] = b_state.detach().cpu()
        targets: Dict[str, torch.Tensor] = {}
        for c, ch in enumerate(channels):
            v_c = (
                integ_w * (selected_next[..., c] + b_next)
            ).sum(dim=(1, 2))                                    # [B]
            if b_state is not None:
                v_c = v_c + b_state
            targets[ch.name] = (
                rewards[:, c] * reward_scale
                + ch.gamma * bootstrap * v_c
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
    expectation_rng: Optional[torch.Generator] = None,
    capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Run one 1-step SAC update over enumerated π candidates."""
    del grad_norm_stats, grad_norm_step
    if sp.use_grad_norm:
        raise SACTrainerError(
            "SAC action-gradient normalization is not part of the approved "
            "first-version trainer; set use_grad_norm=False"
        )
    num_components = _require_expectation_actor(actor)
    _validate_regularizer(actor, sp, alpha_optimizer)
    validate_sac_channels(channels)
    _validate_batch(batch, channels)

    obs = _require_tensor(batch, "obs").to(device)
    actions = _require_tensor(batch, "actions").to(device)
    next_obs = _require_tensor(batch, "next_obs").to(device)
    rewards = _require_tensor(batch, "rewards").to(device)
    channel_valid = batch["channel_valid"].to(device)
    bootstrap = _require_tensor(batch, "bootstrap").to(device)
    actor_weight = _require_tensor(batch, "actor_weight").to(device).detach()
    actor_weight_next = _require_tensor(
        batch, "actor_weight_next",
    ).to(device).detach()
    sample_weight = _require_tensor(batch, "sample_weight").to(device).detach()

    alpha = log_alpha.exp().detach()
    stats: Dict[str, float] = {}
    if capture is not None:
        capture.clear()
    if expectation_rng is None:
        expectation_rng = torch.Generator()
        expectation_rng.manual_seed(0)
    target_info: Dict[str, torch.Tensor] = {}
    q_targets = compute_critic_targets(
        actor,
        critic,
        batch,
        channels,
        alpha=alpha,
        reward_scale=sp.reward_scale,
        device=device,
        n_expectation_samples=sp.expectation_samples,
        expectation_rng=expectation_rng,
        regularizer_mode=sp.regularizer_mode,
        reg_lambda=sp.reg_lambda,
        u_floor=sp.u_floor,
        u_kind=sp.u_kind,
        capture=capture,
        target_info=target_info,
    )
    if capture is not None:
        capture["alpha_pre"] = alpha.detach().cpu()
        capture["targets"] = {
            name: target.detach().cpu() for name, target in q_targets.items()
        }

    critic.zero_grad_all()
    total_critic_loss = torch.zeros((), device=device)
    updated_channels: list[str] = []
    for c, ch in enumerate(channels):
        mask = channel_valid[:, c].float() * sample_weight
        mask_sum = mask.sum()
        stats[f"critic_valid_weight_{ch.name}"] = float(mask_sum.item())
        if not torch.isfinite(mask_sum) or mask_sum.item() <= 0:
            stats[f"critic_updated_{ch.name}"] = 0.0
            continue
        q1_pred = critic.q1_forward(obs, actions, ch.name)
        q2_pred = critic.q2_forward(obs, actions, ch.name)
        if capture is not None:
            capture.setdefault("q1_pred", {})[ch.name] = q1_pred.detach().cpu()
            capture.setdefault("q2_pred", {})[ch.name] = q2_pred.detach().cpu()
        target = q_targets[ch.name]
        q1_loss = (mask * (q1_pred - target).pow(2)).sum() / mask_sum
        q2_loss = (mask * (q2_pred - target).pow(2)).sum() / mask_sum
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(critic.channel_parameters(ch.name)), grad_clip_norm,
        )
        critic.step_channel(ch.name)
        updated_channels.append(ch.name)
        total_critic_loss = total_critic_loss + q1_loss.detach() + q2_loss.detach()
        stats[f"critic_updated_{ch.name}"] = 1.0
        stats[f"q1_loss_{ch.name}"] = float(q1_loss.item())
        stats[f"q2_loss_{ch.name}"] = float(q2_loss.item())
        stats[f"q1_mean_{ch.name}"] = float(q1_pred.mean().item())
        stats[f"q2_mean_{ch.name}"] = float(q2_pred.mean().item())
        stats[f"td_abs_mean_{ch.name}"] = float(
            (q1_pred.detach() - target).abs().mean().item()
        )
    if not updated_channels:
        raise SACTrainerError("SAC batch has no valid critic channels")

    actor_rows = channel_valid.all(dim=1)
    actor_mask = actor_rows.float() * sample_weight
    actor_den = actor_mask.sum()
    stats["actor_valid_weight"] = float(actor_den.item())
    stats["actor_valid_count"] = float(actor_rows.sum().item())
    if not torch.isfinite(actor_den) or actor_den.item() <= 0:
        raise SACTrainerError("SAC batch has no actor-valid rows")

    M = int(sp.expectation_samples)
    if M < 1:
        raise SACTrainerError(f"expectation_samples must be >= 1, got {M}")
    u_actor = _draw_u(
        expectation_rng, (obs.shape[0], num_components, M, actor.action_dim),
        device,
    )
    new_actions, new_log_probs, integ_w = actor.expectation_samples(
        obs, u_actor,
    )
    if capture is not None:
        capture["u_actor"] = u_actor.detach().cpu()
        capture["new_actions"] = new_actions.detach().cpu()
        capture["new_log_probs"] = new_log_probs.detach().cpu()
        capture["actor_integration_weights"] = integ_w.detach().cpu()

    requires_grad = [
        (param, param.requires_grad) for param in critic.all_parameters()
    ]
    for param, _ in requires_grad:
        param.requires_grad_(False)
    C = len(channels)
    q1_all = torch.stack(
        [
            _forward_all_candidates(
                lambda o, a: critic.q1_forward(o, a, ch.name),
                obs, new_actions,
            )
            for ch in channels
        ],
        dim=-1,
    )                                                      # [B,K,M,C]
    q2_all = torch.stack(
        [
            _forward_all_candidates(
                lambda o, a: critic.q2_forward(o, a, ch.name),
                obs, new_actions,
            )
            for ch in channels
        ],
        dim=-1,
    )
    for param, required in requires_grad:
        param.requires_grad_(required)

    q_online_pairs = torch.stack((q1_all, q2_all), dim=-1)  # [B,K,M,C,2]
    w_actor = actor_weight[:, None, None, :, None]          # [B,1,1,C,1]
    f_actor = (q_online_pairs * w_actor).sum(dim=3)         # [B,K,M,2]
    pair_actor = f_actor.argmin(dim=-1)                     # [B,K,M]
    f_selected = f_actor.gather(-1, pair_actor[..., None]).squeeze(-1)
    # F*_km already is Σ_c w_c·Q_{c,j*} — the joint soft-Q per candidate.
    is_shannon = sp.regularizer_mode == "shannon"
    if is_shannon:
        candidate_actor_term = integ_w * (
            alpha.detach() * new_log_probs - f_selected
        )                                                  # [B,K,M]
    else:
        candidate_actor_term = integ_w * (-f_selected)
    weighted_q = candidate_actor_term.sum(dim=(1, 2))       # [B]
    # U regularizer is a per-state term, applied once — not per candidate.
    reg_term = None
    if not is_shannon:
        b_state = _state_regularizer(
            actor, obs,
            _RegularizerArgs(
                regularizer_mode=sp.regularizer_mode,
                reg_lambda=sp.reg_lambda,
                u_floor=sp.u_floor,
                u_kind=sp.u_kind,
            ),
        )
        reg_term = -b_state                                # actor: −Bθ(s)
    if capture is not None:
        capture["actor_pair_index"] = pair_actor.detach().cpu()
        capture["weighted_q"] = weighted_q.detach().cpu()
        if reg_term is not None:
            capture["reg_actor"] = reg_term.detach().cpu()
    actor_objective = weighted_q + (
        reg_term if reg_term is not None else torch.zeros_like(weighted_q)
    )
    actor_loss = (actor_mask * actor_objective).sum() / actor_den
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), grad_clip_norm,
    )
    actor_optimizer.step()

    # Entropy estimate for auto-alpha: the enumerated joint logπ
    # expectation −Σ_km w·logp (detached), per A4.7.
    logp_bar = (integ_w.detach() * new_log_probs.detach()).sum(dim=(1, 2))
    alpha_loss_value = 0.0
    if alpha_optimizer is not None:
        target_entropy = (
            -float(actor.action_dim)
            if sp.target_entropy is None else float(sp.target_entropy)
        )
        alpha_loss = -(
            actor_mask
            * log_alpha
            * (logp_bar + target_entropy)
        ).sum() / actor_den
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            log_alpha.clamp_(sp.log_alpha_min, sp.log_alpha_max)
        alpha_loss_value = float(alpha_loss.item())

    for channel_name in updated_channels:
        critic.soft_update_channel(channel_name, float(sp.tau))

    stats.update({
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(total_critic_loss.item()),
        "alpha_loss": alpha_loss_value,
        "alpha": float(alpha.item()),
        "log_prob_mean": float(logp_bar.mean().item()),
        "entropy_proxy_mean": float((-logp_bar).mean().item()),
        "grad_norm_actor": float(actor_grad_norm),
        "target_pair1_frac": float(
            target_info["pair_index"].float().mean().item()
        ),
        "actor_pair1_frac": float(pair_actor.float().mean().item()),
    })
    for c, ch in enumerate(channels):
        stats[f"actor_weight_mean_{ch.name}"] = float(
            actor_weight[:, c].mean().item()
        )
        stats[f"actor_weight_next_mean_{ch.name}"] = float(
            actor_weight_next[:, c].mean().item()
        )
    stats["regularizer_shannon"] = 1.0 if is_shannon else 0.0
    if reg_term is not None:
        stats["reg_actor_term_mean"] = float(
            (actor_mask * reg_term.detach()).sum().item() / actor_den.item()
        )
    if capture is not None and "reg_next" in capture:
        stats["reg_next_mean"] = float(capture["reg_next"].mean().item())
    return stats


sac_update = sac_update_v2


def trainer_state_dict(
    actor: nn.Module,
    critic: MultiHeadQCritic,
    actor_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    expectation_rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Serialize all train/optimizer state needed for full resume."""
    return {
        "schema": "sac_trainer_v1",
        "actor_state_dict": actor.state_dict(),
        "actor_rng_state": (
            actor.rng_state() if hasattr(actor, "rng_state") else None
        ),
        "expectation_rng_state": (
            expectation_rng.get_state()
            if expectation_rng is not None else None
        ),
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
    expectation_rng: Optional[torch.Generator] = None,
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
    rng_state = state.get("actor_rng_state")
    if rng_state is not None:
        if not hasattr(actor, "set_rng_state"):
            raise SACTrainerError(
                "checkpoint carries actor_rng_state but actor cannot restore it"
            )
        actor.set_rng_state(rng_state)
    exp_rng_state = state.get("expectation_rng_state")
    if exp_rng_state is not None:
        if expectation_rng is None:
            raise SACTrainerError(
                "checkpoint carries expectation_rng_state but no "
                "expectation_rng was provided for restore"
            )
        expectation_rng.set_state(exp_rng_state)
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
