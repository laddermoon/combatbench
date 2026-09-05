"""Mixture of diagonal Gaussians policy (Stage 3).

The first family with genuine multimodality: for a single state, the
raw distribution can have ``K`` modes, allowing the policy to represent
"step left" and "step right" as separate high-density regions instead
of compromising on a single mean in the (bad) middle.

Sampling uses Gumbel-max component selection + per-component rsample.
Scoring uses ``logsumexp(log_weight + component_log_prob)``.  The tanh
Jacobian is added **after** the logsumexp (it depends only on the
action being scored, not on the component).

Entropy has no closed form for a mixture; a sampled estimate
``-mean(log_prob(rsample()))`` is used as the regularizer.

See ``DESIGN_mixture_gaussian.md`` for the full design rationale.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from baseline.framework.ppo.policies.tanh_squashed_base import TanhSquashedPolicyBase

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 0.0
DEFAULT_K = 3


class MoGTanhMLPPolicy(TanhSquashedPolicyBase):
    """Tanh-squashed mixture of diagonal Gaussians.

    Architecture:
        trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
        head:  Linear(hidden, K + K*action_dim + K*action_dim)
               → split into mixture_logits, means, raw_log_stds

    Distribution (raw, pre-tanh):
        p(a|s) = Σ_k π_k(s) · N(a | μ_k(s), diag(σ_k(s)²))

    ``explore_intensity`` scales component σ only (not mixture logits).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        K: int = DEFAULT_K,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        entropy_coef: float = 0.0,
        noise_tau_steps: float = 0.0,
        noise_scale: float = 0.0,
        model_path: Optional[str] = None,
    ):
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            device=device, deterministic=deterministic,
            entropy_coef=entropy_coef,
            noise_tau_steps=noise_tau_steps, noise_scale=noise_scale,
        )
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Trunk.
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # Head: logits (K) + means (K*ad) + raw_log_stds (K*ad)
        ad = action_dim
        head_out = K + K * ad + K * ad
        self.head = nn.Linear(hidden_dim, head_out)

        self._init_head()

        if model_path is not None:
            payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            self.load_state_dict(state_dict, strict=True)
            self.to(self.device)

    def _init_head(self):
        """Initialize so K=1 degenerate case matches ① at init."""
        with torch.no_grad():
            ad = self.action_dim
            K = self.K
            target_log_std = -1.0
            tanh_b = 2.0 * (target_log_std - self.log_std_min) / (self.log_std_max - self.log_std_min) - 1.0
            tanh_b = max(-0.999, min(0.999, tanh_b))
            bias_val = float(np.arctanh(tanh_b))

            # Mixture logits: zero (uniform weights at init).
            self.head.weight.data[:K, :] = 0.0
            self.head.bias.data[:K] = 0.0
            # Means: default init (already done).
            # Log-stds: zero weight, bias for target.
            self.head.weight.data[K + K*ad:, :] = 0.0
            self.head.bias.data[K + K*ad:] = bias_val

    # ------------------------------------------------------------------
    # Bounded log-std (same smooth squash as ①)
    # ------------------------------------------------------------------

    def _bounded_log_std(
        self, raw_log_std: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> torch.Tensor:
        """Squash raw log-std into [log_std_min, log_std_max].

        Input shape: (B, K, action_dim).

        ``explore_intensity`` shifts the log-std before squashing:
        0.5 = neutral (learned σ as-is), →0 = compress, →1 = expand.
        May be a scalar float or a ``(B,)`` tensor.
        """
        if isinstance(explore_intensity, torch.Tensor):
            offset = (explore_intensity - 0.5) * 2.0
            # Broadcast (B,) with (B, K, action_dim).
            offset = offset.view(offset.shape[0], 1, 1)
        else:
            offset = float(explore_intensity - 0.5) * 2.0
        t = torch.tanh(raw_log_std + offset)
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (t + 1.0)

    def _forward_head(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, means, bounded_log_stds).

        logits: (B, K)
        means:  (B, K, action_dim)
        log_stds: (B, K, action_dim)
        """
        h = self.trunk(obs)
        out = self.head(h)
        ad = self.action_dim
        K = self.K
        logits, means_flat, raw_log_stds_flat = out.split(
            [K, K * ad, K * ad], dim=-1
        )
        means = means_flat.view(-1, K, ad)
        raw_log_stds = raw_log_stds_flat.view(-1, K, ad)
        log_stds = self._bounded_log_std(raw_log_stds, explore_intensity=explore_intensity)
        return logits, means, log_stds

    # ------------------------------------------------------------------
    # Raw-space hooks
    # ------------------------------------------------------------------

    def _raw_sample(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Sample via Gumbel-max component selection + per-component rsample."""
        logits, means, log_stds = self._forward_head(obs, explore_intensity=explore_intensity)
        B, K, ad = means.shape

        # Gumbel-max: sample one component per batch element.
        # This is the standard correct sampling path for a mixture.
        # The component selection is non-differentiable (correct —
        # mixture weights are learned via the log_prob path, not the
        # sampling path).
        gumbel = -torch.empty_like(logits).exponential_().log()
        idx = (logits + gumbel).argmax(dim=-1)  # (B,)

        # Gather the selected component's parameters.
        idx_exp = idx.view(B, 1, 1).expand(B, 1, ad)
        comp_mean = means.gather(1, idx_exp).squeeze(1)  # (B, ad)
        comp_log_std = log_stds.gather(1, idx_exp).squeeze(1)  # (B, ad)

        # rsample from the selected component.
        std = comp_log_std.exp()
        raw = comp_mean + std * torch.randn_like(comp_mean)

        extras = {"component_idx": idx}
        return raw, extras

    def _raw_log_prob(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
        *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Mixture log_prob via logsumexp.

        log_prob = logsumexp(log_softmax(logits) + component_log_prob, dim=K)

        The tanh Jacobian is added by the base class **after** this
        (it depends only on the action, not on the component).
        """
        logits, means, log_stds = self._forward_head(obs, explore_intensity=explore_intensity)
        B, K, ad = means.shape

        # Per-component log_prob: (B, K)
        # Normal.log_prob returns (B, K, ad), sum over ad.
        stds = log_stds.exp()
        comp_lp = Normal(means, stds).log_prob(
            raw_action.unsqueeze(1).expand(B, K, ad)
        ).sum(dim=-1)  # (B, K)

        # Add log mixture weights.
        log_weights = torch.log_softmax(logits, dim=-1)  # (B, K)
        weighted = comp_lp + log_weights  # (B, K)

        # Mixture log_prob via logsumexp.
        raw_log_prob = torch.logsumexp(weighted, dim=-1)  # (B,)

        extras = None
        return raw_log_prob, extras

    def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean of the most probable component.

        This is not the true mode of the mixture (which may be multimodal),
        but it's a reasonable deterministic action.
        """
        logits, means, _ = self._forward_head(obs, explore_intensity=0.5)
        idx = logits.argmax(dim=-1)  # (B,)
        ad = self.action_dim
        idx_exp = idx.view(-1, 1, 1).expand(-1, 1, ad)
        return means.gather(1, idx_exp).squeeze(1)

    def _regularizer_and_stats(
        self, obs, raw_action, raw_log_prob, want_stats,
        sample_extras, score_extras,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """Sampled entropy estimate + mixture-specific diagnostics."""
        # The regularizer is the sampled entropy estimate, computed
        # in the base class's evaluate_actions via a fresh _raw_sample
        # + _raw_log_prob.  Here we only compute stats.
        regularizer = None  # Base class handles the regularizer.

        stats = None
        if want_stats:
            with torch.no_grad():
                logits, means, log_stds = self._forward_head(obs, explore_intensity=0.5)
                # Mixture weight statistics.
                weights = torch.softmax(logits, dim=-1)  # (B, K)
                log_weights = torch.log_softmax(logits, dim=-1)
                # Categorical entropy: -Σ π_k log π_k
                cat_entropy = -(weights * log_weights).sum(dim=-1)  # (B,)
                # Max weight (collapse indicator).
                max_weight = weights.max(dim=-1).values  # (B,)
                # Component std statistics.
                eff_stds = log_stds.exp()
                # Component usage from sample_extras (if available).
                comp_usage = None
                if sample_extras and "component_idx" in sample_extras:
                    idx = sample_extras["component_idx"]  # (B,)
                    K = self.K
                    usage = torch.zeros(K, device=idx.device)
                    for k in range(K):
                        usage[k] = (idx == k).float().mean()
                    comp_usage = usage

                stats = {
                    "mixture_weight_entropy": float(cat_entropy.mean().item()),
                    "mixture_weight_max_mean_batch": float(max_weight.mean().item()),
                    "comp_std_mean_batch": float(eff_stds.mean().item()),
                }
                if comp_usage is not None:
                    for k in range(self.K):
                        stats[f"comp_{k}_usage"] = float(comp_usage[k].item())

        return regularizer, stats

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_config(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "K": self.K,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }

    @property
    def export_class_path(self) -> str:
        return "baseline.framework.ppo.policies.mog_tanh_mlp:MoGTanhMLPPolicy"
