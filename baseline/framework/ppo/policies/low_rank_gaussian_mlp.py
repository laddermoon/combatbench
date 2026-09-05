"""Low-rank covariance Gaussian policy (Stage 2).

Models raw-space covariance as ``Σ = diag(σ²) + U Uᵀ`` where ``U`` is
a rank-``k`` factor.  This allows the raw (pre-tanh) distribution to
express pairwise correlations between action dimensions — e.g. hip and
knee on the same leg flexing together — which a diagonal Gaussian
cannot.

Uses :class:`torch.distributions.LowRankMultivariateNormal` for stable,
library-implemented ``log_prob`` / ``rsample`` / ``entropy``.

See ``DESIGN_low_rank_gaussian.md`` for the full design rationale.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal

from baseline.framework.ppo.policies.tanh_squashed_base import TanhSquashedPolicyBase

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 0.0
DEFAULT_RANK = 4
_PD_MARGIN = 1e-6  # Positive-definite margin on cov_diag


class LowRankGaussianMLPPolicy(TanhSquashedPolicyBase):
    """Tanh-squashed low-rank Gaussian with state-dependent parameters.

    Architecture:
        trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
        head:  Linear(hidden, action_dim + action_dim + action_dim * rank)
               → split into mean, raw_log_std, U_flat

    Covariance:  Σ = diag(σ² + ε) + U Uᵀ
    where σ is state-dependent (bounded via tanh squash like ①) and
    U is reshaped from U_flat to (B, action_dim, rank).

    ``explore_intensity`` scales **both** σ and U by exp(offset) (where
    offset = (explore_intensity - 0.5) * 2), so the whole covariance
    is scaled by exp(offset)² — preserving the correlation structure
    while scaling the spread.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        rank: int = DEFAULT_RANK,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        entropy_coef: float = 0.0,
        temperature: float = 1.0,
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
        self.rank = int(rank)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Trunk (same as ①).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # Head: mean + raw_log_std + U_flat
        head_out = action_dim + action_dim + action_dim * rank
        self.head = nn.Linear(hidden_dim, head_out)

        self._init_head()

        if model_path is not None:
            payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            self.load_state_dict(state_dict, strict=True)
            self.to(self.device)

    def _init_head(self):
        """Initialize so initial σ ≈ 0.368 and U ≈ 0 (degenerate to ①)."""
        with torch.no_grad():
            target_log_std = -1.0
            tanh_b = 2.0 * (target_log_std - self.log_std_min) / (self.log_std_max - self.log_std_min) - 1.0
            tanh_b = max(-0.999, min(0.999, tanh_b))
            bias_val = float(np.arctanh(tanh_b))

            ad = self.action_dim
            # Mean half: default init (already done).
            # Log-std half: zero weight, bias for target.
            self.head.weight.data[ad:2*ad, :] = 0.0
            self.head.bias.data[ad:2*ad] = bias_val
            # U half: small random initialization (NOT zero!).
            # When U=0, ∂log_prob/∂U = 0 because U enters log_prob
            # through UUᵀ (quadratic), so a zero init creates a saddle
            # point that PPO cannot escape.  Small random values break
            # the symmetry and let gradients flow.
            nn.init.normal_(self.head.weight.data[2*ad:, :], mean=0.0, std=0.01)
            nn.init.zeros_(self.head.bias.data[2*ad:])

    # ------------------------------------------------------------------
    # Bounded log-std (same smooth squash as ①)
    # ------------------------------------------------------------------

    def _bounded_log_std(
        self, raw_log_std: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> torch.Tensor:
        if isinstance(explore_intensity, torch.Tensor):
            offset = (explore_intensity - 0.5) * 2.0
            offset = offset.unsqueeze(-1)  # (B, 1) for broadcasting with (B, action_dim)
        else:
            offset = float(explore_intensity - 0.5) * 2.0
        t = torch.tanh(raw_log_std + offset)
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (t + 1.0)

    def _forward_head(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean, bounded_log_std, U) where U is (B, action_dim, rank)."""
        h = self.trunk(obs)
        out = self.head(h)
        ad = self.action_dim
        mean, raw_log_std, U_flat = out.split([ad, ad, ad * self.rank], dim=-1)
        log_std = self._bounded_log_std(raw_log_std, explore_intensity=explore_intensity)
        U = U_flat.view(-1, ad, self.rank)
        # Scale U by exp(offset) so the whole covariance is scaled by
        # exp(offset)², preserving correlation structure while scaling
        # the spread (matching the σ scaling from the log_std offset).
        if isinstance(explore_intensity, torch.Tensor):
            scale = torch.exp((explore_intensity - 0.5) * 2.0)
            scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        else:
            scale = float(np.exp((explore_intensity - 0.5) * 2.0))
        U = U * scale
        return mean, log_std, U

    def _build_dist(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> LowRankMultivariateNormal:
        mean, log_std, U = self._forward_head(obs, explore_intensity=explore_intensity)
        cov_diag = log_std.exp().pow(2) + _PD_MARGIN
        return LowRankMultivariateNormal(
            loc=mean,
            cov_diag=cov_diag,
            cov_factor=U,
        )

    # ------------------------------------------------------------------
    # Raw-space hooks
    # ------------------------------------------------------------------

    def _raw_sample(
        self, obs: torch.Tensor, *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        dist = self._build_dist(obs, explore_intensity=explore_intensity)
        raw = dist.rsample()
        return raw, None

    def _raw_log_prob(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
        *, explore_intensity: Any = 0.5,
    ) -> Tuple[torch.Tensor, None]:
        dist = self._build_dist(obs, explore_intensity=explore_intensity)
        return dist.log_prob(raw_action), None

    def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self._forward_head(obs, explore_intensity=0.5)
        return mean

    def _regularizer_and_stats(
        self, obs, raw_action, raw_log_prob, want_stats,
        sample_extras, score_extras,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        # Entropy uses the *learned* distribution (neutral explore_intensity).
        dist = self._build_dist(obs, explore_intensity=0.5)
        entropy = dist.entropy()

        regularizer = None
        if self._entropy_coef != 0.0:
            regularizer = -self._entropy_coef * entropy.mean()

        stats = None
        if want_stats:
            with torch.no_grad():
                mean, log_std, U = self._forward_head(obs, explore_intensity=0.5)
                eff_std = log_std.exp()
                # U Frobenius norm: (B,) → mean/max over batch
                U_frob = U.flatten(1).norm(dim=-1)  # (B,)
                # Covariance trace = sum(σ² + ε) + ||U||_F²
                cov_trace = (eff_std.pow(2) + _PD_MARGIN).sum(-1) + U_frob.pow(2)
                # Marginal std = sqrt(σ² + ||U_row||²) per dim
                U_row_norms = U.norm(dim=-1)  # (B, action_dim)
                marginal_std = (eff_std.pow(2) + U_row_norms.pow(2) + _PD_MARGIN).sqrt()
                stats = {
                    "std_mean_batch": float(eff_std.mean().item()),
                    "std_min_batch": float(eff_std.min().item()),
                    "std_max_batch": float(eff_std.max().item()),
                    "marginal_std_mean_batch": float(marginal_std.mean().item()),
                    "U_frob_mean_batch": float(U_frob.mean().item()),
                    "U_frob_max_batch": float(U_frob.max().item()),
                    "cov_trace_mean_batch": float(cov_trace.mean().item()),
                }
        return regularizer, stats

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_config(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }

    @property
    def export_class_path(self) -> str:
        return "baseline.framework.ppo.policies.low_rank_gaussian_mlp:LowRankGaussianMLPPolicy"
