"""SharedMixtureBoundedStdTruncatedNormalPolicy — mixture of truncated
normals with a shared, sigmoid-bounded σ.

Same mixture as ``SharedMixtureTruncatedNormalPolicy``:

    p(a|s) = Σ_k π_k(s) · Π_d TN(a_d; μ_kd(s), σ_kd, [-1,1])

The ONLY difference is the σ parameterization and the explore_factor
mechanism — the bounded axis of the 2×2×2 family:

- σ comes from a trainable ``(K, D)`` raw control ``v`` mapped through
  the bounded sigmoid  σ(v) = exp(r_min + Δr·sigmoid(v)) ∈ (σ_min,
  σ_max), shared across states — the same map and constants
  (σ_min=0.05, σ_max=2.0, init σ≡e⁻¹) as ``BoundedStdTruncatedNormalPolicy``.
- explore_factor shifts the control coordinate BEFORE the sigmoid:
  σ_eff = σ(v + α·e) with α ≈ 1.1993 calibrated so d(log σ)/de at init
  equals the legacy multiplicative 3^e slope.

Everything else (mixture weights, means, sampling, log_prob, Rényi-2
uncertainty on the e=0 σ) is inherited unchanged.  ef widens every
component's coverage but does NOT promise monotonic marginal U on a
mixture — see DESIGN_truncnorm_family.md.

See DESIGN_truncnorm_family.md (cell: MoG=yes, state-σ=no, bounded=yes).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch import nn

from baseline.framework.ppo.policies.bounded_std_truncated_normal_mlp import (
    BoundedStdTruncatedNormalPolicy,
    _bounded_map_geometry,
    _bounded_sigma_value,
    _default_explore_alpha,
    _SAT_HI,
    _SAT_LO,
)
from baseline.framework.ppo.policies.shared_mixture_truncated_normal_mlp import (
    SharedMixtureTruncatedNormalPolicy,
)

__all__ = [
    "SharedMixtureBoundedStdTruncatedNormalPolicy",
]


class SharedMixtureBoundedStdTruncatedNormalPolicy(
    SharedMixtureTruncatedNormalPolicy,
):
    """K-component truncated-normal mixture, shared bounded σ.

    σ control is the parameter ``raw_std`` (K, D) holding the pre-sigmoid
    coordinate v, broadcast to (B, K, D).  ``_policy_sigma`` /
    ``_explored_sigma`` are the bounded map and the v+αe explore
    mechanism — the only differences from the unbounded shared mixture.
    """

    _POLICY_CLASS = "SharedMixtureBoundedStdTruncatedNormalPolicy"
    _EXPORTED_CLASS = "ExportedSharedMixtureBoundedStdTruncNormPolicy"
    _EXPORT_TEMPLATE = "_export_template_shared_mixture_bounded_std_truncnorm.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_components: int = 3,
        component_init_noise: float = 0.02,
        device: torch.device | str = "cpu",
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        init_std: float = math.exp(-1.0),
        explore_alpha: Optional[float] = None,
    ):
        super().__init__(
            obs_dim, action_dim, hidden_dim,
            num_components=num_components,
            component_init_noise=component_init_noise,
            device=device,
        )
        del self.log_std  # bounded cell controls v, not log_std

        for name, val in (
            ("sigma_min", sigma_min),
            ("sigma_max", sigma_max),
            ("init_std", init_std),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
        if not (0.0 < sigma_min < init_std < sigma_max):
            raise ValueError(
                "require 0 < sigma_min < init_std < sigma_max, got "
                f"sigma_min={sigma_min} init_std={init_std} "
                f"sigma_max={sigma_max}"
            )
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.init_std = float(init_std)
        self._r_min, self._r_max, self._delta_r, p0, self._v_init = (
            _bounded_map_geometry(sigma_min, sigma_max, init_std)
        )

        K, D = self.num_components, self.action_dim
        self.raw_std = nn.Parameter(
            torch.full((K, D), self._v_init, dtype=torch.float32)
        )

        if explore_alpha is None:
            explore_alpha = _default_explore_alpha(self._delta_r, p0)
        if not (math.isfinite(explore_alpha) and explore_alpha > 0.0):
            raise ValueError(
                f"explore_alpha must be finite and > 0, got {explore_alpha}"
            )
        self.explore_alpha = float(explore_alpha)

        # raw_std was created after the parent's .to(device); move it.
        self.to(torch.device(device))

    # ------------------------------------------------------------------
    # σ seams — bounded map replaces the unbounded exp/clamp versions
    # ------------------------------------------------------------------

    def _sigma_raw(
        self, head_out: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        """Broadcast the shared (K,D) v parameter to (B,K,D)."""
        K, D = self.num_components, self.action_dim
        return self.raw_std.view(1, K, D).expand(batch_size, -1, -1)

    def _policy_sigma(self, raw: torch.Tensor) -> torch.Tensor:
        """σ = f(v) — the bounded sigmoid map, no ±20 clamp."""
        return _bounded_sigma_value(raw, self._r_min, self._delta_r)

    def _explored_sigma(
        self, raw: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ_eff = f(v + α·e) — additive shift before the sigmoid."""
        BoundedStdTruncatedNormalPolicy._check_ei(explore_factor)
        if isinstance(explore_factor, torch.Tensor):
            v_e = raw + self.explore_alpha * explore_factor.view(-1, 1, 1)
        else:
            v_e = raw + self.explore_alpha * float(explore_factor)
        return _bounded_sigma_value(v_e, self._r_min, self._delta_r)

    # ------------------------------------------------------------------
    # Stats / export metadata
    # ------------------------------------------------------------------

    def _build_stats(
        self,
        log_pi: torch.Tensor,
        mean: torch.Tensor,
        raw: torch.Tensor,
        policy_sigma: torch.Tensor,
        eff_sigma: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, float]:
        """Mixture stats + bounded-σ diagnostics over all (b,k,d).

        ``effective_uncertainty`` is the marginal Rényi-2 U evaluated at
        the explored σ — what the exploration actually covers.
        """
        stats = super()._build_stats(
            log_pi, mean, raw, policy_sigma, eff_sigma, uncertainty,
        )
        with torch.no_grad():
            p = (torch.log(policy_sigma) - self._r_min) / self._delta_r
            p_e = (torch.log(eff_sigma) - self._r_min) / self._delta_r
            u_eff = self._marginal_uncertainty(log_pi, mean, eff_sigma)
            stats["effective_uncertainty"] = float(u_eff.mean().item())
            stats["eff_std_min"] = float(eff_sigma.min().item())
            stats["eff_std_max"] = float(eff_sigma.max().item())
            stats["raw_std_min"] = float(raw.min().item())
            stats["raw_std_max"] = float(raw.max().item())
            stats["std_position_mean"] = float(p.mean().item())
            stats["std_lower_saturation_frac"] = float(
                (p < _SAT_LO).float().mean().item()
            )
            stats["std_upper_saturation_frac"] = float(
                (p > _SAT_HI).float().mean().item()
            )
            stats["log_std_sensitivity"] = float(
                (self._delta_r * p * (1.0 - p)).mean().item()
            )
            stats["exploration_sensitivity"] = float(
                (
                    self.explore_alpha * self._delta_r
                    * p_e * (1.0 - p_e)
                ).mean().item()
            )
        return stats

    def _export_extra(self) -> Dict[str, Any]:
        """Distribution identity + resolved bounded config."""
        return {
            "distribution_kind": "bounded_std_mixture_truncated_normal_v1",
            "std_source": "shared",
            "std_parameterization": "sigmoid_log_std_v1",
            "uncertainty_kind": "marginal_renyi2_width_v1",
            "exploration_kind": "raw_std_additive_shift_v1",
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "init_std": self.init_std,
            "explore_alpha": self.explore_alpha,
        }
