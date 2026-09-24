"""BoundedStdTruncatedNormalPolicy — truncated normal with bounded σ.

Same action-space truncated normal as :class:`TruncatedNormalPolicy`,
but the per-dim std is produced by a bounded sigmoid map over log σ:

    μ   = tanh(net(obs))                                 ∈ (-1, 1)
    v   = raw_std                    (D,) trainable, unbounded
    r   = r_min + (r_max − r_min) · sigmoid(v)           bounded log σ
    σ   = exp(r)                                       ∈ (σ_min, σ_max)
    A   ~ Normal(μ, σ_e²) truncated to [-1, 1]           (per dim)

explore_factor acts *before* the sigmoid, additively:

    v_e = v + alpha · e        e ∈ [-1, 1]

NOT ``v · 3^e`` — v is a signed unbounded coordinate, so multiplying it
pushes negative v toward the *lower* σ bound for e > 0 (and leaves
v = 0 unchanged).  The additive shift moves every dim in the same
σ direction; near saturation the response simply decays, which is the
intended price of a soft bound.

``alpha`` is derived at construction so the local relative σ response
at the init point matches the legacy ``3^e`` multiplier
(alpha = ln3 / (Δr·p₀(1−p₀))); it is then frozen — it is NOT a
σ multiplier and does not mean σ triples at e=1.

All distribution math (truncation, inverse-CDF sampling, log_prob,
peak-based U, deterministic act) is inherited unchanged — this class
only provides the raw v source via ``_distribution_params`` and the
bounded map.  See DESIGN_bounded_std_truncated_normal.md.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    _ACTION_WIDTH,
    _DistParams,
    _SQRT_2PI,
    TruncatedNormalPolicy,
)

__all__ = [
    "BoundedStdTruncatedNormalPolicy",
]

_EI_TOLERANCE = 1e-6            # float32 slack on the [-1, 1] ei range

# Sigmoid-saturation diagnostic thresholds (stats only, not guards).
_SAT_LO = 1e-3
_SAT_HI = 1.0 - 1e-3


class BoundedStdTruncatedNormalPolicy(TruncatedNormalPolicy):
    """Truncated normal on [-1, 1] with sigmoid-bounded shared σ."""

    _POLICY_CLASS = "BoundedStdTruncatedNormalPolicy"
    _EXPORTED_CLASS = "ExportedBoundedStdTruncNormPolicy"
    _EXPORT_TEMPLATE = "_export_template_bounded_std_truncnorm.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        init_std: float = math.exp(-1.0),
        explore_alpha: Optional[float] = None,
    ):
        # Parent __init__ builds the identical mean net (same RNG
        # consumption order → same weights as TruncatedNormalPolicy for
        # a given seed); the unused log_std parameter is then replaced
        # by raw_std.
        super().__init__(obs_dim, action_dim, hidden_dim, device)
        del self.log_std

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
        self._r_min = math.log(self.sigma_min)
        self._r_max = math.log(self.sigma_max)
        self._delta_r = self._r_max - self._r_min

        # Invert the bounded map so σ(v_init, e=0) == init_std.
        p0 = (math.log(self.init_std) - self._r_min) / self._delta_r
        v_init = math.log(p0) - math.log1p(-p0)  # logit(p0)
        self.raw_std = nn.Parameter(
            torch.full((self.action_dim,), v_init, dtype=torch.float32)
        )

        if explore_alpha is None:
            # Local calibration: d(log σ)/de at (v_init, e=0) equals
            # the legacy 3^e multiplier's slope (ln 3).  This is a
            # first-order match at the init point only — σ does not
            # triple at e=1.
            explore_alpha = math.log(3.0) / (
                self._delta_r * p0 * (1.0 - p0)
            )
        if not (math.isfinite(explore_alpha) and explore_alpha > 0.0):
            raise ValueError(
                f"explore_alpha must be finite and > 0, got {explore_alpha}"
            )
        self.explore_alpha = float(explore_alpha)

        # raw_std was created after the parent's .to(device); move it.
        self.to(torch.device(device))

    # ------------------------------------------------------------------
    # Bounded σ map
    # ------------------------------------------------------------------

    def _bounded_sigma(self, v_e: torch.Tensor) -> torch.Tensor:
        """σ = exp(r_min + Δr · sigmoid(v_e))  ∈ (σ_min, σ_max)."""
        return torch.exp(
            self._r_min + self._delta_r * torch.sigmoid(v_e)
        )

    def policy_sigma(self) -> torch.Tensor:
        """σ without explore shift — for uncertainty U."""
        return self._bounded_sigma(self.raw_std)

    def _policy_params(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mean, policy_sigma) — kept for hook-contract compatibility.

        The distribution path uses ``_distribution_params`` (explore
        acts on raw v, not on σ); this override exists so external
        callers of the parent hook still get a consistent view.
        """
        mean = torch.tanh(self.net(obs))
        return mean, self.policy_sigma()

    @staticmethod
    def _check_ei(explore_factor: Any) -> None:
        """explore_factor must stay within [-1, 1] — no silent clamp."""
        if isinstance(explore_factor, torch.Tensor):
            if not bool(torch.isfinite(explore_factor).all()):
                raise ValueError("explore_factor contains non-finite values")
            bad = (explore_factor < -1.0 - _EI_TOLERANCE) | (
                explore_factor > 1.0 + _EI_TOLERANCE
            )
            if bool(bad.any()):
                raise ValueError(
                    f"explore_factor out of [-1, 1]: "
                    f"min={float(explore_factor.min())}, "
                    f"max={float(explore_factor.max())}"
                )
        else:
            e = float(explore_factor)
            if not math.isfinite(e) or abs(e) > 1.0 + _EI_TOLERANCE:
                raise ValueError(
                    f"explore_factor out of [-1, 1]: {explore_factor}"
                )

    def _distribution_params(
        self, obs: torch.Tensor, explore_factor: Any = 0.0,
    ) -> _DistParams:
        """(μ, v) → bounded σ; explore shifts v *before* the sigmoid.

        Scalar explore_factor broadcasts over dims ((D,) σ); a (B,)
        tensor yields per-frame (B, D) σ — same broadcast contract as
        the parent's ``effective_sigma``.
        """
        self._check_ei(explore_factor)
        mean = torch.tanh(self.net(obs))
        v = self.raw_std  # (D,)
        if isinstance(explore_factor, torch.Tensor):
            v_e = v.unsqueeze(0) + (
                self.explore_alpha * explore_factor.unsqueeze(-1)
            )
        else:
            v_e = v + self.explore_alpha * float(explore_factor)
        return _DistParams(
            mean=mean,
            std_control=v,
            policy_sigma=self._bounded_sigma(v),
            eff_sigma=self._bounded_sigma(v_e),
        )

    # ------------------------------------------------------------------
    # Policy ABC reset hook (rollout reproducibility)
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed torch RNG so per-episode derived seeds make rollouts
        reproducible when the same policy instance is reused."""
        if seed is not None:
            torch.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Stats / export metadata
    # ------------------------------------------------------------------

    def _build_stats(
        self,
        uncertainty: torch.Tensor,
        params: _DistParams,
    ) -> Dict[str, float]:
        """Parent stats + bounded-σ diagnostics.

        The sigmoid position is recovered analytically from σ
        (p = (log σ − r_min)/Δr) — no logit inversion needed, and
        saturated float32 σ maps to p = 0/1 exactly.
        """
        stats = super()._build_stats(uncertainty, params)
        with torch.no_grad():
            v = params.std_control
            p = (torch.log(params.policy_sigma) - self._r_min) / self._delta_r
            p_e = (
                torch.log(params.eff_sigma) - self._r_min
            ) / self._delta_r
            _, _, log_Z_eff = self._trunc_params(
                params.mean, params.eff_sigma,
            )
            u_eff = (
                params.eff_sigma * _SQRT_2PI * torch.exp(log_Z_eff)
                / _ACTION_WIDTH
            )
            stats["effective_uncertainty"] = float(u_eff.mean().item())
            stats["eff_std_min"] = float(params.eff_sigma.min().item())
            stats["eff_std_max"] = float(params.eff_sigma.max().item())
            stats["raw_std_min"] = float(v.min().item())
            stats["raw_std_max"] = float(v.max().item())
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
        """Distribution identity + resolved config for the export."""
        return {
            "distribution_kind": "bounded_std_diagonal_truncated_normal_v1",
            "std_source": "shared",
            "std_parameterization": "sigmoid_log_std_v1",
            "uncertainty_kind": "marginal_peak_width_v1",
            "exploration_kind": "raw_std_additive_shift_v1",
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "init_std": self.init_std,
            "explore_alpha": self.explore_alpha,
        }
