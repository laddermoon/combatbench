"""StateBoundedStdTruncatedNormalPolicy — state-dependent bounded σ.

Identical to :class:`BoundedStdTruncatedNormalPolicy` in every respect
except one: the raw sigmoid input v is a function of the observation,
not a global ``nn.Parameter``.

    μ       = tanh(head_mean(trunk(obs)))                ∈ (-1, 1)
    v(obs)  = head_v(trunk(obs))                         unbounded, (B, D)
    σ(obs)  = exp(r_min + Δr·sigmoid(v))                 ∈ (σ_min, σ_max)
    A       ~ Normal(μ, σ_e²) truncated to [-1, 1]        (per dim)

explore_factor keeps the shared variant's additive raw-space shift,
applied elementwise on the per-state v:

    v_e = v(obs) + alpha · e        e ∈ [-1, 1]

so d(log σ_e)/de = alpha·Δr·p_e(1−p_e) > 0 at every state — the same
monotone coverage guarantee, now with state-dependent base σ.

The σ half of the head is zero-initialised with bias = v_init
(= logit((log init_std − r_min)/Δr)), so at step 0 σ(obs) ≡ init_std
everywhere and the policy is *equivalent* to the shared variant given
the same trunk/mean weights.  Because the 2D-wide head consumes a
different amount of init RNG, degenerate equivalence requires copying
the mean-network weights explicitly — same seed alone does not give
identical weights.

All distribution math (truncation, inverse-CDF sampling, log_prob,
peak-based U, bounded map, explore shift, deterministic act) is
inherited — this class only provides the per-state v source via
``_distribution_params`` and the network architecture.  See
DESIGN_bounded_std_truncated_normal.md §3.2.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from baseline.framework.ppo.policies.bounded_std_truncated_normal_mlp import (
    BoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    _DistParams,
)

__all__ = [
    "StateBoundedStdTruncatedNormalPolicy",
]


class StateBoundedStdTruncatedNormalPolicy(BoundedStdTruncatedNormalPolicy):
    """Bounded-σ truncated normal with state-dependent raw v.

    Head layout: ``[raw_mean | v]`` — (B, 2D) split into (B, D) each.
    """

    _POLICY_CLASS = "StateBoundedStdTruncatedNormalPolicy"
    _EXPORTED_CLASS = "ExportedStateBoundedStdTruncNormPolicy"
    _EXPORT_TEMPLATE = "_export_template_state_bounded_std_truncnorm.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
        init_std: float = 0.36787944117144233,
        explore_alpha: Optional[float] = None,
    ):
        # Parent validates config, resolves _r_min/_r_max/_delta_r,
        # _v_init and explore_alpha, and builds the shared-variant net
        # + raw_std — those two are replaced by the state architecture.
        super().__init__(
            obs_dim, action_dim, hidden_dim, device,
            sigma_min=sigma_min, sigma_max=sigma_max,
            init_std=init_std, explore_alpha=explore_alpha,
        )
        del self.net
        del self.raw_std

        # Trunk shape matches TruncatedNormalPolicy's first two layers so
        # the parameter delta vs the shared variant is attributable only
        # to the wider head (hidden → 2·action_dim instead of →
        # action_dim) — same convention as StateTruncatedNormalPolicy.
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Head outputs raw_mean | v (each action_dim wide).
        self.head = nn.Linear(hidden_dim, 2 * action_dim)
        self._init_head()

        # Device is derived from parameters via the @property below, so
        # .to(device) / .cuda() / DataParallel all keep it in sync (P0-4).
        self.to(torch.device(device))

    def _init_head(self) -> None:
        """Init so σ(obs) = init_std everywhere at step 0.

        v half: weights zeroed, bias = _v_init → v(obs) ≡ v_init,
        matching BoundedStdTruncatedNormalPolicy's ``raw_std`` init
        exactly.  This makes the degenerate-equivalence test (copy
        shared-variant weights, get bit-identical outputs) possible.

        Mean half keeps default PyTorch Linear init — same distribution
        as the shared variant's final layer init.
        """
        d = self.action_dim
        with torch.no_grad():
            self.head.weight[d:, :].zero_()
            self.head.bias[d:].fill_(self._v_init)

    # ------------------------------------------------------------------
    # The ONLY semantic difference: v comes from the head, not a param
    # ------------------------------------------------------------------

    def _raw_v(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One trunk+head pass → (mean, v), both (B, action_dim)."""
        out = self.head(self.trunk(obs))
        raw_mean, v = out.split(self.action_dim, dim=-1)
        return torch.tanh(raw_mean), v

    def _policy_params(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mean, policy_sigma) — kept for hook-contract compatibility.

        The distribution path uses ``_distribution_params`` (explore
        acts on raw v, not on σ); this override exists so external
        callers of the parent hook still get a consistent view.
        """
        mean, v = self._raw_v(obs)
        return mean, self._bounded_sigma(v)

    def _distribution_params(
        self, obs: torch.Tensor, explore_factor: Any = 0.0,
    ) -> _DistParams:
        """(μ, v(obs)) → bounded σ; explore shifts v *before* the sigmoid.

        v is already (B, D): a scalar explore_factor broadcasts over
        batch and dims; a (B,) tensor yields per-frame (B, D) σ — same
        broadcast contract as the shared variant.
        """
        self._check_ei(explore_factor)
        mean, v = self._raw_v(obs)
        if isinstance(explore_factor, torch.Tensor):
            v_e = v + self.explore_alpha * explore_factor.unsqueeze(-1)
        else:
            v_e = v + self.explore_alpha * float(explore_factor)
        return _DistParams(
            mean=mean,
            std_control=v,
            policy_sigma=self._bounded_sigma(v),
            eff_sigma=self._bounded_sigma(v_e),
        )

    # ------------------------------------------------------------------
    # Stats / export metadata
    # ------------------------------------------------------------------

    def _build_stats(
        self,
        uncertainty: torch.Tensor,
        params: _DistParams,
    ) -> Dict[str, float]:
        """Shared-variant stats + ``sigma_state_std``.

        ``sigma_state_std`` = mean over dims of the across-batch std of
        σ — isolates state dependence (0 for the shared variant).
        raw_std_min/max aggregate over batch and dims, so they bound the
        per-state v range.
        """
        stats = super()._build_stats(uncertainty, params)
        with torch.no_grad():
            stats["sigma_state_std"] = float(
                params.policy_sigma.std(dim=0, correction=0).mean().item()
            )
        return stats

    def _export_extra(self) -> Dict[str, Any]:
        """Same identity metadata, with std_source = state."""
        extra = super()._export_extra()
        extra["std_source"] = "state"
        return extra
