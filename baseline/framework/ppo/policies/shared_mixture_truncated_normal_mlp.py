"""SharedMixtureTruncatedNormalPolicy — mixture of truncated normals
with a shared (state-independent) σ parameter.

Same mixture as ``MixtureTruncatedNormalPolicy``:

    p(a|s) = Σ_k π_k(s) · Π_d TN(a_d; μ_kd(s), σ_kd, [-1,1])

The ONLY difference is the σ source: instead of a per-state head block,
σ comes from a trainable ``(K·D,)`` parameter ``log_std`` broadcast over
the batch — the mixture analogue of ``TruncatedNormalPolicy``'s global
``log_std`` vs ``StateTruncatedNormalPolicy``'s σ head.  Everything else
(mixture weights, means, sampling, log_prob, explore_factor scaling
σ·3^e, Rényi-2 uncertainty) is inherited unchanged, so an A/B against
``MixtureTruncatedNormalPolicy`` isolates exactly one variable:
state-dependence of the component σ.

Init parity: ``log_std = -1`` → σ ≡ e⁻¹ for every component and dim,
matching the σ head's (w=0, b=-1) init in the state-σ sibling.

See DESIGN_truncnorm_family.md (cell: MoG=yes, state-σ=no, bounded=no).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from baseline.framework.ppo.policies.mixture_truncated_normal_mlp import (
    MixtureTruncatedNormalPolicy,
)

__all__ = [
    "SharedMixtureTruncatedNormalPolicy",
]


class SharedMixtureTruncatedNormalPolicy(MixtureTruncatedNormalPolicy):
    """K-component truncated-normal mixture, shared ``(K·D,)`` σ.

    Head layout: ``logits (K) | raw_mean (K·D)`` — no σ block.  The σ
    control quantity is the parameter ``log_std`` (K, D), broadcast to
    (B, K, D) by :meth:`_sigma_raw`.
    """

    _POLICY_CLASS = "SharedMixtureTruncatedNormalPolicy"
    _EXPORTED_CLASS = "ExportedSharedMixtureTruncNormPolicy"
    _EXPORT_TEMPLATE = "_export_template_shared_mixture_truncnorm.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_components: int = 3,
        component_init_noise: float = 0.02,
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            obs_dim, action_dim, hidden_dim,
            num_components=num_components,
            component_init_noise=component_init_noise,
            device=device,
        )
        # σ ≡ e⁻¹ at init — matches the state-σ sibling's (w=0, b=-1)
        # σ-head init and TruncatedNormalPolicy's log_std=-1.
        K, D = self.num_components, self.action_dim
        self.log_std = nn.Parameter(
            torch.full((K, D), -1.0, dtype=torch.float32)
        )
        # Created after the parent's .to(device); move it over.
        self.to(torch.device(device))

    def _head_out_dim(self) -> int:
        """No σ block — the head emits only logits and means."""
        K, D = self.num_components, self.action_dim
        return K + K * D

    def _sigma_raw(
        self, head_out: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        """Broadcast the shared (K,D) log_std parameter to (B,K,D)."""
        K, D = self.num_components, self.action_dim
        return self.log_std.view(1, K, D).expand(batch_size, -1, -1)

    def _build_stats(
        self,
        log_pi: torch.Tensor,
        mean: torch.Tensor,
        policy_sigma: torch.Tensor,
        eff_sigma: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, float]:
        """Same stats minus ``sigma_state_std`` — σ has no state
        dependence in this cell (same convention as the shared-σ
        single-component policies)."""
        stats = super()._build_stats(
            log_pi, mean, policy_sigma, eff_sigma, uncertainty,
        )
        stats.pop("sigma_state_std", None)
        return stats

    def _export_extra(self) -> Dict[str, Any]:
        """Same identity metadata, with std_source = shared."""
        extra = super()._export_extra()
        extra["std_source"] = "shared"
        return extra
