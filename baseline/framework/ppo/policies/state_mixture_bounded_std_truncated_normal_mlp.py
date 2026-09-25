"""StateMixtureBoundedStdTruncatedNormalPolicy — mixture of truncated
normals with state-dependent, sigmoid-bounded σ.

Same mixture as ``SharedMixtureBoundedStdTruncatedNormalPolicy``:

    p(a|s) = Σ_k π_k(s) · Π_d TN(a_d; μ_kd(s), σ_kd(s), [-1,1])

The ONLY difference is the σ source — the state axis of the 2×2×2
family: the shared ``(K, D)`` v parameter becomes a per-state head block
``v(s)`` (w=0, b=v_init at init → σ ≡ e⁻¹), fed through the same bounded
sigmoid map σ(v) = exp(r_min + Δr·sigmoid(v)) and the same v+αe
explore mechanism.

Everything else (mixture weights, means, sampling, log_prob, Rényi-2
uncertainty on the e=0 σ, bounded diagnostics) is inherited unchanged.

See DESIGN_truncnorm_family.md (cell: MoG=yes, state-σ=yes, bounded=yes).
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from baseline.framework.ppo.policies.shared_mixture_bounded_std_truncated_normal_mlp import (
    SharedMixtureBoundedStdTruncatedNormalPolicy,
)

__all__ = [
    "StateMixtureBoundedStdTruncatedNormalPolicy",
]


class StateMixtureBoundedStdTruncatedNormalPolicy(
    SharedMixtureBoundedStdTruncatedNormalPolicy,
):
    """K-component bounded-σ mixture, state-dependent v(s).

    Head layout (single Linear, component-major):
        logits        (K,)
        raw_mean      (K·D,) — view (K, D), mean = tanh(raw_mean)
        raw_v         (K·D,) — view (K, D), σ = f(v), f = bounded sigmoid

    Init: v block w=0, b=v_init → σ ≡ e⁻¹ per component/dim, matching
    the shared sibling's raw_std init and the whole family's σ≡e⁻¹
    convention.
    """

    _POLICY_CLASS = "StateMixtureBoundedStdTruncatedNormalPolicy"
    _EXPORTED_CLASS = "ExportedStateMixtureBoundedStdTruncNormPolicy"
    _EXPORT_TEMPLATE = "_export_template_state_mixture_bounded_std_truncnorm.py"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.raw_std  # v is produced by the head, not a parameter
        # _init_sigma_block ran during super().__init__ before _v_init
        # existed (placeholder -1.0); fill the v bias with v_init now.
        K, D = self.num_components, self.action_dim
        with torch.no_grad():
            self.head.bias[K + K * D:].fill_(self._v_init)

    def _head_out_dim(self) -> int:
        """State-σ cell — the head carries a v block after the means."""
        K, D = self.num_components, self.action_dim
        return K + 2 * K * D

    def _sigma_raw(
        self, head_out: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        """v(s) — the head's last block, (B,K,D)."""
        K, D = self.num_components, self.action_dim
        return head_out[..., K + K * D:].reshape(-1, K, D)

    def _build_stats(
        self,
        log_pi: torch.Tensor,
        mean: torch.Tensor,
        raw: torch.Tensor,
        policy_sigma: torch.Tensor,
        eff_sigma: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, float]:
        """Bounded mixture stats + ``sigma_state_std`` (v is state-
        dependent here — restore the key the shared parent drops)."""
        stats = super()._build_stats(
            log_pi, mean, raw, policy_sigma, eff_sigma, uncertainty,
        )
        with torch.no_grad():
            stats["sigma_state_std"] = float(
                policy_sigma.std(dim=0, correction=0).mean().item()
            )
        return stats

    def _export_extra(self) -> Dict[str, Any]:
        """Same identity metadata, with std_source = state."""
        extra = super()._export_extra()
        extra["std_source"] = "state"
        return extra
