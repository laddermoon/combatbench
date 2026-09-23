"""StatePreTanhNormalPolicy — state-dependent σ variant of
PreTanhNormalPolicy.

Identical distribution, exploration mapping, uncertainty, scoring path
and guards — the ONLY difference is where σ comes from:

    shared  : log_std is an (D,) nn.Parameter (same for every state)
    state   : log_std is a head output r(s) = f_trunk(obs)  (per state)

All error-prone math (float64 helpers, coverage-radial exploration,
atanh inversion, tail-risk / action-margin guards) is inherited from
PreTanhNormalPolicy unchanged.  The σ head is initialized with zeroed
weights and bias -1 so that σ(s) ≡ e⁻¹ at init — exactly matching the
shared version's initialization (degenerate-equivalence test).

See DESIGN_pre_tanh_normal.md — §12.1 documents this variant.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from baseline.framework.ppo.policies.pre_tanh_normal_mlp import (
    PreTanhNormalPolicy,
)

__all__ = [
    "StatePreTanhNormalPolicy",
]


class StatePreTanhNormalPolicy(PreTanhNormalPolicy):
    """Pre-tanh diagonal Gaussian with state-dependent per-dim σ.

    Head layout: ``[mean | log_std]`` — (B, 2D) split into (B, D) each.
    """

    _POLICY_CLASS = "StatePreTanhNormalPolicy"
    _DISTRIBUTION_KIND = "tanh_diagonal_normal_state_std_v1"
    _EXPORTED_CLASS = "ExportedStatePreTanhNormalPolicy"
    _EXPORT_TEMPLATE = "_export_template_state_pre_tanh_normal.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ):
        nn.Module.__init__(self)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 2 * action_dim)

        # σ head init: zero weights + bias -1 → σ(s) ≡ e⁻¹ at init,
        # identical to the shared version's log_std init.
        D = self.action_dim
        with torch.no_grad():
            self.head.weight[D:].zero_()
            self.head.bias[D:].fill_(-1.0)

        self.to(torch.device(device))

    def _policy_params(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mu, r) both float64, both (B, D) — r is state-dependent."""
        out = self.head(self.trunk(obs)).double()
        mu, r = out.chunk(2, dim=-1)
        return mu, r

    def _build_stats(
        self,
        mu: torch.Tensor,
        r: torch.Tensor,
        mu_e: torch.Tensor,
        r_e: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, float]:
        """Parent stats (computed on the (B,D) σ) + sigma_state_std."""
        stats = super()._build_stats(mu, r, mu_e, r_e, uncertainty)
        with torch.no_grad():
            sigma = r.exp()
            stats["sigma_state_std"] = float(
                sigma.std(dim=0).mean().item()
            )
            sigma_e = r_e.exp()
            stats["effective_sigma_state_std"] = float(
                sigma_e.std(dim=0).mean().item()
            )
        return stats
