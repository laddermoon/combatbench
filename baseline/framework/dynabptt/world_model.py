"""WorldModel — deterministic differentiable model of the single-robot dynamics.

Trunk:  f(s_t, a_t) -> delta s  (residual; next state = s + delta, quat renormed)
Heads on the predicted next state ŝ_{t+1}:
  obs_head:  ŝ -> ô (96-dim, feeds the policy inside imagination)
  aux_heads: ŝ -> named scalar observers (e.g. phi, r_cross)
  done_head: ŝ -> termination logit (soft survival weight)

All heads are learned. Kept deliberately small and configurable — v1 is a
fast-validation build, not an architecture study.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

QUAT_SLICE = slice(3, 7)  # root_rot inside the 55-dim state


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 55,
        action_dim: int = 21,
        obs_dim: int = 96,
        aux_names: Iterable[str] = ("phi", "r_cross"),
        hidden: Tuple[int, ...] = (1024, 1024, 1024),
        head_hidden: Tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.aux_names = tuple(aux_names)

        self.dyn = _mlp(state_dim + action_dim, hidden, state_dim)
        self.obs_head = _mlp(state_dim, head_hidden, obs_dim)
        self.aux_heads = nn.ModuleDict(
            {name: _mlp(state_dim, (256, 256), 1) for name in self.aux_names}
        )
        self.done_head = _mlp(state_dim, (256, 256), 1)

    def next_state(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Residual dynamics + quaternion renormalisation."""
        delta = self.dyn(torch.cat([s, a], dim=-1))
        s_next = s + delta
        q = s_next[..., QUAT_SLICE]
        s_next = torch.cat(
            [s_next[..., :3], F.normalize(q, p=2, dim=-1), s_next[..., 7:]], dim=-1
        )
        return s_next

    def heads(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        """All per-state predictions on ŝ (obs / aux scalars / done logit)."""
        out: Dict[str, torch.Tensor] = {"obs": self.obs_head(s)}
        for name, head in self.aux_heads.items():
            out[name] = head(s).squeeze(-1)
        out["done_logit"] = self.done_head(s).squeeze(-1)
        return out

    def step(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s_next = self.next_state(s, a)
        return s_next, self.heads(s_next)
