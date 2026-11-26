"""Vanilla observation-space world model — the simplest possible WM.

Per the main-experiment spec the WM is a plain network:

    F(obs, action) -> next_obs

WMState IS the observation (57 dims), so every required mapping is
trivial:  EnvState->WMState is the env's observer, WMState->Obs is the
identity, and the loss phi is an analytic function of obs dims
(obs[42:45]=projected gravity, obs[45]=height).

Parameterization: the MLP predicts a per-dim normalized delta,
``o' = o + dyn([norm(o), a]) * o_std`` — residual form so the network
doesn't waste capacity re-learning the near-identity 20 ms map.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from baseline.framework.dynabptt.smbptt.task import STANDING_HEIGHT


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ObsWorldModel(nn.Module):
    def __init__(self, obs_dim: int = 57, action_dim: int = 21,
                 hidden: Tuple[int, ...] = (1024, 1024, 1024)) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.dyn = _mlp(obs_dim + action_dim, hidden, obs_dim)
        self.register_buffer("o_mean", torch.zeros(obs_dim))
        self.register_buffer("o_std", torch.ones(obs_dim))

    def norm_o(self, o: torch.Tensor) -> torch.Tensor:
        return (o - self.o_mean) / self.o_std

    def fit_normalization(self, obs: torch.Tensor) -> None:
        """Estimate obs input/delta stats from a (N, obs_dim) sample."""
        self.o_mean.copy_(obs.mean(0).to(self.o_mean))
        self.o_std.copy_(obs.std(0).clamp_min(1e-6).to(self.o_std))

    def next_obs(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        delta = self.dyn(torch.cat([self.norm_o(o), a], dim=-1))
        return o + delta * self.o_std


def phi_from_obs(o: torch.Tensor) -> torch.Tensor:
    """Analytic phi on the observation: uprightness * z / standing_height.

    obs[44] = projected_gravity z-component (g_b.z), obs[45] = height.
    Same quantity as analytic_phi(rep), read straight off the obs.
    """
    uprightness = -o[..., 44]
    return uprightness * o[..., 45] / STANDING_HEIGHT


def onestep_nrmse(model: ObsWorldModel,
                  obs0: torch.Tensor, acts: torch.Tensor,
                  obs_t: torch.Tensor) -> float:
    """Criterion: teacher-forced 1-step obs regression error.

    Global NRMSE = sqrt( mean(err^2) / mean(var_d) ) — the fraction of
    the data's total per-dim variance left unexplained.  Robust to
    near-constant dims (their error and variance both vanish).
    """
    with torch.no_grad():
        pred = model.next_obs(obs0, acts)
        mse = (pred - obs_t).pow(2).mean()
        var = obs_t.var(0, unbiased=False).mean()
        return float((mse / var.clamp_min(1e-12)).sqrt())
