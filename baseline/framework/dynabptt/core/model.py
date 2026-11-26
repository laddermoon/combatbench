"""Lean world model for the DynaBPTT main experiment.

Per MAIN_EXP_DECISIONS.md the WM's only job is state-transition
regression:  F(WMState, a) = Next WMState.

WMState = (aux 7, rep 52) — an analytic re-parameterization of the env
core state cs(55) via gencoord.  The learned part is a single MLP that
predicts acceleration-level deltas (Δv_b, Δw_b, Δvj, 27 dims); pose/vel
integration is exact differentiable math in gc.apply_prediction.

All three required mappings are analytic:
  EnvState -> WMState   gc.represent / gc.extract_aux
  WMState -> Observation  compose_obs (feet channels are constant zero)
  WMState -> Loss        analytic_phi(rep)

There are no learned heads at all — if the model cannot fit the
deterministic data, the MLP itself is the only suspect.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.framework.dynabptt.hybrid_ppo import gencoord as gc


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class LeanWorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 21,
        hidden: Tuple[int, ...] = (1024, 1024, 1024),
        dt: float = gc.DT,
        substep_coeff: float = gc.SUBSTEP_COEFF,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.substep_coeff = float(substep_coeff)
        self.rep_dim = gc.REP_DIM
        self.aux_dim = gc.AUX_DIM
        self.dyn = _mlp(gc.REP_DIM + action_dim, hidden, gc.PRED_DIM)

        self.register_buffer("s_mean", torch.zeros(gc.REP_DIM))
        self.register_buffer("s_std", torch.ones(gc.REP_DIM))

    def norm_s(self, rep: torch.Tensor) -> torch.Tensor:
        return (rep - self.s_mean) / self.s_std

    def fit_normalization(self, reps: torch.Tensor) -> None:
        """Estimate rep input stats from a (N, rep_dim) sample."""
        eps = 1e-6
        self.s_mean.copy_(reps.mean(0).to(self.s_mean))
        self.s_std.copy_(reps.std(0).clamp_min(eps).to(self.s_std))

    def next_state(self, rep: torch.Tensor, aux: torch.Tensor,
                   a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(rep, aux, a) -> (aux', rep').  aux first — matches gc order."""
        pred = self.dyn(torch.cat([self.norm_s(rep), a], dim=-1))
        aux_next, rep_next = gc.apply_prediction(
            aux, rep, pred, dt=self.dt, c=self.substep_coeff)
        return aux_next, rep_next

    @staticmethod
    def rep_aux_from_cs(cs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return gc.represent(cs), gc.extract_aux(cs)

    @staticmethod
    def cs_from_rep_aux(rep: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return gc.reconstruct_cs(aux, rep)


def state_loss(rep: torch.Tensor, aux: torch.Tensor,
               rep_t: torch.Tensor, aux_t: torch.Tensor) -> torch.Tensor:
    """Bryson-weighted regression loss on predicted WMState.

    Groups mirror the official GenCoord loss_blocks: integrated root
    pose + joint pos/vel; weights = 1/nominal_error^2.
    """
    pb = gc.loss_blocks(rep, aux)
    tb = gc.loss_blocks(rep_t, aux_t)
    num = den = 0.0
    for name, blk in pb.items():
        w = 1.0 / gc.NOMINAL_ERRORS[name] ** 2
        num = num + w * blk.shape[-1] * F.mse_loss(blk, tb[name])
        den = den + blk.shape[-1]
    return num / den


def onestep_nrmse(model: LeanWorldModel,
                  rep0: torch.Tensor, aux0: torch.Tensor,
                  acts: torch.Tensor,
                  rep_t: torch.Tensor, aux_t: torch.Tensor,
                  dim_std: torch.Tensor) -> float:
    """Criterion metric: teacher-forced 1-step next-WMState error.

    rep0/aux0 (B, ...) states before each transition, acts (B, A) the
    executed actions, rep_t/aux_t the true successors.  dim_std (rep+aux
    concat dims) is the per-dim std estimated on the fit pool.

    Returns mean over the 59 WMState dims of rmse_d / std_d.
    """
    with torch.no_grad():
        aux_p, rep_p = model.next_state(rep0, aux0, acts)
        err = torch.cat([rep_p - rep_t, aux_p - aux_t], dim=-1)
        rmse = err.pow(2).mean(0).sqrt()
        nrmse = (rmse / dim_std.clamp_min(1e-6)).mean()
    return float(nrmse)
