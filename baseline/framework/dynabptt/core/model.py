"""Representation-space world model — the rep route.

After the obs-space baseline established that obs is a poor WMState
(sqrt-compressed velocity channels + unobservable contact state), the
spec moves the WM to the GenCoord representation:

    WMState = (aux 7, rep 52, feet 2)

  EnvState -> WMState : cs -> (aux, rep) analytic; feet = obs[52:54]
                        — contact forces live in the rep as ORDINARY
                        state dims, regressed by the transition model
                        like everything else (no separate head; fit
                        quality whatever it is)
  F                   : dyn MLP predicts accel deltas + feet delta;
                        pose/joints integrate analytically
  WMState -> Obs      : compose_obs(rep, aux, feet) — all from WMState
  WMState -> Loss     : analytic_phi(rep)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.framework.dynabptt.hybrid_ppo import gencoord as gc
from baseline.framework.dynabptt.smbptt.task import (
    STANDING_HEIGHT, analytic_phi, compose_obs)

FEET_DIM = 2


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class RepWorldModel(nn.Module):
    """F(rep, aux, feet, a) -> (aux', rep', feet').

    Learned part = one MLP predicting [Δv_b, Δw_b, Δvj, Δfeet] (29 dims);
    root pose + joints integrate exactly via gc.apply_prediction.
    """

    def __init__(self, action_dim: int = 21,
                 hidden: Tuple[int, ...] = (1024, 1024, 1024),
                 dt: float = gc.DT,
                 substep_coeff: float = gc.SUBSTEP_COEFF) -> None:
        super().__init__()
        self.dt = float(dt)
        self.substep_coeff = float(substep_coeff)
        self.rep_dim = gc.REP_DIM      # 52
        self.aux_dim = gc.AUX_DIM      # 7
        in_dim = gc.REP_DIM + FEET_DIM
        self.dyn = _mlp(in_dim + action_dim, hidden,
                        gc.PRED_DIM + FEET_DIM)   # 27 + 2 = 29
        self.register_buffer("s_mean", torch.zeros(in_dim))
        self.register_buffer("s_std", torch.ones(in_dim))

    def norm_s(self, repf: torch.Tensor) -> torch.Tensor:
        return (repf - self.s_mean) / self.s_std

    def fit_normalization(self, repf: torch.Tensor) -> None:
        """Estimate input stats on (N, rep+feet=54) samples."""
        self.s_mean.copy_(repf.mean(0).to(self.s_mean))
        self.s_std.copy_(repf.std(0).clamp_min(1e-6).to(self.s_std))

    def next_state(self, rep: torch.Tensor, aux: torch.Tensor,
                   feet: torch.Tensor, a: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repf = torch.cat([rep, feet], dim=-1)
        pred = self.dyn(torch.cat([self.norm_s(repf), a], dim=-1))
        aux_n, rep_n = gc.apply_prediction(
            aux, rep, pred[..., :gc.PRED_DIM],
            dt=self.dt, c=self.substep_coeff)
        feet_n = feet + pred[..., gc.PRED_DIM:]
        return aux_n, rep_n, feet_n

    def obs_of(self, rep: torch.Tensor, aux: torch.Tensor,
               feet: torch.Tensor) -> torch.Tensor:
        return compose_obs(rep, aux, feet)


def phi_of_rep(rep: torch.Tensor) -> torch.Tensor:
    return analytic_phi(rep)


def state_loss(rep: torch.Tensor, aux: torch.Tensor,
               rep_t: torch.Tensor, aux_t: torch.Tensor,
               feet_p: torch.Tensor, feet_t: torch.Tensor,
               feet_std: torch.Tensor) -> torch.Tensor:
    """Bryson-weighted regression on (rep, aux) + normalized feet MSE."""
    pb = gc.loss_blocks(rep, aux)
    tb = gc.loss_blocks(rep_t, aux_t)
    num = den = 0.0
    for name, blk in pb.items():
        w = 1.0 / gc.NOMINAL_ERRORS[name] ** 2
        num = num + w * blk.shape[-1] * F.mse_loss(blk, tb[name])
        den = den + blk.shape[-1]
    L = num / den
    feet_err = (feet_p - feet_t) / feet_std.clamp_min(1e-3)
    return L + 0.5 * feet_err.pow(2).mean()


def bryson_error(rep_p, aux_p, rep_t, aux_t) -> float:
    """1-step fit criterion: sqrt of the nominal-error-normalized
    mean squared error over the integrated blocks (root pose, joint
    pos/vel).  1.0 == prediction error at the physical nominal scale.
    Pure rep-vs-env comparison; feet excluded (acknowledged
    unpredictable), phi nowhere."""
    pb = gc.loss_blocks(rep_p, aux_p)
    tb = gc.loss_blocks(rep_t, aux_t)
    num = den = 0.0
    for name, blk in pb.items():
        w = 1.0 / gc.NOMINAL_ERRORS[name] ** 2
        num = num + w * blk.shape[-1] * F.mse_loss(blk, tb[name])
        den = den + blk.shape[-1]
    return float((num / den).sqrt())


def wmstate_nrmse(rep_p, aux_p, feet_p, rep_t, aux_t, feet_t) -> float:
    """Global NRMSE over the whole WMState vector:
    sqrt( sum err^2 / sum centered-var )."""
    err = torch.cat([rep_p - rep_t, aux_p - aux_t,
                     feet_p - feet_t], dim=-1)
    true = torch.cat([rep_t, aux_t, feet_t], dim=-1)
    mse = err.pow(2).mean()
    var = true.var(0, unbiased=False).mean()
    return float((mse / var.clamp_min(1e-12)).sqrt())


def wmstate_breakdown(rep_p, aux_p, feet_p, rep_t, aux_t, feet_t):
    """Top-8 WMState dims by error share, labeled."""
    labels = ([f"rep{i}" for i in range(rep_t.shape[-1])]
              + [f"aux{i}" for i in range(aux_t.shape[-1])]
              + [f"feet{i}" for i in range(feet_t.shape[-1])])
    err = torch.cat([rep_p - rep_t, aux_p - aux_t,
                     feet_p - feet_t], dim=-1)
    true = torch.cat([rep_t, aux_t, feet_t], dim=-1)
    mse_per = err.pow(2).mean(0)
    var_per = true.var(0, unbiased=False)
    share = mse_per / mse_per.sum().clamp_min(1e-12)
    top = share.argsort(descending=True)[:8]
    return [(labels[int(d)], float(share[d]),
             float(mse_per[d].sqrt()), float(var_per[d].sqrt()))
            for d in top]
