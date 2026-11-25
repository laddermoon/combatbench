"""SelfWorldModel — local dynamics model for the self-only standup task.

SuperMimic-style structure (see SUPERMIMIC_BOUNDARY.md §1.1):

    dyn:  (rep_t, a_t) -> pred_t          accel-level deltas ONLY (27)
    rep', aux' = apply_prediction(aux, rep, pred)   analytic integration
    heads: (rep, aux) -> obs / phi / stage

The network never predicts position-level quantities — poses, joint
angles and the gravity direction all advance by exact kinematic
integration (gencoord.py), so learned-model error stays confined to the
bounded velocity channels.  Fit loss scores the INTEGRATED state blocks
with Bryson's-rule group weights, mirroring the official learner.

Differentiability is NOT required by the model-assisted-PPO path.
Per-dim standardization (rep input / obs target) uses train-pool
statistics refit each update and stored on the module.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gencoord as gc

N_STAGES = 4


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SelfWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = gc.CS_DIM,
        action_dim: int = 21,
        obs_dim: int = 57,
        hidden: Tuple[int, ...] = (512, 512, 512),
        head_hidden: Tuple[int, ...] = (256, 256),
        dt: float = gc.DT,
        substep_coeff: float = gc.SUBSTEP_COEFF,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.substep_coeff = float(substep_coeff)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.rep_dim = gc.REP_DIM
        self.aux_dim = gc.AUX_DIM
        self.pred_dim = gc.PRED_DIM
        head_in = gc.REP_DIM + gc.AUX_DIM

        self.dyn = _mlp(gc.REP_DIM + action_dim, hidden, gc.PRED_DIM)
        self.obs_head = _mlp(head_in, head_hidden, obs_dim)
        self.phi_head = _mlp(head_in, head_hidden, 1)
        self.stage_head = _mlp(head_in, head_hidden, N_STAGES)

        # Per-dim standardization, fit on the current train pool.
        # Buffers (not parameters): saved in checkpoints, no gradients.
        self.register_buffer("s_mean", torch.zeros(gc.REP_DIM))
        self.register_buffer("s_std", torch.ones(gc.REP_DIM))
        self.register_buffer("o_mean", torch.zeros(obs_dim))
        self.register_buffer("o_std", torch.ones(obs_dim))

        w = {k: 1.0 / v ** 2 for k, v in gc.NOMINAL_ERRORS.items()}
        self._loss_weights = w

    # --- normalization helpers -------------------------------------------
    def norm_s(self, rep: torch.Tensor) -> torch.Tensor:
        return (rep - self.s_mean) / self.s_std

    def norm_o(self, o: torch.Tensor) -> torch.Tensor:
        return (o - self.o_mean) / self.o_std

    def fit_normalization(self, windows: Dict[str, np.ndarray]) -> None:
        """Estimate rep/obs stats from a sampled window batch."""
        s0 = torch.as_tensor(windows["s0"], dtype=torch.float32)
        cs = torch.as_tensor(windows["cs"], dtype=torch.float32)
        obs = torch.as_tensor(windows["obs"], dtype=torch.float32)
        cs_all = torch.cat([s0.unsqueeze(1), cs], dim=1)
        rep_all = gc.represent(cs_all.reshape(-1, self.state_dim))
        o_all = obs.reshape(-1, self.obs_dim)
        eps = 1e-6
        self.s_mean.copy_(rep_all.mean(0).to(self.s_mean))
        self.s_std.copy_(rep_all.std(0).clamp_min(eps).to(self.s_std))
        self.o_mean.copy_(o_all.mean(0).to(self.o_mean))
        self.o_std.copy_(o_all.std(0).clamp_min(eps).to(self.o_std))

    # --- forward ----------------------------------------------------------
    def next_state(self, rep: torch.Tensor, aux: torch.Tensor,
                   a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(rep, aux, a) -> (aux', rep').  Order matches the official
        Dynamics: returns (aux_next, rep_next)."""
        pred = self.dyn(torch.cat([self.norm_s(rep), a], dim=-1))
        aux_next, rep_next = gc.apply_prediction(
            aux, rep, pred, dt=self.dt, c=self.substep_coeff)
        return aux_next, rep_next

    def heads(self, rep: torch.Tensor, aux: torch.Tensor
              ) -> Dict[str, torch.Tensor]:
        """Predictions on a (predicted or real) model state."""
        h_in = torch.cat([self.norm_s(rep), aux], dim=-1)
        on = self.obs_head(h_in)
        return {
            "obs_norm": on,
            "obs": on * self.o_std + self.o_mean,
            "phi": self.phi_head(h_in).squeeze(-1),
            "stage_logits": self.stage_head(h_in),
        }

    def step(self, rep: torch.Tensor, aux: torch.Tensor, a: torch.Tensor):
        aux_n, rep_n = self.next_state(rep, aux, a)
        return rep_n, aux_n, self.heads(rep_n, aux_n)

    # --- state conversion helpers ------------------------------------------
    @staticmethod
    def rep_aux_from_cs(cs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cs -> (rep, aux)."""
        return gc.represent(cs), gc.extract_aux(cs)

    @staticmethod
    def cs_from_rep_aux(rep: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return gc.reconstruct_cs(aux, rep)


def grouped_state_metrics(s_pred: torch.Tensor, s_true: torch.Tensor,
                          ) -> Dict[str, float]:
    """Raw-unit error groups used by validation (on reconstructed cs):
    root z MAE (m), rotation angle (deg), per-group NRMSE."""
    with torch.no_grad():
        p = s_pred.detach().float().cpu()
        t = s_true.detach().float().cpu()
        out: Dict[str, float] = {}
        out["root_z_mae"] = float((p[..., 2] - t[..., 2]).abs().mean())
        qp = F.normalize(p[..., gc.CS_QUAT], p=2, dim=-1)
        qt = F.normalize(t[..., gc.CS_QUAT], p=2, dim=-1)
        cosang = (qp * qt).sum(-1).abs().clamp(max=1.0)
        out["rot_deg_mean"] = float(
            torch.rad2deg(2.0 * torch.acos(cosang)).mean())
        out["rot_deg_p95"] = float(torch.quantile(
            torch.rad2deg(2.0 * torch.acos(cosang)), 0.95))
        groups = {
            "linvel": slice(7, 10), "angvel": slice(10, 13),
            "jpos": slice(13, 34), "jvel": slice(34, 55),
        }
        for name, sl in groups.items():
            err = (p[..., sl] - t[..., sl])
            nrmse = err.pow(2).mean().sqrt() / (
                t[..., sl].std().clamp_min(1e-6))
            out[f"nrmse_{name}"] = float(nrmse)
        return out


def bryson_state_loss(model: SelfWorldModel, rep: torch.Tensor,
                      aux: torch.Tensor, rep_t: torch.Tensor,
                      aux_t: torch.Tensor) -> torch.Tensor:
    """Σ_group w_g * width_g * MSE(pred_block_g, true_block_g) / Σ width.

    Mirrors the official _compute_losses weighting.
    """
    pb = gc.loss_blocks(rep, aux)
    tb = gc.loss_blocks(rep_t, aux_t)
    num = den = 0.0
    for name, blk in pb.items():
        w = model._loss_weights[name]
        num = num + w * blk.shape[-1] * F.mse_loss(blk, tb[name])
        den = den + blk.shape[-1]
    return num / den


def fit_model(model: SelfWorldModel, opt: torch.optim.Optimizer,
              replay, cfg, rng: np.random.Generator, cur_update: int,
              device: torch.device) -> Dict[str, float]:
    """Autoregressive fit over k-step windows from the fit pool."""
    stats = {"s": 0.0, "obs": 0.0, "phi": 0.0, "stage": 0.0, "total": 0.0}
    for _ in range(cfg.model_fit_batches):
        w = replay.sample_windows(
            cfg.model_batch, cfg.model_k, rng,
            recent_frac=cfg.model_recent_frac,
            recent_updates=cfg.model_recent_updates,
            cur_update=cur_update)
        cs0 = torch.as_tensor(w["s0"], device=device)
        acts = torch.as_tensor(w["act"], device=device)
        cs_t = torch.as_tensor(w["cs"], device=device)
        obs_t = torch.as_tensor(w["obs"], device=device)
        phi_t = torch.as_tensor(w["phi"], device=device)
        stage_t = torch.as_tensor(w["stage"], device=device).long() - 1

        rep_t_all = gc.represent(cs_t)
        aux_t_all = gc.extract_aux(cs_t)
        rep = gc.represent(cs0)
        aux = gc.extract_aux(cs0)

        L_s = L_obs = L_phi = L_stage = 0.0
        for m in range(cfg.model_k):
            aux, rep = model.next_state(rep, aux, acts[:, m])
            h = model.heads(rep, aux)
            L_s = L_s + bryson_state_loss(
                model, rep, aux, rep_t_all[:, m], aux_t_all[:, m])
            L_obs = L_obs + F.mse_loss(h["obs_norm"],
                                       model.norm_o(obs_t[:, m]))
            L_phi = L_phi + F.mse_loss(h["phi"], phi_t[:, m])
            L_stage = L_stage + F.cross_entropy(
                h["stage_logits"], stage_t[:, m])
        loss = (cfg.w_state * L_s + cfg.w_obs * L_obs
                + cfg.w_phi * L_phi + cfg.w_stage * L_stage) / cfg.model_k
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        stats["s"] += float(L_s) / cfg.model_k
        stats["obs"] += float(L_obs) / cfg.model_k
        stats["phi"] += float(L_phi) / cfg.model_k
        stats["stage"] += float(L_stage) / cfg.model_k
        stats["total"] += float(loss)
    n = max(cfg.model_fit_batches, 1)
    return {k: v / n for k, v in stats.items()}
