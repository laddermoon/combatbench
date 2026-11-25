"""SelfWorldModel — local dynamics model for the self-only standup task.

    F(s_t, a_t) -> s_{t+1}          residual dynamics (quat renormed)
    G(s_{t+1})   -> o_{t+1}         57-dim self observation
    R(s_{t+1})   -> phi_{t+1}, stage logits

Differentiability is NOT required by the model-assisted-PPO path; the
trunk is a plain residual MLP.  All per-dim standardization uses
train-pool statistics refit each update and stored on the module so
checkpoints are self-contained.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUAT_SLICE = slice(3, 7)
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
        state_dim: int = 55,
        action_dim: int = 21,
        obs_dim: int = 57,
        hidden: Tuple[int, ...] = (512, 512, 512),
        head_hidden: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)

        self.dyn = _mlp(state_dim + action_dim, hidden, state_dim)
        self.obs_head = _mlp(state_dim, head_hidden, obs_dim)
        self.phi_head = _mlp(state_dim, head_hidden, 1)
        self.stage_head = _mlp(state_dim, head_hidden, N_STAGES)

        # Per-dim standardization, fit on the current train pool.
        # Buffers (not parameters): saved in checkpoints, no gradients.
        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))
        self.register_buffer("d_mean", torch.zeros(state_dim))
        self.register_buffer("d_std", torch.ones(state_dim))
        self.register_buffer("o_mean", torch.zeros(obs_dim))
        self.register_buffer("o_std", torch.ones(obs_dim))

    # --- normalization helpers -------------------------------------------
    def norm_s(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.s_mean) / self.s_std

    def norm_o(self, o: torch.Tensor) -> torch.Tensor:
        return (o - self.o_mean) / self.o_std

    def fit_normalization(self, windows: Dict[str, np.ndarray]) -> None:
        """Estimate s/ds/obs stats from a sampled window batch."""
        s0 = torch.as_tensor(windows["s0"], dtype=torch.float32)
        cs = torch.as_tensor(windows["cs"], dtype=torch.float32)
        obs = torch.as_tensor(windows["obs"], dtype=torch.float32)
        s_all = torch.cat([s0.unsqueeze(1), cs], dim=1).reshape(-1, self.state_dim)
        d_all = (cs - torch.cat([s0.unsqueeze(1), cs[:, :-1]], dim=1)
                 ).reshape(-1, self.state_dim)
        o_all = obs.reshape(-1, self.obs_dim)
        eps = 1e-6
        self.s_mean.copy_(s_all.mean(0))
        self.s_std.copy_(s_all.std(0).clamp_min(eps))
        self.d_mean.copy_(d_all.mean(0))
        self.d_std.copy_(d_all.std(0).clamp_min(eps))
        self.o_mean.copy_(o_all.mean(0))
        self.o_std.copy_(o_all.std(0).clamp_min(eps))

    # --- forward ----------------------------------------------------------
    def next_state(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        delta_n = self.dyn(torch.cat([self.norm_s(s), a], dim=-1))
        s_next = s + delta_n * self.d_std + self.d_mean
        q = s_next[..., QUAT_SLICE]
        s_next = torch.cat(
            [s_next[..., :3], F.normalize(q, p=2, dim=-1), s_next[..., 7:]],
            dim=-1,
        )
        return s_next

    def heads(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predictions on a (predicted or real) state s."""
        sn = self.norm_s(s)
        on = self.obs_head(sn)
        return {
            "obs_norm": on,
            "obs": on * self.o_std + self.o_mean,
            "phi": self.phi_head(sn).squeeze(-1),
            "stage_logits": self.stage_head(sn),
        }

    def step(self, s: torch.Tensor, a: torch.Tensor):
        s_next = self.next_state(s, a)
        return s_next, self.heads(s_next)


def grouped_state_metrics(s_pred: torch.Tensor, s_true: torch.Tensor,
                          ) -> Dict[str, float]:
    """Raw-unit error groups used by validation:
    root z MAE (m), rotation angle (deg), per-group NRMSE."""
    with torch.no_grad():
        p = s_pred.detach().float().cpu()
        t = s_true.detach().float().cpu()
        out: Dict[str, float] = {}
        out["root_z_mae"] = float((p[..., 2] - t[..., 2]).abs().mean())
        qp = F.normalize(p[..., QUAT_SLICE], p=2, dim=-1)
        qt = F.normalize(t[..., QUAT_SLICE], p=2, dim=-1)
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
        s = torch.as_tensor(w["s0"], device=device)
        acts = torch.as_tensor(w["act"], device=device)
        cs_t = torch.as_tensor(w["cs"], device=device)
        obs_t = torch.as_tensor(w["obs"], device=device)
        phi_t = torch.as_tensor(w["phi"], device=device)
        stage_t = torch.as_tensor(w["stage"], device=device).long() - 1

        L_s = L_obs = L_phi = L_stage = 0.0
        for m in range(cfg.model_k):
            s = model.next_state(s, acts[:, m])
            h = model.heads(s)
            L_s = L_s + F.mse_loss(model.norm_s(s), model.norm_s(cs_t[:, m]))
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
