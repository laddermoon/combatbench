"""Compact PPO update replicating baseline.framework.ppo semantics for
the single-channel self-only standup task.

Replicated semantics (source: baseline/framework/ppo/trainer.py):
  * values computed once at update start with the pre-update critic
  * GAE (gamma=0.99, lam=0.95); timeout => bootstrap V(last_obs)
  * per-channel adv: zscore over active frames; combined = conf * zadv,
    conf = sqrt(clip(EV,0,1)) and EV = 1 - Var(ret - V)/Var(ret)
  * surrogate: min(ratio*A, clip(ratio,1±eps)*A), eps=0.2
  * old log_prob recomputed under theta_old via evaluate_actions
    (ctx-free == explore_factor 0: identical by construction)
  * k3 KL = mean((ratio-1) - log ratio); early-stop actor when the mean
    of the last ``kl_window`` minibatch KLs exceeds target_kl; critics
    keep updating afterwards
  * per-minibatch order: critic step first, then actor step
  * value loss: masked MSE on returns, normalized by active count
  * uncertainty-floor term kept for fidelity (floor=0.0 => inactive)

The optional model aux loss adds ``alpha * mean(surrogate_model)`` where
model samples carry their own (o,a,old_lp,adv) — see rollout.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from baseline.framework.ppo.algos.advantages import compute_gae

from .transitions import AlignedTraj


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    target_kl: float = 0.05
    epochs: int = 4
    minibatch: int = 4096
    kl_window: int = 10
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    uncertainty_coef: float = 0.001
    uncertainty_floor: float = 0.0
    use_confidence: bool = True
    # model aux loss
    alpha: float = 0.0          # 0 => sidecar only (P1), >0 => gated
    model_minibatch: int = 1024


@dataclass
class PPOBatch:
    obs: np.ndarray       # (N,57)
    act: np.ndarray       # (N,21)
    adv: np.ndarray       # (N,)
    ret: np.ndarray       # (N,)
    val: np.ndarray       # (N,)  V_old per frame


def build_ppo_batch(trajs: List[AlignedTraj], critic,
                    device: torch.device, gamma: float,
                    lam: float) -> PPOBatch:
    """Flatten trajectories into a PPO batch with GAE advantages."""
    obs_l, act_l, adv_l, ret_l, val_l = [], [], [], [], []
    with torch.no_grad():
        for tr in trajs:
            o = torch.as_tensor(tr.obs, device=device)
            v = critic(o).cpu().numpy()
            lo = torch.as_tensor(tr.final_obs, device=device).unsqueeze(0)
            last_v = 0.0 if tr.terminated else float(critic(lo).item())
            adv, ret = compute_gae(
                tr.phi * 0.01, v, last_value=last_v,
                gamma=gamma, lam=lam)
            obs_l.append(tr.obs)
            act_l.append(tr.act)
            adv_l.append(adv)
            ret_l.append(ret)
            val_l.append(v)
    return PPOBatch(
        obs=np.concatenate(obs_l).astype(np.float32),
        act=np.concatenate(act_l).astype(np.float32),
        adv=np.concatenate(adv_l).astype(np.float32),
        ret=np.concatenate(ret_l).astype(np.float32),
        val=np.concatenate(val_l).astype(np.float32),
    )


def _explained_variance(ret: np.ndarray, val: np.ndarray) -> float:
    var_r = float(ret.var())
    if var_r < 1e-12:
        return 0.0
    return float(1.0 - np.var(ret - val) / var_r)


def _surrogate(ratio: torch.Tensor, adv: torch.Tensor,
               clip_eps: float) -> torch.Tensor:
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return torch.min(s1, s2)


def ppo_update(actor, critic, opt_a, opt_c, batch: PPOBatch,
               cfg: PPOConfig, device: torch.device,
               model_batch: Optional[Dict[str, np.ndarray]] = None,
               rng: Optional[np.random.Generator] = None,
               ) -> Dict[str, float]:
    """One PPO update.  Returns flat stats for metrics.jsonl.

    ``model_batch``: optional dict(obs, act, old_lp, adv) — aux
    surrogate term scaled by cfg.alpha.  The model data never touches
    the critic.
    """
    rng = rng or np.random.default_rng(0)
    obs_t = torch.as_tensor(batch.obs, device=device)
    act_t = torch.as_tensor(batch.act, device=device)
    ret_t = torch.as_tensor(batch.ret, device=device)
    n = obs_t.shape[0]

    # --- old log_prob under theta_old (chunked, no grad) ---
    old_lp = torch.empty(n, device=device)
    with torch.no_grad():
        for i in range(0, n, 8192):
            old_lp[i:i + 8192] = actor.evaluate_actions(
                obs_t[i:i + 8192], act_t[i:i + 8192]).log_prob

    # --- advantage combine: zscore * confidence ---
    ev = _explained_variance(batch.ret, batch.val)
    conf = float(np.clip(ev, 0.0, 1.0) ** 0.5) if cfg.use_confidence else 1.0
    std = float(batch.adv.std())
    zadv = (batch.adv - batch.adv.mean()) / max(std, 1e-8)
    combined = torch.as_tensor(conf * zadv, dtype=torch.float32,
                               device=device)

    stats: Dict[str, float] = {
        "ev": ev, "conf": conf, "adv_raw_std": std,
    }

    # --- model aux arrays ---
    m_obs = m_act = m_lp = m_adv = None
    if model_batch is not None and cfg.alpha > 0.0:
        m_obs = torch.as_tensor(model_batch["obs"], device=device)
        m_act = torch.as_tensor(model_batch["act"], device=device)
        m_lp = torch.as_tensor(model_batch["old_lp"], device=device)
        m_adv = torch.as_tensor(model_batch["adv"], device=device)

    # first-minibatch ratio fidelity guard (pre-update: must be ~1)
    with torch.no_grad():
        idx0 = torch.arange(min(4096, n), device=device)
        dev = float(torch.exp(
            actor.evaluate_actions(obs_t[idx0], act_t[idx0]).log_prob
            - old_lp[idx0]).sub(1.0).abs().max())
    stats["first_mb_ratio_dev"] = dev

    kl_hist: List[float] = []
    actor_stopped = False
    acc = {"ploss": [], "vloss": [], "kl": [], "clipfrac": [],
           "mloss": [], "mkl": [], "n_actor_mb": 0}
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        m_perm = (rng.permutation(m_obs.shape[0])
                  if m_obs is not None else None)
        for mb_i in range(0, n, cfg.minibatch):
            idx = torch.as_tensor(
                perm[mb_i:mb_i + cfg.minibatch], device=device)

            # --- critic first (reference order) ---
            opt_c.zero_grad(set_to_none=True)
            v = critic(obs_t[idx])
            vloss = ((v - ret_t[idx]) ** 2).mean()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),
                                           cfg.grad_clip)
            opt_c.step()
            acc["vloss"].append(float(vloss))

            if actor_stopped:
                continue
            # --- actor: real surrogate ---
            ev_act = actor.evaluate_actions(obs_t[idx], act_t[idx])
            new_lp = ev_act.log_prob
            log_r = torch.clamp(new_lp - old_lp[idx], -20.0, 20.0)
            ratio = torch.exp(log_r)
            with torch.no_grad():
                kl = float(((ratio - 1.0) - log_r).mean())
                clipfrac = float(((ratio > 1 + cfg.clip_eps)
                                  | (ratio < 1 - cfg.clip_eps)).float().mean())
            kl_hist.append(kl)
            acc["kl"].append(kl)
            acc["clipfrac"].append(clipfrac)
            surr = _surrogate(ratio, combined[idx], cfg.clip_eps)
            loss = -surr.mean()
            if cfg.uncertainty_floor > 0.0:
                loss = loss + cfg.uncertainty_coef * torch.relu(
                    cfg.uncertainty_floor - ev_act.uncertainty).mean()

            # --- actor: model surrogate (same optimizer step) ---
            if m_obs is not None:
                mb_no = mb_i // cfg.minibatch
                M = m_obs.shape[0]
                midx_np = m_perm[
                    (mb_no * cfg.model_minibatch
                     + np.arange(cfg.model_minibatch)) % M]
                midx = torch.as_tensor(midx_np, device=device)
                if midx.numel() > 0:
                    m_ev = actor.evaluate_actions(m_obs[midx], m_act[midx])
                    m_logr = torch.clamp(m_ev.log_prob - m_lp[midx],
                                         -20.0, 20.0)
                    m_ratio = torch.exp(m_logr)
                    with torch.no_grad():
                        acc["mkl"].append(float(
                            ((m_ratio - 1.0) - m_logr).mean()))
                    m_surr = _surrogate(m_ratio, m_adv[midx],
                                        cfg.clip_eps)
                    loss = loss - cfg.alpha * m_surr.mean()
                    acc["mloss"].append(float(-m_surr.mean()))

            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),
                                           cfg.grad_clip)
            opt_a.step()
            acc["ploss"].append(float(loss))
            acc["n_actor_mb"] += 1

            if (cfg.target_kl > 0.0
                    and len(kl_hist) >= cfg.kl_window
                    and float(np.mean(kl_hist[-cfg.kl_window:]))
                    > cfg.target_kl):
                actor_stopped = True

    stats.update({
        "policy_loss": float(np.mean(acc["ploss"])) if acc["ploss"] else float("nan"),
        "value_loss": float(np.mean(acc["vloss"])) if acc["vloss"] else float("nan"),
        "kl_mean": float(np.mean(acc["kl"])) if acc["kl"] else 0.0,
        "kl_max": float(np.max(acc["kl"])) if acc["kl"] else 0.0,
        "clipfrac": float(np.mean(acc["clipfrac"])) if acc["clipfrac"] else 0.0,
        "model_loss": float(np.mean(acc["mloss"])) if acc["mloss"] else float("nan"),
        "model_kl": float(np.mean(acc["mkl"])) if acc["mkl"] else float("nan"),
        "actor_mb": float(acc["n_actor_mb"]),
        "actor_stopped": float(actor_stopped),
    })
    return stats
