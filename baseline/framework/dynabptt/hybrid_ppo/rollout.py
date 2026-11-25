"""Short model branch generation for the model-assisted PPO aux loss.

Semantics (plan §5.1): seeds are REAL visited states from the current
update's fit pool, restricted to admission-passed stages.  From each
seed we roll the model h steps with actions sampled from the frozen
pi_old on predicted observations, compute a V_old-bootstrapped GAE, and
return ONLY the first-action advantage — the aux PPO update reuses the
real seed frame (o_0, a_0, old_log_prob_0).

The world model never receives gradients from this path.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import SelfWorldModel
from .standup import REWARD_COEF
from .validation import _gae


@torch.no_grad()
def gen_model_batch(model: SelfWorldModel, actor, critic,
                    seeds: List[Tuple], horizon: int,
                    device: torch.device,
                    gamma: float = 0.99, lam: float = 0.95,
                    ) -> Optional[Dict[str, np.ndarray]]:
    """Generate first-action advantages for a set of real seed states.

    seeds: list of (traj, t) — t >= 1; the seed is (cs[t-1], obs[t]).
    Returns dict with:
        obs      (N,57) real seed observations o_t
        act      (N,21) actions sampled from pi_old(o_t)
        old_lp   (N,)   log_prob under pi_old
        adv      (N,)   model-branch first-action GAE advantage
        phi_path (N,h)  predicted per-step phi (diagnostics)
    """
    if not seeds:
        return None
    s0 = torch.as_tensor(
        np.stack([tr.cs[t - 1] for tr, t in seeds]), device=device)
    o0 = torch.as_tensor(
        np.stack([tr.obs[t] for tr, t in seeds]), device=device)
    B = s0.shape[0]

    a0, lp0 = actor.sample_action(o0)
    rep, aux = model.rep_aux_from_cs(s0)
    o = o0
    acts = [a0]
    phis: List[torch.Tensor] = []
    obs_seq: List[torch.Tensor] = []
    for m in range(horizon):
        if m > 0:
            a, _ = actor.sample_action(o)
            acts.append(a)
        aux, rep = model.next_state(rep, aux, acts[m])
        hd = model.heads(rep, aux)
        o = hd["obs"]
        obs_seq.append(o)
        phis.append(hd["phi"])

    phi_path = torch.stack(phis, 1)                        # (B,h)
    obs_all = torch.cat([o0.unsqueeze(1),
                         torch.stack(obs_seq, 1)], dim=1)  # (B,h+1,57)
    v_all = critic(obs_all.reshape(-1, o0.shape[-1])).reshape(B, -1)
    adv = _gae(REWARD_COEF * phi_path, v_all[:, :-1], v_all[:, -1],
               gamma, lam)

    return {
        "obs": o0.cpu().numpy(),
        "act": a0.cpu().numpy(),
        "old_lp": lp0.cpu().numpy(),
        "adv": adv.cpu().numpy(),
        "phi_path": phi_path.cpu().numpy(),
    }


def normalize_model_adv(adv: np.ndarray, real_adv_std: float,
                        min_scale: float = 1e-6) -> np.ndarray:
    """Center within the model batch, scale by the REAL advantage std.

    Never divides by the model's own (possibly tiny) std — that would
    amplify noise into full-scale advantages.
    """
    scale = max(float(real_adv_std), min_scale)
    out = (adv - adv.mean()) / scale
    return np.clip(out, -5.0, 5.0).astype(np.float32)
