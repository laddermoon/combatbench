"""Differentiable imagined rollout + BPTT policy improvement.

Mirrors HybridRobotics/supermimic `learners/policy_learner.py`
(ActorCriticPolicyLearner): imagine H steps from real seed states with
deterministic actions, accumulate the survival-masked return plus a
rep-critic bootstrap, take ONE actor step on -J, then fit the critic on
the same imagined rollout via GAE returns.

Model parameters are frozen (requires_grad False) but the computation
graph stays live — gradients flow obs -> action -> dynamics -> next obs
into the actor.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..hybrid_ppo.model import RepCritic, SelfWorldModel
from .task import (MAX_STEP_REWARD, STEP_PHI_COEF, W_FALL, analytic_phi,
                   compose_obs)


def gae_returns(values: torch.Tensor, last_value: torch.Tensor,
                rewards: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float) -> torch.Tensor:
    """Port of supermimic gae_returns (time-major)."""
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = (rewards[t] + not_done * gamma * next_value
                 - values[t])
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


class ImaginedRollout:
    """Stateful differentiable env over the frozen world model."""

    def __init__(self, model: SelfWorldModel, actor, vrep: RepCritic,
                 horizon: int = 16, gamma: float = 0.99, lam: float = 0.95):
        self.model = model
        self.actor = actor
        self.vrep = vrep
        self.H = int(horizon)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.v_max = MAX_STEP_REWARD / (1.0 - gamma)

    def _obs(self, rep: torch.Tensor, aux: torch.Tensor,
             heads: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if heads is None:
            with torch.no_grad():
                heads = self.model.heads(rep, aux)
        return compose_obs(rep, aux, heads["obs"][..., 52:54])

    def rollout(self, cs0: torch.Tensor, o0: torch.Tensor):
        """Run H imagined steps; returns (loss, stats, critic data).

        ``o0`` is the REAL recorded observation at the seed — keeps the
        first action anchored on true obs (incl. foot forces)."""
        model, actor, vrep = self.model, self.actor, self.vrep
        rep, aux = model.rep_aux_from_cs(cs0)
        o = o0

        J = 0.0
        survival = torch.ones(cs0.shape[0], device=cs0.device)
        reps_seq, auxs_seq, rews, terms, obs_seq = [], [], [], [], []
        for t in range(self.H):
            a = actor.deterministic_action(o)
            # critic input = PRE-step state s_t (official convention):
            # V(s_t) targets r_{t+1} + gamma * V(s_{t+1}), so the
            # bootstrap term gamma^H * V(s_H) cannot double-count r(s_H).
            reps_seq.append(rep.detach())
            auxs_seq.append(aux.detach())
            aux, rep = model.next_state(rep, aux, a)
            hd = model.heads(rep, aux)
            p_term = torch.sigmoid(hd["done_logit"])
            # scalar-equivalent of the real two-channel objective:
            # r_fall*w_fall + r_cross*(w_cross*phi^2), phi clipped [0,1]
            phi = analytic_phi(rep).clamp(0.0, 1.0)
            r = (STEP_PHI_COEF * W_FALL * phi
                 + phi * phi * hd["aux_rew"].clamp(-1.0, 1.0))
            J = J + (self.gamma ** t) * survival * r
            survival = survival * (1.0 - p_term)
            o = self._obs(rep, aux, hd)
            rews.append(r.detach())
            terms.append(p_term.detach())
            obs_seq.append(o.detach())

        with torch.no_grad():
            v_last = vrep(rep, aux).clamp(max=self.v_max)
        J = J + (self.gamma ** self.H) * v_last * survival
        loss = -J.mean()

        crit = {
            "rep": torch.stack(reps_seq, 0),      # (H,B,52)
            "aux": torch.stack(auxs_seq, 0),      # (H,B,7)
            "reward": torch.stack(rews, 0),       # (H,B)
            "term": torch.stack(terms, 0),        # (H,B)
            "last_rep": rep.detach(),
            "last_aux": aux.detach(),
        }
        stats = {
            "J": float(J.mean()),
            "alive_end": float(survival.mean()),
            "r_mean": float(torch.stack(rews).mean()),
        }
        return loss, stats, crit

    def fit_critic(self, crit, opt_c: torch.optim.Optimizer,
                   epochs: int = 5, minibatches: int = 4) -> float:
        """GAE-return regression on the imagined rollout (official 5x4)."""
        reps, auxs = crit["rep"], crit["aux"]
        H, B = reps.shape[0], reps.shape[1]
        with torch.no_grad():
            vals = torch.stack(
                [self.vrep(reps[t], auxs[t]) for t in range(H)], 0)
            last_v = self.vrep(crit["last_rep"], crit["last_aux"])
            targets = gae_returns(
                vals, last_v, crit["reward"], crit["term"],
                self.gamma, self.lam).clamp(max=self.v_max)
        xs = torch.cat([reps, auxs], -1).reshape(H * B, -1)
        ys = targets.reshape(-1)
        n4 = len(ys) // minibatches * minibatches
        perm = torch.randperm(n4, device=ys.device)
        self.vrep.requires_grad_(True)
        loss = 0.0
        for _ in range(epochs):
            for idx in perm.view(minibatches, -1):
                x = xs[idx]
                l = ((self.vrep(x[..., :52], x[..., 52:]) - ys[idx])
                     ** 2).mean()
                opt_c.zero_grad(set_to_none=True)
                l.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vrep.parameters(), 1.0)
                opt_c.step()
                loss += float(l)
        return loss / (epochs * minibatches)


def bptt_update(model: SelfWorldModel, actor, vrep: RepCritic,
                opt_a: torch.optim.Optimizer,
                opt_c: torch.optim.Optimizer,
                cs0: torch.Tensor, o0: torch.Tensor, horizon: int = 16,
                gamma: float = 0.99, lam: float = 0.95,
                grad_elem_clip: float = 0.1,
                ) -> Dict[str, float]:
    """One imagined rollout + one actor step + critic fit (one batch)."""
    # Freeze weights, keep the graph live for the actor gradient.
    for p in list(model.parameters()) + list(vrep.parameters()):
        p.requires_grad_(False)
    actor.train()

    env = ImaginedRollout(model, actor, vrep, horizon, gamma, lam)
    loss, stats, crit = env.rollout(cs0, o0)
    opt_a.zero_grad(set_to_none=True)
    loss.backward()
    for p in actor.parameters():
        if p.grad is not None:
            p.grad.clamp_(-grad_elem_clip, grad_elem_clip)
    gnorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_a.step()
    stats["vloss"] = env.fit_critic(crit, opt_c)
    stats["actor_gnorm"] = float(gnorm)
    for p in list(model.parameters()) + list(vrep.parameters()):
        p.requires_grad_(True)
    return stats
