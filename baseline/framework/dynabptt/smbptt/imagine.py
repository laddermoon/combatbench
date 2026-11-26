"""Differentiable imagined rollout + BPTT policy improvement.

Mirrors HybridRobotics/supermimic `learners/policy_learner.py`
(ActorCriticPolicyLearner): imagine H steps from real seed states,
accumulate the survival-masked return plus a rep-critic bootstrap,
take ONE actor step on -J, then fit the critic on the same imagined
rollout via GAE returns.

Deviation from official: they use deterministic actions in imagination
because their collection sigma is 0.1; ours is ~0.37, so we sample
actions via the reparameterization trick — imagination must see the
deployment-time action noise to stay honest.

Model parameters are frozen (requires_grad False) but the computation
graph stays live — gradients flow obs -> action -> dynamics -> next obs
into the actor.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..hybrid_ppo.model import RepCritic, SelfWorldModel
from .task import (MAX_STEP_REWARD, OBS_NOISE_HALF, STEP_PHI_COEF, W_FALL,
                   analytic_phi,
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
                 horizon: int = 16, gamma: float = 0.99, lam: float = 0.95,
                 stochastic: bool = True, done_mode: str = "learned",
                 phi_done: float = 0.45, obs_noise: float = 0.0,
                 rew_mode: str = "combined"):
        self.model = model
        self.actor = actor
        self.vrep = vrep
        self.H = int(horizon)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.v_max = MAX_STEP_REWARD / (1.0 - gamma)
        self.stochastic = bool(stochastic)
        # done_mode 'learned': done head (inherits data-policy failure
        # stats — pessimistic); 'analytic': sigmoid((phi_done-phi)/0.05),
        # only deep falls terminate, moderate dips stay survivable.
        self.done_mode = done_mode
        self.phi_done = float(phi_done)
        # official obs-jitter scale (0 = off); only the actor's imagined
        # input is noised — critic/model see the clean rep.
        self.obs_noise = float(obs_noise)
        self._noise_half = None
        # rew_mode: 'combined' (raw scalarization — INVERTED: r_cross floor
        # -0.24/step makes dying outscore surviving), 'fall_only'
        # (0.03*phi survival channel only), 'cross_centered' (r_fall +
        # phi^2*(aux_rew - EMA baseline): the penalty channel can only
        # reward better-than-average stepping, never punish existence).
        self.rew_mode = rew_mode
        self._aux_ema = -0.24  # measured mean on teacher data

    def _obs(self, rep: torch.Tensor, aux: torch.Tensor,
             heads: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if heads is None:
            with torch.no_grad():
                heads = self.model.heads(rep, aux)
        o = compose_obs(rep, aux, heads["obs"][..., 52:54])
        if self.obs_noise > 0.0:
            if self._noise_half is None:
                self._noise_half = torch.as_tensor(
                    OBS_NOISE_HALF, dtype=o.dtype, device=o.device)
            o = o + (torch.rand_like(o) * 2 - 1) * (
                self._noise_half * self.obs_noise)
        return o

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
            # Stochastic actions via reparam: real collection runs sigma~0.37,
            # so imagination must see the same action noise to stay honest
            # (deterministic imagination produced phi~1.0 while reality
            # decayed to 0.76 — the trained policy never met its own noise).
            if self.stochastic:
                a, _ = actor.sample_action(o)
            else:
                a = actor.deterministic_action(o)
            # critic input = PRE-step state s_t (official convention):
            # V(s_t) targets r_{t+1} + gamma * V(s_{t+1}), so the
            # bootstrap term gamma^H * V(s_H) cannot double-count r(s_H).
            reps_seq.append(rep.detach())
            auxs_seq.append(aux.detach())
            aux, rep = model.next_state(rep, aux, a)
            hd = model.heads(rep, aux)
            if self.done_mode == "analytic":
                p_term = torch.sigmoid(
                    (self.phi_done - analytic_phi(rep).clamp(0.0, 1.0))
                    / 0.05)
            else:
                p_term = torch.sigmoid(hd["done_logit"])
            # scalar-equivalent of the real two-channel objective:
            # r_fall*w_fall + r_cross*(w_cross*phi^2), phi clipped [0,1]
            phi = analytic_phi(rep).clamp(0.0, 1.0)
            r_fall = STEP_PHI_COEF * W_FALL * phi
            if self.rew_mode == "fall_only":
                r = r_fall
            elif self.rew_mode == "cross_centered":
                r = r_fall + phi * phi * (
                    hd["aux_rew"].clamp(-1.0, 1.0) - self._aux_ema)
            else:
                r = r_fall + phi * phi * hd["aux_rew"].clamp(-1.0, 1.0)
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


def fit_critic_real(model: SelfWorldModel, vrep: RepCritic,
                    opt_c: torch.optim.Optimizer, trajs,
                    gamma: float, lam: float, v_max: float,
                    minibatches: int = 8) -> float:
    """Fit rep-critic on REAL trajectories (GAE returns on real
    rewards/dones) — grounds the bootstrap so model errors cannot
    corrupt V. trajs: AlignedTraj list (cs/phi/aux_rew/done)."""
    import numpy as np
    xs, ys = [], []
    for tr in trajs:
        cs = torch.as_tensor(tr.cs, dtype=torch.float32,
                             device=next(vrep.parameters()).device)
        rep, aux = model.rep_aux_from_cs(cs)
        with torch.no_grad():
            phi = analytic_phi(rep).clamp(0.0, 1.0)
            auxr = torch.as_tensor(tr.aux_rew, dtype=torch.float32,
                                   device=cs.device).clamp(-1.0, 1.0)
            rew = (STEP_PHI_COEF * W_FALL * phi
                   + phi * phi * auxr)
            done = torch.as_tensor(
                tr.done if tr.done is not None else np.zeros(tr.T),
                dtype=torch.float32, device=cs.device)
            ret = torch.zeros(tr.T, device=cs.device)
            g = torch.zeros((), device=cs.device)
            for t in reversed(range(tr.T)):
                nd = 1.0 - done[t]
                g = rew[t] + gamma * g * nd
                ret[t] = g
            ret = ret.clamp(max=v_max)
        xs.append(torch.cat([rep, aux], -1))
        ys.append(ret)
    if not xs:
        return 0.0
    X = torch.cat(xs); Y = torch.cat(ys)
    vrep.requires_grad_(True)
    loss = 0.0
    n = 0
    perm = torch.randperm(len(Y), device=Y.device)[:minibatches * 4096]
    for i in range(0, len(perm), 4096):
        idx = perm[i:i + 4096]
        l = ((vrep(X[idx, :52], X[idx, 52:]) - Y[idx]) ** 2).mean()
        opt_c.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(vrep.parameters(), 1.0)
        opt_c.step()
        loss += float(l); n += 1
    return loss / max(n, 1)


def bptt_update(model: SelfWorldModel, actor, vrep: RepCritic,
                opt_a: torch.optim.Optimizer,
                opt_c: torch.optim.Optimizer,
                cs0: torch.Tensor, o0: torch.Tensor, horizon: int = 16,
                gamma: float = 0.99, lam: float = 0.95,
                grad_elem_clip: float = 0.1,
                stochastic: bool = True, done_mode: str = "learned",
                phi_done: float = 0.45,
                critic_trajs=None, obs_noise: float = 0.0,
                rew_mode: str = "combined",
                ) -> Dict[str, float]:
    """One imagined rollout + one actor step + critic fit (one batch)."""
    # Freeze weights, keep the graph live for the actor gradient.
    for p in list(model.parameters()) + list(vrep.parameters()):
        p.requires_grad_(False)
    actor.train()

    env = ImaginedRollout(model, actor, vrep, horizon, gamma, lam,
                          stochastic=stochastic, done_mode=done_mode,
                          phi_done=phi_done, obs_noise=obs_noise,
                          rew_mode=rew_mode)
    loss, stats, crit = env.rollout(cs0, o0)
    opt_a.zero_grad(set_to_none=True)
    loss.backward()
    for p in actor.parameters():
        if p.grad is not None:
            p.grad.clamp_(-grad_elem_clip, grad_elem_clip)
    gnorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_a.step()
    if critic_trajs is not None:
        stats["vloss"] = fit_critic_real(
            model, vrep, opt_c, critic_trajs, gamma, lam, env.v_max)
    else:
        stats["vloss"] = env.fit_critic(crit, opt_c)
    stats["actor_gnorm"] = float(gnorm)
    for p in list(model.parameters()) + list(vrep.parameters()):
        p.requires_grad_(True)
    return stats
