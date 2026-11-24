"""DynaBPTT trainer — fast-validation build (self-only world model).

Loop per update:
  1. collect episodes in the real env (ParallelRollouter, injected
     CoreStateObserver logs per-step core state into observer_outputs)
  2. extract per-agent transitions -> FIFO buffer
  3. fit world model on k-step autoregressive windows
  4. improve policy: imagine H steps from buffer states, BPTT through the
     frozen model, per-channel critics bootstrap beyond H
  5. periodic deterministic eval in the real env

Usage:
  PYTHONPATH=. python3 baseline/framework/dynabptt/train.py \
      --experiment basic_balance
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.ppo.policies.truncated_normal_mlp import TruncatedNormalPolicy
from baseline.framework.rollout.job import Job, SamplingSpec
from baseline.framework.rollout.observer_utils import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from baseline.framework.rollout.parallel_rollouter import ParallelRollouter

from baseline.framework.dynabptt.buffer import AgentTraj, TransitionBuffer
from baseline.framework.dynabptt.world_model import WorldModel

CS_OBSERVER_NAME = "dynabptt_cs"
CS_OBSERVER_SPEC = {
    "cls": "baseline.framework.dynabptt.corestate:CoreStateObserver",
    "config": {},
}


# ---------------------------------------------------------------------------
# Env blueprint injection (dict-level; no blueprint files touched)
# ---------------------------------------------------------------------------
def inject_corestate(env_bp: EnvBlueprint) -> EnvBlueprint:
    d = env_bp.to_dict()
    d.setdefault("observer_plugins", {})[CS_OBSERVER_NAME] = CS_OBSERVER_SPEC
    return EnvBlueprint.from_dict(d)


def inject_push(env_bp: EnvBlueprint, force_min: float, force_max: float,
                interval_min: int, interval_max: int) -> EnvBlueprint:
    """Append RandomPushPlugin (one per robot) for data collection.

    Why: a buffer with only "standing->stand" (teacher) and "tilt->die"
    (learner) transitions contains no *counterfactual* coverage — states
    where the ACTION decides survival. The model therefore predicts doom
    from every tilted seed regardless of action and the boundary gradient
    is noise. Random torso pushes force the anchored policy to produce
    tilt->(recover|die) arcs, which is exactly the missing coverage.
    Eval/ratchet must stay push-free: only collection uses this blueprint.
    """
    d = env_bp.to_dict()
    plugins = d.setdefault("plugins", [])
    for rid in ("robot_a", "robot_b"):
        plugins.append({
            "cls": "envs.humanoid21.disturbance_plugins:RandomPushPlugin",
            "config": {
                "target_robot": rid,
                "force_magnitude_min": float(force_min),
                "force_magnitude_max": float(force_max),
                "min_interval": int(interval_min),
                "max_interval": int(interval_max),
                "push_duration_steps": 1,
            },
        })
    return EnvBlueprint.from_dict(d)


# ---------------------------------------------------------------------------
# Episode -> AgentTraj extraction
# ---------------------------------------------------------------------------
def extract_trajs(episode, agent_ids, aux_keys_fn) -> List[AgentTraj]:
    cs_node = episode.observer_outputs.get(CS_OBSERVER_NAME)
    if cs_node is None:
        raise KeyError(
            f"observer '{CS_OBSERVER_NAME}' missing from episode.observer_outputs "
            f"(available={list(episode.observer_outputs.keys())}) — env blueprint "
            "must be injected via inject_corestate()"
        )
    T_full = episode.num_frames
    trajs: List[AgentTraj] = []
    for aid in agent_ids:
        cs = cs_node.get(aid)
        obs = episode.observations.get(aid)
        act = episode.actions.get(aid)
        if cs is None or obs is None or act is None:
            continue
        B = episode.agent_frame_boundary[aid]
        if B < 2:
            continue
        records = episode.agent_termination_proposal_records.get(aid, ())
        terminal = bool(records) and records[0][0] != "timeout"
        aux: Dict[str, np.ndarray] = {}
        for name, (obs_name, field) in aux_keys_fn(aid).items():
            if field is None:
                arr = extract_per_step_scalar(
                    episode.observer_outputs, obs_name, T_full
                )
            else:
                arr = extract_per_step_field(
                    episode.observer_outputs, obs_name, field, T_full
                )
                if arr is None:
                    raise KeyError(
                        f"observer '{obs_name}' field '{field}' missing "
                        f"(available={list(episode.observer_outputs.keys())})"
                    )
            aux[name] = arr
        trajs.append(AgentTraj(
            cs=np.asarray(cs[:B], dtype=np.float32),
            obs=np.asarray(obs[:B], dtype=np.float32),
            act=np.asarray(act[:B], dtype=np.float32),
            aux={k: np.asarray(v[:B], dtype=np.float32) for k, v in aux.items()},
            terminal=terminal,
            n_trans=B - 1,
        ))
    return trajs


# ---------------------------------------------------------------------------
# World-model fit (k-step autoregressive windows)
# ---------------------------------------------------------------------------
def fit_wm(wm, opt, buf, batches, batch_size, k, device, aux_names, weights,
           recent_frac=0.0, recent_pool=0.25, teacher_share=0.0):
    stats = {"s": 0.0, "obs": 0.0, "aux": 0.0, "done": 0.0, "total": 0.0}
    for _ in range(batches):
        w = buf.sample_windows(batch_size, k, recent_frac=recent_frac,
                               recent_pool=recent_pool,
                               teacher_share=teacher_share)
        s = torch.as_tensor(w["s0"], device=device)
        acts = torch.as_tensor(w["act"], device=device)
        cs_t = torch.as_tensor(w["cs"], device=device)
        obs_t = torch.as_tensor(w["obs"], device=device)
        done_t = torch.as_tensor(w["done"], device=device)
        aux_t = {n: torch.as_tensor(v, device=device) for n, v in w["aux"].items()}

        L_s = L_obs = L_aux = L_done = 0.0
        for m in range(k):
            s = wm.next_state(s, acts[:, m])
            h = wm.heads(s)
            L_s = L_s + F.mse_loss(s, cs_t[:, m])
            L_obs = L_obs + F.mse_loss(h["obs"], obs_t[:, m])
            for name in aux_names:
                L_aux = L_aux + F.mse_loss(h[name], aux_t[name][:, m])
            L_done = L_done + F.binary_cross_entropy_with_logits(
                h["done_logit"], done_t[:, m]
            )
        loss = (
            weights["s"] * L_s + weights["obs"] * L_obs
            + weights["aux"] * L_aux + weights["done"] * L_done
        ) / k
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 10.0)
        opt.step()
        stats["s"] += float(L_s) / k
        stats["obs"] += float(L_obs) / k
        stats["aux"] += float(L_aux) / k
        stats["done"] += float(L_done) / k
        stats["total"] += float(loss)
    return {kk: vv / max(batches, 1) for kk, vv in stats.items()}


# ---------------------------------------------------------------------------
# Policy improvement — BPTT through imagined rollouts
# ---------------------------------------------------------------------------
def _set_requires_grad(module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def improve(actor, critics, tgt_critics, wm, buf, exp, batch, H, device,
            entropy_coef=0.0, use_done=True, seed_start_frac=0.0,
            recent_frac=0.0, recent_pool=0.25,
            seed_tail_frac=0.0, seed_tail_span=16, teacher_share=0.0,
            ref_actor=None, kl_coef=0.0, no_bootstrap=False,
            signal_floor=0.0, pess_lambda=0.0, bc_coef=0.0,
            dagger_net=None, dagger_coef=0.0, dagger_phi_min=0.0,
            si_coef=0.0, si_top_frac=0.1,
            actor_mode="bptt", qcritics=None, tgt_qcritics=None):
    """One imagination batch. Returns (stats dict, detached history) — the
    caller does opt steps so both actor and critic grads share the rollout."""
    for _m in (wm if isinstance(wm, (list, tuple)) else [wm]):
        _set_requires_grad(_m, False)
    for _c in critics.values():
        _set_requires_grad(_c, False)
    if qcritics:
        for _c in qcritics.values():
            _set_requires_grad(_c, False)
    channels = exp.channels  # [{"name","gamma"}]
    aux_names = exp.aux_names

    s_np, o_np = buf.sample_seeds(batch, start_frac=seed_start_frac,
                                  recent_frac=recent_frac,
                                  recent_pool=recent_pool,
                                  tail_frac=seed_tail_frac,
                                  tail_span=seed_tail_span,
                                  teacher_share=teacher_share)
    o_np_t = torch.as_tensor(o_np, device=device)
    wms = wm if isinstance(wm, (list, tuple)) else [wm]

    # Imagined rollout under EACH ensemble member on the same seeds. A
    # phantom gradient direction has to fool every member to survive the
    # pessimistic combination — measured: single-model BPTT proposals at
    # ~0.01 action-RMS score ep_len ~350 vs ~560 for same-magnitude random
    # noise, i.e. the ascent direction is systematically worse than random
    # (it chases per-model Jacobian errors).
    J_members = []
    R_members = []
    s_hist, c_hist, r_hist = [], [], {ch["name"]: [] for ch in channels}
    a_hist, o_hist = [], []
    ent = 0.0
    alive_last = phi0 = last_aux = None
    for m_i, wm_i in enumerate(wms):
        s = torch.as_tensor(s_np, device=device)
        o = o_np_t
        alive = torch.ones(s.shape[0], device=device)
        J_m = torch.zeros(s.shape[0], device=device)
        R_m = {ch["name"]: torch.zeros(s.shape[0], device=device)
               for ch in channels}
        for t in range(H):
            if entropy_coef != 0.0:
                ent = ent + actor.forward(o)[1].log().mean() / len(wms)
            # Deterministic actions inside imagination (SuperMimic recipe):
            # action noise lives in data collection only — it widens the
            # model's action coverage but must not perturb the rollout.
            a = actor.deterministic_action(o)
            if m_i == 0:
                s_hist.append(s.detach())  # s_t — pairs with targets[t]
                a_hist.append(a.detach())
            s, h = wm_i.step(s, a)
            o = h["obs"]
            if m_i == 0:
                o_hist.append(o.detach())  # ô_{t+1}
            # Aux heads are unbounded MLP outputs; imagined states drift
            # off-manifold and the heads extrapolate reward features to
            # absurd values (observed phi_hat ~30 -> phantom reward basin
            # the actor learns to chase while the real policy keeps
            # falling). Clamp predicted aux to each channel's plausible
            # range before any reward/weight computation — the model
            # itself stays unclamped (fit targets are real data; the clamp
            # only removes the actor's ability to arbitrage head
            # extrapolation).
            aux = {}
            for name in aux_names:
                lo_hi = getattr(exp, "aux_clamps", {}).get(name)
                aux[name] = (h[name].clamp(lo_hi[0], lo_hi[1])
                             if lo_hi else h[name])
            if m_i == 0:
                if t == 0 and "phi" in aux:
                    phi0 = aux["phi"].detach()
                last_aux = aux
            if use_done:
                alive = alive * (1.0 - torch.sigmoid(h["done_logit"]))
            for ch in channels:
                name = ch["name"]
                r_c = exp.channel_reward(name, aux)
                w_c = exp.channel_weight(name, aux)
                g_t = ch["gamma"] ** t
                R_m[name] = R_m[name] + g_t * alive * r_c
                J_m = J_m + g_t * alive * w_c * r_c
                if m_i == 0:
                    r_hist[name].append(r_c.detach())
            if m_i == 0:
                c_hist.append(alive.detach())
        J_members.append(J_m)
        R_members.append(R_m)
        if m_i == 0:
            alive_last = alive
            s_fin = s

    J_stack = torch.stack(J_members)  # (K, b)
    J = J_stack.mean(0) - pess_lambda * (
        J_stack.std(0) if len(wms) > 1 else 0.0)
    R = {n: torch.stack([rm[n] for rm in R_members]).mean(0)
         for n in R_members[0]}
    alive = alive_last

    vrange = getattr(exp, "channel_vrange", {})

    def _boot_v(name, states):
        v = tgt_critics[name](states)
        lo_hi = vrange.get(name)
        return v.clamp(lo_hi[0], lo_hi[1]) if lo_hi else v

    last_aux = aux
    vH_stats = {}
    for ch in channels:
        name = ch["name"]
        g_H = ch["gamma"] ** H
        vH = _boot_v(name, s)
        vH_stats[f"vH_{name}"] = float(vH.mean().detach())
        R[name] = R[name] + g_H * alive * vH
        if not no_bootstrap:
            J = J + g_H * alive * exp.channel_weight(name, last_aux) * vH

    # Per-seed informativeness weighting. On a buffer dominated by standing
    # data the model predicts "everything stands" — phi~1, alive~1 — so the
    # imagined objective is *flat* on most seeds and their gradients are pure
    # model/Adam noise. The real signal lives on seeds whose imagination
    # actually reaches the termination boundary (alive drops) or changes phi
    # materially. Weight each seed's J by that signal so the batch gradient
    # is dominated by boundary-adjacent data instead of flat noise.
    if actor_mode == "mbpo":
        # MBPO-style actor: ascend Q(s, pi(s)) on REAL seed states only.
        # The dynamics model is not in the actor's gradient path at all —
        # it only fabricates critic training data. J's honest flatness at
        # the frontier (the BPTT failure mode) cannot hurt a Q-ascent.
        s_seed = s_hist[0]
        a_pi = actor.deterministic_action(o_np_t)
        actor_loss = -sum(
            qcritics[n](torch.cat([s_seed, a_pi], dim=-1)).mean()
            for n in R)
    elif signal_floor > 0.0:
        sig = (1.0 - alive.detach())
        if phi0 is not None:
            sig = sig + (last_aux["phi"].detach() - phi0).abs()
        w_seed = sig.clamp(signal_floor, 1.0)
        actor_loss = -(w_seed * J).sum() / w_seed.sum()
    else:
        actor_loss = -J.mean()
    if entropy_coef != 0.0:
        actor_loss = actor_loss - entropy_coef * ent / H

    # Trust region on real seed states: the imagined objective rewards
    # whatever the *model* prefers — including its errors. Without an
    # anchor, 40 minibatch steps per update let the actor race off the
    # data manifold faster than the model/critic can track (a
    # teacher-initialized standing policy was destroyed within ~9
    # updates). Penalize the mean-action drift from the frozen policy
    # that generated this batch's data, at the real seed observations
    # where the anchor is actually grounded.
    kl_pen = None
    if ref_actor is not None and kl_coef > 0.0:
        with torch.no_grad():
            a_ref = ref_actor.deterministic_action(
                torch.as_tensor(o_np, device=device))
        a_now = actor.deterministic_action(torch.as_tensor(
            o_np, device=device))
        kl_pen = F.mse_loss(a_now, a_ref)
        actor_loss = actor_loss + kl_coef * kl_pen

    # Behavior anchor toward real high-return data: the learned model's
    # dJ/da carries no usable improvement signal near the boundary (measured:
    # proposals at ~0.01 action-RMS are worse than same-magnitude random
    # noise). Teacher-tagged windows are the one place where "which action
    # survives" is actually recorded — a light BC term pulls the actor back
    # onto the manifold where model gradients mean anything.
    bc_pen = None
    if bc_coef > 0.0:
        w_bc = buf.sample_windows(max(batch // 4, 8), 4,
                                  recent_frac=recent_frac,
                                  recent_pool=recent_pool,
                                  teacher_share=1.0)
        obs_bc = torch.as_tensor(w_bc["obs"], device=device)  # (b,4,obs)
        act_bc = torch.as_tensor(w_bc["act"], device=device)  # (b,4,act)
        a_bc = actor.deterministic_action(
            obs_bc.reshape(-1, obs_bc.shape[-1]))
        bc_pen = F.mse_loss(a_bc, act_bc.reshape(-1, act_bc.shape[-1]))
        actor_loss = actor_loss + bc_coef * bc_pen

    # DAgger-style anchor: label the states the LEARNER actually visits with
    # the teacher's actions. Plain BC only anchors the teacher's own state
    # distribution; the learner dies on its own deviation states where its
    # action is unanchored (compounding error). The labeler runs in-process
    # so any real buffer state can be tagged.
    dagger_pen = None
    if dagger_coef > 0.0 and dagger_net is not None:
        w_dg = buf.sample_windows(max(batch // 4, 8), 4,
                                  recent_frac=recent_frac,
                                  recent_pool=recent_pool,
                                  teacher_share=teacher_share)
        obs_dg = torch.as_tensor(w_dg["obs"], device=device)
        flat_obs = obs_dg.reshape(-1, obs_dg.shape[-1])
        with torch.no_grad():
            a_t = torch.tanh(dagger_net(flat_obs))
        a_l = actor.deterministic_action(flat_obs)
        err = (a_l - a_t).pow(2).mean(-1)  # (b*k,)
        if dagger_phi_min > 0.0 and "phi" in w_dg["aux"]:
            # The teacher's "label" on a deep-fall state is meaningless —
            # it never survives there, so its reflexes aren't valid
            # demonstrations. Restrict imitation to near-standing /
            # recoverable states where matching the teacher matters.
            phi_dg = torch.as_tensor(
                w_dg["aux"]["phi"], device=device).reshape(-1)
            mask = (phi_dg >= dagger_phi_min).float()
            dagger_pen = (err * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            dagger_pen = err.mean()
        actor_loss = actor_loss + dagger_coef * dagger_pen

    # Self-imitation on OWN-best trajectories (no external teacher): the
    # top-quantile learner runs are the only recorded demonstrations of the
    # full task being solved — they are lucky (pose+noise) but real. Cloning
    # their actions widens the set of recoverable poses; BPTT alone cannot
    # cross the flat J region to reach them.
    si_pen = None
    if si_coef > 0.0:
        w_si = buf.sample_top_windows(max(batch // 4, 8), 4,
                                      top_frac=si_top_frac)
        obs_si = torch.as_tensor(w_si["obs"], device=device)
        act_si = torch.as_tensor(w_si["act"], device=device)
        a_si = actor.deterministic_action(
            obs_si.reshape(-1, obs_si.shape[-1]))
        si_pen = F.mse_loss(a_si, act_si.reshape(-1, act_si.shape[-1]))
        actor_loss = actor_loss + si_coef * si_pen

    # --- Critic targets: backwards gamma-return on the imagined trajectory ---
    # Under --no-bootstrap this self-referential regression is skipped: its
    # targets bottom out at V(ŝ_H) on off-manifold states, which is exactly
    # the corruption channel this flag exists to remove.
    critic_loss = 0.0
    if not no_bootstrap:
        with torch.no_grad():
            targets = {}
            for ch in channels:
                name = ch["name"]
                g = ch["gamma"]
                G = _boot_v(name, s).detach()
                tgt = [None] * H
                for t in range(H - 1, -1, -1):
                    G = r_hist[name][t] + g * c_hist[t] * G
                    tgt[t] = G
                targets[name] = tgt

        for ch in channels:
            name = ch["name"]
            for t in range(H):
                critic_loss = critic_loss + F.mse_loss(
                    critics[name](s_hist[t]), targets[name][t]
                )
        critic_loss = critic_loss / (H * len(channels))

    # --- Critic anchoring on REAL trajectories ---
    # The imagined loss alone is terminal-blind under --no-done: V would
    # converge to the value of the *model* MDP where fallen states keep
    # earning reward. Regress V on real buffer windows with true reward and
    # true termination masking so the bootstrap V(s_H) in J carries the
    # "about to fall -> truncated return" information into imagination.
    for _c in critics.values():
        _set_requires_grad(_c, True)
    w = buf.sample_windows(batch, 8, recent_frac=recent_frac,
                           recent_pool=recent_pool,
                           teacher_share=teacher_share)
    cs_w = torch.as_tensor(np.concatenate(
        [w["s0"][:, None, :], w["cs"][:, :-1, :]], axis=1), device=device)
    # states trained: s_{j0}..s_{j0+k-1} = cs[j0-1] then cs[j0..j0+k-2]
    s_last = torch.as_tensor(w["cs"][:, -1, :], device=device)
    done_w = torch.as_tensor(w["done"], device=device)  # (b, k)
    real_loss = 0.0
    for ch in channels:
        name = ch["name"]
        g = ch["gamma"]
        with torch.no_grad():
            r_steps = torch.stack([
                exp.channel_reward(name, {
                    n: torch.as_tensor(v[:, m], device=device)
                    for n, v in w["aux"].items()
                })
                for m in range(w["done"].shape[1])
            ], dim=1)  # (b, k) real per-channel rewards
            boot = (1.0 - done_w[:, -1]) * _boot_v(name, s_last).detach()
            Gt = r_steps[:, -1] + g * boot
            gt = [None] * r_steps.shape[1]
            gt[-1] = Gt
            for m in range(r_steps.shape[1] - 2, -1, -1):
                Gt = r_steps[:, m] + g * (1.0 - done_w[:, m]) * Gt
                gt[m] = Gt
            gt = torch.stack(gt, dim=1)  # (b, k)
        real_loss = real_loss + F.mse_loss(critics[name](cs_w), gt)
    real_loss = real_loss / len(channels)
    critic_loss = critic_loss + real_loss

    # --- MBPO Q-critic training ---
    # Q(s,a) fitted on BOTH real transitions and imagined rollout
    # transitions (the model's data-augmentation job); actor ascends it.
    q_loss = 0.0
    if actor_mode == "mbpo" and qcritics:
        for _c in qcritics.values():
            _set_requires_grad(_c, True)
        act_w = torch.as_tensor(w["act"], device=device)  # (b,k,act)
        s0_w = torch.as_tensor(w["s0"], device=device)
        obs_w = torch.as_tensor(w["obs"], device=device)
        cs_full = torch.as_tensor(w["cs"], device=device)
        ref = ref_actor if ref_actor is not None else actor
        for ch in channels:
            name = ch["name"]
            g = ch["gamma"]
            # Real transitions: (cs_{j-1}, a_j) -> cs_j, r_j from real aux.
            with torch.no_grad():
                r_steps_q = torch.stack([
                    exp.channel_reward(name, {
                        n: torch.as_tensor(v[:, m], device=device)
                        for n, v in w["aux"].items()
                    })
                    for m in range(w["done"].shape[1])
                ], dim=1)
                a_next = ref.deterministic_action(
                    obs_w.reshape(-1, obs_w.shape[-1]))
                q_next = tgt_qcritics[name](torch.cat(
                    [cs_full.reshape(-1, cs_full.shape[-1]), a_next], -1)
                    ).reshape(done_w.shape)
                q_tgt_r = r_steps_q + g * (1.0 - done_w) * q_next
            s_prev = torch.cat(
                [s0_w[:, None, :], cs_full[:, :-1, :]], dim=1)
            q_in = torch.cat(
                [s_prev.reshape(-1, s_prev.shape[-1]),
                 act_w.reshape(-1, act_w.shape[-1])], -1)
            q_loss = q_loss + F.mse_loss(
                qcritics[name](q_in), q_tgt_r.reshape(-1))
            # Imagined transitions (member-0 rollout): (ŝ_t, â_t) -> ŝ_{t+1},
            # r̂_t from model aux, bootstrap on imagined next state.
            s_im = torch.stack([s_hist[t] for t in range(H)], dim=1)
            a_im = torch.stack([a_hist[t] for t in range(H)], dim=1)
            s_nxt = torch.stack(
                [s_hist[t + 1] if t + 1 < H else s_fin for t in range(H)],
                dim=1)
            o_nxt = torch.stack([o_hist[t] for t in range(H)], dim=1)
            al_nxt = torch.stack([c_hist[t] for t in range(H)], dim=1)
            r_im = torch.stack([r_hist[name][t] for t in range(H)], dim=1)
            with torch.no_grad():
                a_ni = ref.deterministic_action(
                    o_nxt.reshape(-1, o_nxt.shape[-1]))
                q_ni = tgt_qcritics[name](torch.cat(
                    [s_nxt.reshape(-1, s_nxt.shape[-1]), a_ni], -1)
                    ).reshape(r_im.shape)
                q_tgt_i = r_im + g * al_nxt * q_ni
            q_in_i = torch.cat(
                [s_im.reshape(-1, s_im.shape[-1]),
                 a_im.reshape(-1, a_im.shape[-1])], -1)
            q_loss = q_loss + F.mse_loss(
                qcritics[name](q_in_i), q_tgt_i.reshape(-1))
        critic_loss = critic_loss + q_loss / len(channels)

    stats = {
        "J": float(J.mean()),
        **vH_stats,
        "alive_end": float(alive.mean()),
        "kl_pen": float(kl_pen) if kl_pen is not None else 0.0,
        "bc_pen": float(bc_pen) if bc_pen is not None else 0.0,
        "dagger_pen": float(dagger_pen) if dagger_pen is not None else 0.0,
        "si_pen": float(si_pen) if si_pen is not None else 0.0,
        **{f"R_{n}": float(R[n].mean()) for n in R},
        "critic_loss": float(critic_loss),
        "real_closs": float(real_loss),
        "q_loss": float(q_loss) if qcritics else 0.0,
    }
    for _m in (wm if isinstance(wm, (list, tuple)) else [wm]):
        _set_requires_grad(_m, True)
    return actor_loss, critic_loss, stats


# ---------------------------------------------------------------------------
# Job building (self-play collection, both agents harvested)
# ---------------------------------------------------------------------------
def build_jobs(env_bp, policy_bp, base_seed, n_episodes, exp, stochastic,
               explore_factor):
    rng = np.random.default_rng(base_seed)
    spec = SamplingSpec(explore_factor=explore_factor)
    jobs = []
    for i in range(n_episodes):
        jobs.append(Job(
            policy_a_bp=policy_bp,
            policy_b_bp=policy_bp,
            env_bp=env_bp,
            seed=int(base_seed + i),
            episode_options={
                "initial_distance": float(rng.uniform(
                    exp.init_distance_min, exp.init_distance_max)),
            },
            sampling_a=spec,
            sampling_b=spec,
            stochastic=stochastic,
        ))
    return jobs


def eval_stats(episodes, exp):
    """Deterministic-eval metrics: per-agent survival rate and mean episode
    length (frames). ep_len is the fine-grained signal — a policy drifting
    toward the fall boundary loses length long before it loses survival."""
    lens = []
    survived = total = 0
    for ep in episodes:
        for aid in exp.agent_ids:
            total += 1
            if not ep.agent_termination_reason.get(
                    aid, "").startswith("imbalance"):
                survived += 1
        for tr in extract_trajs(ep, exp.agent_ids, exp.aux_keys):
            lens.append(tr.n_frames)
    pots_max, pots_final = [], []
    for ep in episodes:
        for tr in extract_trajs(ep, exp.agent_ids, exp.aux_keys):
            phi = tr.aux.get("phi")
            if phi is not None and len(phi):
                pots_max.append(float(np.max(phi)))
                pots_final.append(float(phi[-1]))
    return {
        "survival": survived / max(total, 1),
        "len": float(np.mean(lens)) if lens else 0.0,
        "max_pot": float(np.mean(pots_max)) if pots_max else 0.0,
        "final_pot": float(np.mean(pots_final)) if pots_final else 0.0,
    }


def eval_survival(episodes, agent_ids) -> float:
    survived = total = 0
    for ep in episodes:
        for aid in agent_ids:
            total += 1
            if not ep.agent_termination_reason.get(aid, "").startswith("imbalance"):
                survived += 1
    return survived / max(total, 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(exp, args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[dynabptt] experiment={exp.name} device={device}", flush=True)

    actor = exp.build_actor(device)
    if args.actor_init:
        # Warm-start the actor's mean net from an exported policy
        # checkpoint (e.g. the PPO basic_balance teacher). The exported
        # net is the same Linear-Tanh stack; its pre-tanh mean maps
        # directly onto actor.net. Answers the machinery check: if a
        # standing-init policy still collapses under DynaBPTT updates,
        # the training loop itself is broken — not the cold start.
        _p = torch.load(args.actor_init, map_location=device)
        _sd = _p.get("state_dict", _p)
        _missing, _unexpected = actor.net.load_state_dict(
            {k[4:]: v for k, v in _sd.items() if k.startswith("net.")},
            strict=False)
        if _missing or _unexpected:
            raise RuntimeError(
                f"--actor-init state mismatch missing={_missing} "
                f"unexpected={_unexpected}")
        print(f"[init] actor mean-net <- {args.actor_init}", flush=True)
    critics = {ch["name"]: CriticMLP(exp.state_dim, exp.critic_hidden_dim).to(device)
               for ch in exp.channels}
    # Slow-moving critic copies provide every bootstrap/target value. Without
    # them the real-window TD (V <- r + gamma V) self-excites: bootstrap
    # targets track the very weights being regressed and V inflates until it
    # dominates J. Hard-synced once per outer update.
    tgt_critics = {n: copy.deepcopy(c) for n, c in critics.items()}
    for _c in tgt_critics.values():
        _set_requires_grad(_c, False)
        _c.eval()
    # MBPO mode: action-conditioned Q(s,a) critics — the actor's ascent
    # surface. Same MLP body, input cat(s,a).
    qcritics = tgt_qcritics = None
    if args.actor_mode == "mbpo":
        qcritics = {ch["name"]: CriticMLP(
            exp.state_dim + exp.action_dim,
            exp.critic_hidden_dim).to(device)
            for ch in exp.channels}
        tgt_qcritics = {n: copy.deepcopy(c)
                        for n, c in qcritics.items()}
        for _c in tgt_qcritics.values():
            _set_requires_grad(_c, False)
            _c.eval()
    # Ensemble of independently-initialized dynamics models. Imagination
    # runs under every member on shared seeds and the actor maximizes a
    # pessimistic combination (mean - lambda*std) of per-member J — a
    # phantom gradient direction must fool all members to survive.
    wms = [WorldModel(
        state_dim=exp.state_dim, action_dim=exp.action_dim,
        obs_dim=exp.obs_dim, aux_names=exp.aux_names,
        hidden=exp.wm_hidden,
    ).to(device) for _ in range(max(1, args.wm_ensemble))]
    wm = wms[0] if len(wms) == 1 else wms

    # Frozen reference actor for the per-update trust region — synced to
    # the live actor at the start of each improve phase so the KL anchor
    # always points at the policy that generated the current buffer data.
    # With --anchor-mode init the reference is instead frozen forever at
    # the initialized policy (e.g. a BC/teacher warm start), making the
    # penalty an absolute rather than drifting anchor.
    ref_actor = copy.deepcopy(actor)
    _set_requires_grad(ref_actor, False)
    ref_actor.eval()

    dagger_net = None
    if args.dagger_coef > 0.0:
        if not args.dagger_teacher:
            raise SystemExit("--dagger-coef requires --dagger-teacher")
        _p = torch.load(args.dagger_teacher, map_location=device)
        _h = int(_p.get("hidden_dim", 256))
        dagger_net = nn.Sequential(
            nn.Linear(int(_p["obs_dim"]), _h), nn.Tanh(),
            nn.Linear(_h, _h), nn.Tanh(),
            nn.Linear(_h, int(_p["action_dim"]))).to(device)
        dagger_net.load_state_dict(
            {k[4:]: v for k, v in _p["state_dict"].items()
             if k.startswith("net.")})
        dagger_net.eval()
        _set_requires_grad(dagger_net, False)
        print(f"[init] dagger labeler <- {args.dagger_teacher}", flush=True)
    if args.sigma0 is not None:
        # Collection noise must be low enough that a good mean policy
        # actually survives — sigma=0.37 torque noise kills even a
        # standing teacher within ~20 frames, so the learner's own data
        # never reflects its true (deterministic) capability.
        with torch.no_grad():
            actor.log_std.fill_(float(np.log(args.sigma0)))

    opt_wms = [torch.optim.Adam(_m.parameters(), lr=args.wm_lr)
               for _m in wms]
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_params = [p for c in critics.values() for p in c.parameters()]
    if qcritics:
        critic_params += [p for c in qcritics.values()
                          for p in c.parameters()]
    opt_critic = torch.optim.Adam(critic_params, lr=args.critic_lr)

    # Ratchet state (see --ratchet): the best deterministic-eval actor seen
    # so far plus its score. The loop may propose whatever the model
    # rewards; only updates that do not regress REAL performance are kept.
    best_len = None
    best_actor = None

    env_bp = inject_corestate(exp.env_bp())
    env_bp_push = (inject_push(env_bp, args.push_force_min,
                               args.push_force_max, args.push_interval_min,
                               args.push_interval_max)
                   if args.push_frac > 0.0 else None)
    buf = TransitionBuffer(max_transitions=args.buffer_size, seed=args.seed)
    rollouter = ParallelRollouter(num_workers=args.workers)

    start_update = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        actor.load_state_dict(ck["actor"])
        if "wms" in ck:
            for _m, _sd in zip(wms, ck["wms"]):
                _m.load_state_dict(_sd)
        else:
            wms[0].load_state_dict(ck["wm"])
        for n, sd in ck["critics"].items():
            critics[n].load_state_dict(sd)
        start_update = int(ck["update"]) + 1
        print(f"[resume] from {args.resume} at update {start_update}", flush=True)

    export_root = Path(args.run_dir) / "policy_exports"
    ckpt_dir = Path(args.run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    try:
        seed_bp = (PolicyBlueprint.load(args.seed_policy)
                   if args.seed_policy else None)
        for u in range(start_update, args.updates):
            t0 = time.perf_counter()
            if u == start_update and args.seed_episodes > 0:
                # Cold-start coverage fix: pure exploration data is all
                # monotone falls, so the model can never imagine (or value)
                # recovery. Seeding the buffer with a standing-capable
                # policy's transitions gives imagination surviving futures
                # and gives the critic non-trivial values to anchor to.
                if env_bp_push is not None:
                    n_push = args.seed_episodes // 2
                    sjobs = (build_jobs(env_bp, seed_bp, args.seed + 55555,
                                        args.seed_episodes - n_push, exp,
                                        False, 0.0)
                             + build_jobs(env_bp_push, seed_bp,
                                          args.seed + 55556,
                                          n_push, exp, False, 0.0))
                else:
                    sjobs = build_jobs(env_bp, seed_bp, args.seed + 55555,
                                       args.seed_episodes, exp, False, 0.0)
                n_seed = 0
                for ep in rollouter.collect(sjobs):
                    for tr in extract_trajs(ep, exp.agent_ids, exp.aux_keys):
                        tr.source = "teacher"
                        buf.add(tr)
                        n_seed += tr.n_trans
                print(f"[seed] {args.seed_episodes} eps {args.seed_policy} "
                      f"-> +{n_seed} trans (buf={len(buf)})", flush=True)
            policy_bp = actor.to_blueprint(str(export_root / f"u{u:05d}"))
            # Keep a constant trickle of teacher data: a one-shot seed
            # decays to ~2% of the buffer within ~20 updates, after which
            # seeds/critic lose the healthy-standing anchor again.
            # --seed-frac of every collection batch is the teacher policy.
            n_teacher = (int(args.episodes * args.seed_frac)
                         if seed_bp is not None else 0)
            n_learner = args.episodes - n_teacher
            n_push_l = (int(n_learner * args.push_frac)
                        if env_bp_push is not None else 0)
            jobs = build_jobs(env_bp, policy_bp, args.seed + u * 100003,
                              n_learner - n_push_l, exp, True, args.ef)
            if n_push_l > 0:
                jobs = jobs + build_jobs(
                    env_bp_push, policy_bp, args.seed + u * 100003 + 7,
                    n_push_l, exp, True, args.ef)
            episodes = rollouter.collect(jobs)
            tr_added = 0
            for ep in episodes:
                for tr in extract_trajs(ep, exp.agent_ids, exp.aux_keys):
                    buf.add(tr)
                    tr_added += tr.n_trans
            if n_teacher > 0:
                n_push_t = (n_teacher // 2 if env_bp_push is not None else 0)
                tjobs = build_jobs(env_bp, seed_bp,
                                   args.seed + 77777 + u * 100003,
                                   n_teacher - n_push_t, exp, False, 0.0)
                if n_push_t > 0:
                    tjobs = tjobs + build_jobs(
                        env_bp_push, seed_bp,
                        args.seed + 77778 + u * 100003,
                        n_push_t, exp, False, 0.0)
                for ep in rollouter.collect(tjobs):
                    for tr in extract_trajs(
                            ep, exp.agent_ids, exp.aux_keys):
                        tr.source = "teacher"
                        buf.add(tr)
                        tr_added += tr.n_trans
            t_coll = time.perf_counter() - t0

            t0 = time.perf_counter()
            fit_batches = args.warmup_fit if u == start_update else args.fit_batches
            wm_stats_acc: Dict[str, float] = {}
            for _m, _o in zip(wms, opt_wms):
                _st = fit_wm(
                    _m, _o, buf, fit_batches, args.fit_batch_size,
                    args.k, device, exp.aux_names, exp.wm_loss_weights,
                    recent_frac=args.recent_frac,
                    recent_pool=args.recent_pool,
                    teacher_share=args.teacher_share)
                for _kk, _vv in _st.items():
                    wm_stats_acc[_kk] = wm_stats_acc.get(_kk, 0.0) + _vv
            wm_stats = {k: v / len(wms) for k, v in wm_stats_acc.items()}
            t_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            if args.anchor_mode == "last" and not args.ratchet:
                ref_actor.load_state_dict(actor.state_dict())
            imp_stats_acc: Dict[str, float] = {}
            train_actor = u >= args.actor_warmup
            for _bi in range(args.improve_batches):
                a_loss, c_loss, imp_stats = improve(
                    actor, critics, tgt_critics, wm, buf, exp,
                    args.improve_batch_size, args.horizon, device,
                    entropy_coef=args.entropy, use_done=not args.no_done,
                    seed_start_frac=args.seed_start_frac,
                    recent_frac=args.recent_frac,
                    recent_pool=args.recent_pool,
                    seed_tail_frac=args.seed_tail_frac,
                    seed_tail_span=args.seed_tail_span,
                    teacher_share=args.teacher_share,
                    ref_actor=ref_actor, kl_coef=args.actor_kl,
                    no_bootstrap=args.no_bootstrap,
                    signal_floor=args.signal_floor,
                    pess_lambda=args.pess_lambda,
                    bc_coef=args.bc_coef,
                    si_coef=args.si_coef, si_top_frac=args.si_top_frac,
                    actor_mode=args.actor_mode,
                    qcritics=qcritics, tgt_qcritics=tgt_qcritics,
                    dagger_net=dagger_net, dagger_coef=args.dagger_coef,
                    dagger_phi_min=args.dagger_phi_min,
                )
                if train_actor:
                    opt_actor.zero_grad(set_to_none=True)
                    a_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    opt_actor.step()
                opt_critic.zero_grad(set_to_none=True)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, 10.0)
                opt_critic.step()
                for kk, vv in imp_stats.items():
                    imp_stats_acc[kk] = imp_stats_acc.get(kk, 0.0) + vv
                # Per-update drift budget. The balance policy is a knife
                # edge: *random* weight perturbations producing only ~0.1
                # action-RMS collapse ep_len 544->~60, indistinguishable
                # from BPTT proposals of the same magnitude. Survival
                # therefore requires bounding the action-space drift per
                # update to the ~0.03 RMS regime — PPO-style target-KL
                # early stop does exactly that.
                if (args.target_kl > 0.0
                        and imp_stats.get("kl_pen", 0.0) > args.target_kl):
                    imp_stats_acc["kl_stop"] = float(_bi + 1)
                    break
            imp_stats = {kk: vv / max(_bi + 1, 1)
                         for kk, vv in imp_stats_acc.items()}
            # EMA target update — a hard sync every update lets TD errors
            # ratchet instantly into bootstrap targets (self-referential
            # value inflation); a slow target damps the feedback loop.
            with torch.no_grad():
                for _n, _c in critics.items():
                    for _pt, _ps in zip(tgt_critics[_n].parameters(),
                                        _c.parameters()):
                        _pt.lerp_(_ps, args.target_ema)
                if qcritics:
                    for _n, _c in qcritics.items():
                        for _pt, _ps in zip(
                                tgt_qcritics[_n].parameters(),
                                _c.parameters()):
                            _pt.lerp_(_ps, args.target_ema)
            sigma_mean = float(actor.policy_sigma().mean().detach().cpu())
            t_imp = time.perf_counter() - t0

            ratchet_str = ""
            if args.ratchet:
                # Per-update acceptance gate on REAL deterministic rollouts
                # (fixed seeds -> paired comparison). When the imagined
                # objective is flat or corrupt, actor updates are a
                # KL-bounded random walk that still breaks knife-edge
                # balance controllers — measured: teacher-init policy
                # destroyed within ~5 active updates while model J claimed
                # parity. Revert on regression; accept and re-anchor the
                # trust region otherwise.
                policy_bp = actor.to_blueprint(
                    str(export_root / f"u{u:05d}_ratchet"))
                rjobs = build_jobs(env_bp, policy_bp, args.seed + 31337,
                                   args.ratchet_episodes, exp, False, 0.0)
                m = eval_stats(rollouter.collect(rjobs), exp)
                score = m[args.ratchet_metric]
                if best_len is None or score >= best_len - args.ratchet_tol:
                    best_len = score if best_len is None else max(
                        best_len, score)
                    best_actor = {k: v.detach().clone()
                                  for k, v in actor.state_dict().items()}
                    ref_actor.load_state_dict(actor.state_dict())
                    ratchet_str = (
                        f" | ratchet {args.ratchet_metric}={score:.3f} "
                        f"best={best_len:.3f} acc")
                else:
                    actor.load_state_dict(best_actor)
                    opt_actor = torch.optim.Adam(actor.parameters(),
                                                 lr=args.actor_lr)
                    ratchet_str = (
                        f" | ratchet {args.ratchet_metric}={score:.3f} "
                        f"best={best_len:.3f} REJ")

            eval_str = ""
            if (u + 1) % args.eval_interval == 0 or u == start_update:
                policy_bp = actor.to_blueprint(str(export_root / f"u{u:05d}_eval"))
                ejobs = build_jobs(env_bp, policy_bp, args.seed + 999983 + u,
                                   args.eval_episodes, exp, False, 0.0)
                em = eval_stats(rollouter.collect(ejobs), exp)
                eval_str = (f" | eval_survival={em['survival']:.3f}"
                            f" eval_pot={em['max_pot']:.3f}")

            if (u + 1) % args.ckpt_interval == 0 or u + 1 == args.updates:
                torch.save({
                    "update": u,
                    "actor": actor.state_dict(),
                    "wm": wms[0].state_dict(),
                    "wms": [_m.state_dict() for _m in wms],
                    "critics": {n: c.state_dict() for n, c in critics.items()},
                }, ckpt_dir / f"ckpt_u{u:05d}.pt")

            r_str = " ".join(
                f"{n}={imp_stats.get(f'R_{n}', 0):.3f}"
                for n in (ch["name"] for ch in exp.channels))
            print(
                f"[u{u}] coll={t_coll:.1f}s fit={t_fit:.1f}s imp={t_imp:.1f}s "
                f"| buf={len(buf)} trajs={buf.n_trajs} (+{tr_added}) "
                f"| wm_s={wm_stats['s']:.4f} wm_obs={wm_stats['obs']:.4f} "
                f"wm_aux={wm_stats['aux']:.4f} wm_done={wm_stats['done']:.4f} "
                f"| J={imp_stats.get('J', 0):.3f} "
                + " ".join(
                    f"vH_{n}={imp_stats.get(f'vH_{n}', 0):.2f}"
                    for n in (ch["name"] for ch in exp.channels))
                + " "
                f"alive={imp_stats.get('alive_end', 0):.2f} "
                f"closs={imp_stats.get('critic_loss', 0):.4f} "
                f"rcloss={imp_stats.get('real_closs', 0):.4f} "
                f"kl={imp_stats.get('kl_pen', 0):.4f}"
                + (f" bc={imp_stats.get('bc_pen', 0):.4f}"
                   if args.bc_coef > 0.0 else "")
                + (f" dg={imp_stats.get('dagger_pen', 0):.4f}"
                   if args.dagger_coef > 0.0 else "")
                + (f" si={imp_stats.get('si_pen', 0):.4f}"
                   if args.si_coef > 0.0 else "")
                + (f" q={imp_stats.get('q_loss', 0):.4f}"
                   if args.actor_mode == "mbpo" else "")
                + (f" klstop={imp_stats['kl_stop']:.0f}"
                   if "kl_stop" in imp_stats else "")
                + f" {r_str} "
                f"| sigma={sigma_mean:.3f}{eval_str}{ratchet_str}",
                flush=True,
            )
    finally:
        rollouter.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_experiment(name: str):
    from baseline.experiments_dynabptt import get_dynabptt_experiment
    return get_dynabptt_experiment(name)


def main() -> None:
    p = argparse.ArgumentParser(description="DynaBPTT trainer (self-only v1)")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--ef", type=float, default=0.15)
    p.add_argument("--buffer-size", type=int, default=2_000_000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--fit-batches", type=int, default=120)
    p.add_argument("--warmup-fit", type=int, default=400)
    p.add_argument("--fit-batch-size", type=int, default=512)
    p.add_argument("--improve-batches", type=int, default=30)
    p.add_argument("--improve-batch-size", type=int, default=512)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--entropy", type=float, default=0.0)
    p.add_argument("--no-done", action="store_true",
                   help="Drop the learned termination head from the imagined "
                        "objective and critic targets (the head is still fit "
                        "for diagnostics). Surviving is then signalled "
                        "entirely through reward + critic bootstrap — useful "
                        "for cold-start balance tasks where early data is "
                        "all falls and a correct done head would zero out "
                        "the learning signal.")
    p.add_argument("--recent-frac", type=float, default=0.5,
                   help="Fraction of window/seed draws taken from the most "
                        "recent --recent-pool share of trajectories.")
    p.add_argument("--recent-pool", type=float, default=0.25)
    p.add_argument("--seed-tail-frac", type=float, default=0.0,
                   help="Fraction of imagination seeds drawn from the last "
                        "--seed-tail-span frames of a trajectory (puts the "
                        "terminal boundary inside the imagined horizon).")
    p.add_argument("--seed-tail-span", type=int, default=16)
    p.add_argument("--seed-start-frac", type=float, default=0.25,
                   help="Fraction of imagination seeds drawn from the first "
                        "frames of a trajectory (guaranteed-upright states) "
                        "so some seeds always retain survival gradient.")
    p.add_argument("--wm-lr", type=float, default=1e-3)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--target-ema", type=float, default=0.05,
                   help="Polyak coefficient for target critic update each "
                        "update (1.0 = hard sync every update).")
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--ckpt-interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Drop the V(s_H) bootstrap term from J. Imagined "
                        "states drift off the real-data manifold, so OOD "
                        "critic values are arbitrary — and the actor climbs "
                        "them (measured: J ranked a destroyed policy 2.7x "
                        "above the standing teacher purely via V(s_H)).")
    p.add_argument("--ratchet", action="store_true",
                   help="Gate every update on a fixed-seed deterministic eval "
                        "of mean episode length; revert the actor to the "
                        "best-scoring snapshot on regression and reset its "
                        "optimizer. Makes model-gradient corruption safe by "
                        "construction.")
    p.add_argument("--ratchet-metric", type=str, default="len",
                   choices=["len", "survival", "max_pot", "final_pot"],
                   help="eval_stats field the ratchet maximizes")
    p.add_argument("--ratchet-tol", type=float, default=30.0,
                   help="ep_len regression tolerance before reverting")
    p.add_argument("--ratchet-episodes", type=int, default=32,
                   help="episodes per ratchet eval (fixed seeds)")
    p.add_argument("--push-frac", type=float, default=0.0,
                   help="Fraction of learner episodes collected under random "
                        "torso pushes; generates tilt->(recover|die) arcs, "
                        "the counterfactual coverage the boundary gradient "
                        "needs. Eval stays push-free.")
    p.add_argument("--push-force-min", type=float, default=150.0)
    p.add_argument("--push-force-max", type=float, default=350.0)
    p.add_argument("--push-interval-min", type=int, default=40)
    p.add_argument("--push-interval-max", type=int, default=100)
    p.add_argument("--signal-floor", type=float, default=0.0,
                   help="If >0, weight each imagined seed's J by its "
                        "boundary signal (1-alive_end + |phi drift|), with "
                        "this value as the floor for flat/saturated seeds")
    p.add_argument("--wm-ensemble", type=int, default=1,
                   help="Number of independently-trained dynamics models; "
                        "imagination runs under every member and the actor "
                        "maximizes a pessimistic J combination")
    p.add_argument("--pess-lambda", type=float, default=0.0,
                   help="Pessimism coefficient: J = mean_k(J) - "
                        "lambda*std_k(J) across the ensemble")
    p.add_argument("--actor-mode", type=str, default="bptt",
                   choices=["bptt", "mbpo"],
                   help="bptt: actor ascends imagined return through "
                        "dynamics. mbpo: model only augments critic data; "
                        "actor ascends Q(s,pi(s)) — no dynamics in the "
                        "actor gradient path")
    p.add_argument("--si-coef", type=float, default=0.0,
                   help="weight of self-imitation on top-quantile "
                        "learner trajectories (0=off)")
    p.add_argument("--si-top-frac", type=float, default=0.1,
                   help="quantile of learner trajs ranked by max potential "
                        "that count as imitation demonstrations")
    p.add_argument("--dagger-coef", type=float, default=0.0,
                   help="If >0, DAgger term: teacher labels learner-visited "
                        "buffer states in-process; anchors the actor on its "
                        "own deviation distribution, not just teacher's")
    p.add_argument("--dagger-teacher", type=str, default=None,
                   help="model.pt of the labeling teacher (exported MLP)")
    p.add_argument("--dagger-phi-min", type=float, default=0.0,
                   help="If >0, only imitate states whose real phi exceeds "
                        "this — teacher labels on doomed deep-fall states "
                        "are meaningless")
    p.add_argument("--bc-coef", type=float, default=0.0,
                   help="If >0, add an MSE behavior-cloning term toward "
                        "teacher-tagged buffer windows — real-data anchor "
                        "for states where model gradients carry no signal")
    p.add_argument("--target-kl", type=float, default=0.0,
                   help="If >0, stop the improve loop once the mean-action "
                        "MSE vs the reference actor exceeds this bound "
                        "(0.001 ~= 0.03 action RMS drift)")
    p.add_argument("--actor-kl", type=float, default=3.0,
                   help="Trust-region coefficient: penalizes mean-action "
                        "drift from the policy that generated the current "
                        "buffer, at real seed observations.")
    p.add_argument("--anchor-mode", choices=["last", "init"],
                   default="last",
                   help="'last': KL anchor re-syncs to the actor each "
                        "update (drifting trust region). 'init': anchor "
                        "stays at the --actor-init policy permanently.")
    p.add_argument("--sigma0", type=float, default=None,
                   help="Override actor log_std init -> collection sigma. "
                        "Default keeps the policy's own init (~0.37).")
    p.add_argument("--actor-warmup", type=int, default=0,
                   help="Updates to skip actor training (model+critic "
                        "calibrate first so bootstrap values are sane).")
    p.add_argument("--actor-init", type=str, default=None,
                   help="Optional model.pt checkpoint to initialize the "
                        "actor's mean net (behavior-cloning warm start).")
    p.add_argument("--seed-frac", type=float, default=0.0,
                   help="Fraction of each update's collection episodes run "
                        "with --seed-policy (persistent teacher trickle).")
    p.add_argument("--teacher-share", type=float, default=0.0,
                   help="Share of each model-fit window batch and "
                        "imagination seed batch drawn from the teacher "
                        "trajectory pool (kept small: teacher trajectories "
                        "are ~10x longer and would otherwise dominate the "
                        "fit, letting the actor exploit model optimism).")
    p.add_argument("--seed-policy", type=str, default=None,
                   help="Optional policy blueprint path: collect "
                        "--seed-episodes with it at update 0 to warm-start "
                        "the buffer (coverage fix for cold-start balance).")
    p.add_argument("--seed-episodes", type=int, default=0)
    p.add_argument("--w-cross", type=float, default=None,
                   help="Override exp.w_cross (r_cross actor weight). Set 0 "
                        "for a pure-survival cold start — the stateful "
                        "cross-support reward's variance otherwise drowns "
                        "the tiny per-step phi reward in J.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--list-experiments", action="store_true")
    args = p.parse_args()

    if args.list_experiments:
        from baseline.experiments_dynabptt import list_dynabptt_experiments
        print("Available dynabptt experiments:", list_dynabptt_experiments())
        return

    if args.smoke:
        args.updates = 3
        args.episodes = 8
        args.workers = 8
        args.fit_batches = 20
        args.warmup_fit = 40
        args.improve_batches = 5
        args.eval_episodes = 4
        args.eval_interval = 1
        args.ckpt_interval = 3

    exp = _load_experiment(args.experiment)
    if args.w_cross is not None:
        exp.w_cross = float(args.w_cross)

    run_name = args.run_name or (
        f"train_{exp.name}_dynabptt_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        Path(__file__).resolve().parents[2] / "runs_dynabptt" / run_name)
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run_dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)

    with open(run_dir / "config.json", "w") as f:
        json.dump({**vars(args), "experiment_class": type(exp).__name__},
                  f, indent=2, default=str)

    run(exp, args)


if __name__ == "__main__":
    main()
