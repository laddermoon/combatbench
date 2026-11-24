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
           recent_frac=0.0, recent_pool=0.25):
    stats = {"s": 0.0, "obs": 0.0, "aux": 0.0, "done": 0.0, "total": 0.0}
    for _ in range(batches):
        w = buf.sample_windows(batch_size, k, recent_frac=recent_frac,
                               recent_pool=recent_pool)
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
            seed_tail_frac=0.0, seed_tail_span=16):
    """One imagination batch. Returns (stats dict, detached history) — the
    caller does opt steps so both actor and critic grads share the rollout."""
    _set_requires_grad(wm, False)
    for _c in critics.values():
        _set_requires_grad(_c, False)
    channels = exp.channels  # [{"name","gamma"}]
    aux_names = exp.aux_names

    s_np, o_np = buf.sample_seeds(batch, start_frac=seed_start_frac,
                                  recent_frac=recent_frac,
                                  recent_pool=recent_pool,
                                  tail_frac=seed_tail_frac,
                                  tail_span=seed_tail_span)
    s = torch.as_tensor(s_np, device=device)
    o = torch.as_tensor(o_np, device=device)
    alive = torch.ones(s.shape[0], device=device)

    J = torch.zeros(s.shape[0], device=device)
    R = {ch["name"]: torch.zeros(s.shape[0], device=device) for ch in channels}
    s_hist, c_hist, r_hist = [], [], {ch["name"]: [] for ch in channels}
    ent = 0.0

    for t in range(H):
        if entropy_coef != 0.0:
            ent = ent + actor.forward(o)[1].log().mean()
        # Deterministic actions inside imagination (SuperMimic recipe):
        # action noise lives in data collection only — it widens the model's
        # action coverage but must not perturb the imagined rollout.
        a = actor.deterministic_action(o)
        s_hist.append(s.detach())  # s_t — pairs with targets[t] below
        s, h = wm.step(s, a)
        o = h["obs"]
        # Aux heads are unbounded MLP outputs; imagined states drift
        # off-manifold and the heads extrapolate reward features to
        # absurd values (observed phi_hat ~30 -> phantom reward basin the
        # actor learns to chase while the real policy keeps falling).
        # Clamp predicted aux to each channel's plausible range before any
        # reward/weight computation — the model itself stays unclamped
        # (fit targets are real data; the clamp only removes the actor's
        # ability to arbitrage head extrapolation).
        aux = {}
        for name in aux_names:
            lo_hi = getattr(exp, "aux_clamps", {}).get(name)
            aux[name] = h[name].clamp(lo_hi[0], lo_hi[1]) if lo_hi else h[name]
        if use_done:
            alive = alive * (1.0 - torch.sigmoid(h["done_logit"]))
        for ch in channels:
            name = ch["name"]
            r_c = exp.channel_reward(name, aux)
            w_c = exp.channel_weight(name, aux)
            g_t = ch["gamma"] ** t
            R[name] = R[name] + g_t * alive * r_c
            J = J + g_t * alive * w_c * r_c
            r_hist[name].append(r_c.detach())
        c_hist.append(alive.detach())

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
        J = J + g_H * alive * exp.channel_weight(name, last_aux) * vH

    actor_loss = -J.mean()
    if entropy_coef != 0.0:
        actor_loss = actor_loss - entropy_coef * ent / H

    # --- Critic targets: backwards gamma-return on the imagined trajectory ---
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

    for _c in critics.values():
        _set_requires_grad(_c, True)
    critic_loss = 0.0
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
    w = buf.sample_windows(batch, 8, recent_frac=recent_frac,
                           recent_pool=recent_pool)
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

    stats = {
        "J": float(J.mean()),
        **vH_stats,
        "alive_end": float(alive.mean()),
        **{f"R_{n}": float(R[n].mean()) for n in R},
        "critic_loss": float(critic_loss),
        "real_closs": float(real_loss),
    }
    _set_requires_grad(wm, True)
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
    wm = WorldModel(
        state_dim=exp.state_dim, action_dim=exp.action_dim,
        obs_dim=exp.obs_dim, aux_names=exp.aux_names,
        hidden=exp.wm_hidden,
    ).to(device)

    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.wm_lr)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_params = [p for c in critics.values() for p in c.parameters()]
    opt_critic = torch.optim.Adam(critic_params, lr=args.critic_lr)

    env_bp = inject_corestate(exp.env_bp())
    buf = TransitionBuffer(max_transitions=args.buffer_size, seed=args.seed)
    rollouter = ParallelRollouter(num_workers=args.workers)

    start_update = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        actor.load_state_dict(ck["actor"])
        wm.load_state_dict(ck["wm"])
        for n, sd in ck["critics"].items():
            critics[n].load_state_dict(sd)
        start_update = int(ck["update"]) + 1
        print(f"[resume] from {args.resume} at update {start_update}", flush=True)

    export_root = Path(args.run_dir) / "policy_exports"
    ckpt_dir = Path(args.run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    try:
        for u in range(start_update, args.updates):
            t0 = time.perf_counter()
            if u == start_update and args.seed_episodes > 0:
                # Cold-start coverage fix: pure exploration data is all
                # monotone falls, so the model can never imagine (or value)
                # recovery. Seeding the buffer with a standing-capable
                # policy's transitions gives imagination surviving futures
                # and gives the critic non-trivial values to anchor to.
                seed_bp = PolicyBlueprint.load(args.seed_policy)
                sjobs = build_jobs(env_bp, seed_bp, args.seed + 55555,
                                   args.seed_episodes, exp, False, 0.0)
                n_seed = 0
                for ep in rollouter.collect(sjobs):
                    for tr in extract_trajs(ep, exp.agent_ids, exp.aux_keys):
                        buf.add(tr)
                        n_seed += tr.n_trans
                print(f"[seed] {args.seed_episodes} eps {args.seed_policy} "
                      f"-> +{n_seed} trans (buf={len(buf)})", flush=True)
            policy_bp = actor.to_blueprint(str(export_root / f"u{u:05d}"))
            jobs = build_jobs(env_bp, policy_bp, args.seed + u * 100003,
                              args.episodes, exp, True, args.ef)
            episodes = rollouter.collect(jobs)
            tr_added = 0
            for ep in episodes:
                for tr in extract_trajs(ep, exp.agent_ids, exp.aux_keys):
                    buf.add(tr)
                    tr_added += tr.n_trans
            t_coll = time.perf_counter() - t0

            t0 = time.perf_counter()
            fit_batches = args.warmup_fit if u == start_update else args.fit_batches
            wm_stats = fit_wm(
                wm, opt_wm, buf, fit_batches, args.fit_batch_size,
                args.k, device, exp.aux_names, exp.wm_loss_weights,
                recent_frac=args.recent_frac, recent_pool=args.recent_pool)
            t_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            imp_stats_acc: Dict[str, float] = {}
            for _ in range(args.improve_batches):
                a_loss, c_loss, imp_stats = improve(
                    actor, critics, tgt_critics, wm, buf, exp,
                    args.improve_batch_size, args.horizon, device,
                    entropy_coef=args.entropy, use_done=not args.no_done,
                    seed_start_frac=args.seed_start_frac,
                    recent_frac=args.recent_frac,
                    recent_pool=args.recent_pool,
                    seed_tail_frac=args.seed_tail_frac,
                    seed_tail_span=args.seed_tail_span,
                )
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
            imp_stats = {kk: vv / args.improve_batches
                         for kk, vv in imp_stats_acc.items()}
            # EMA target update — a hard sync every update lets TD errors
            # ratchet instantly into bootstrap targets (self-referential
            # value inflation); a slow target damps the feedback loop.
            with torch.no_grad():
                for _n, _c in critics.items():
                    for _pt, _ps in zip(tgt_critics[_n].parameters(),
                                        _c.parameters()):
                        _pt.lerp_(_ps, args.target_ema)
            sigma_mean = float(actor.policy_sigma().mean().detach().cpu())
            t_imp = time.perf_counter() - t0

            eval_str = ""
            if (u + 1) % args.eval_interval == 0 or u == start_update:
                policy_bp = actor.to_blueprint(str(export_root / f"u{u:05d}_eval"))
                ejobs = build_jobs(env_bp, policy_bp, args.seed + 999983 + u,
                                   args.eval_episodes, exp, False, 0.0)
                surv = eval_survival(rollouter.collect(ejobs), exp.agent_ids)
                eval_str = f" | eval_survival={surv:.3f}"

            if (u + 1) % args.ckpt_interval == 0 or u + 1 == args.updates:
                torch.save({
                    "update": u,
                    "actor": actor.state_dict(),
                    "wm": wm.state_dict(),
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
                f"rcloss={imp_stats.get('real_closs', 0):.4f} {r_str} "
                f"| sigma={sigma_mean:.3f}{eval_str}",
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
