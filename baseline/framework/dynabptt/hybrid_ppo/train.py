"""Hybrid PPO + gated world-model trainer (self-only Standup).

Per update u:
  1. freeze pi_old (export blueprint), collect real episodes
  2. align trajectories; all real data feeds PPO, model replay splits
     fit/holdout by episode hash
  3. [sidecar|gated] fit world model on fit pool, validate on holdout,
     update admission gate
  4. [gated + admitted] generate short model branches -> aux PPO batch
  5. PPO update (real always; model term weighted by alpha)
  6. periodic deterministic eval; checkpoint + metrics.jsonl

Modes:
  ppo      — no model (arm A, local baseline)
  sidecar  — model trains + validates, never updates the policy (arm B)
  gated    — model aux PPO active only when the gate admits (arm C)

Usage (from repo root):
  PYTHONPATH=. python3 -B -m baseline.framework.dynabptt.hybrid_ppo.train \
      --mode sidecar --run-name hp_p1_sidecar --episodes-per-update 512
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)
from baseline.framework.rollout.episode import blueprint_hash
from baseline.framework.rollout.job import Job, SamplingSpec
from baseline.framework.rollout.parallel_rollouter import ParallelRollouter

from .model import (RepCritic, SelfWorldModel, fit_model,
                    fit_rep_critic)
from .ppo import PPOConfig, build_ppo_batch, ppo_update
from .rollout import gen_model_batch, normalize_model_adv
from .standup import (ACTION_DIM, SELF_OBS_DIM, STATE_DIM,
                      aggregate_metrics, make_standup_env_bp)
from .transitions import ModelReplay, extract_aligned
from .validation import GateState, validate_model

RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass
class ModelConfig:
    hidden: tuple = (512, 512, 512)
    lr: float = 3e-4
    model_fit_batches: int = 60
    model_batch: int = 256
    model_k: int = 3
    model_recent_frac: float = 0.8
    model_recent_updates: int = 2
    model_tail_frac: float = 0.0
    w_state: float = 1.0
    w_obs: float = 1.0
    w_phi: float = 1.0
    w_stage: float = 0.5
    val_windows: int = 4096
    paired_seeds: int = 64
    paired_h: int = 3
    max_seeds_frac: float = 0.25   # of real frames this update
    max_seeds_abs: int = 51200


def _jsonl(path: Path, record: Dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def _build_jobs(env_bp, policy_bp, base_seed, n_episodes, rng,
                stochastic, init_distance_min=1.5, init_distance_max=3.5):
    spec = SamplingSpec(explore_factor=0.0)
    jobs = []
    for i in range(n_episodes):
        jobs.append(Job(
            policy_a_bp=policy_bp, policy_b_bp=policy_bp,
            env_bp=env_bp,
            seed=int(base_seed + i),
            episode_options={
                "initial_distance": float(
                    rng.uniform(init_distance_min, init_distance_max)),
            },
            sampling_a=spec, sampling_b=spec,
            stochastic=stochastic,
        ))
    return jobs


def save_checkpoint(path: Path, update: int, actor, critic, model, vrep,
                    opt_a, opt_c, opt_m, opt_v, gate: GateState, replay,
                    rng: np.random.Generator, torch_rng_state,
                    save_replay: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": update,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "model": model.state_dict(),
        "vrep": vrep.state_dict(),
        "opt_a": opt_a.state_dict(),
        "opt_c": opt_c.state_dict(),
        "opt_m": opt_m.state_dict(),
        "opt_v": opt_v.state_dict(),
        "gate": gate.state_dict(),
        "np_rng": pickle.dumps(rng.bit_generator.state),
        "torch_rng": torch_rng_state,
        "torch_cuda_rng": (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
    }, path)
    if not save_replay:
        return
    # model replay tail: only the most recent trajs (what recency-biased
    # fitting mostly consumes) — keeps resume files small.
    tail = [t for t in replay.fit if t.update >= update - 3]
    tail += list(replay.hold)
    tmp = path.with_suffix(".replay.tmp")
    with tmp.open("wb") as f:
        pickle.dump({"fit": tail, "hold": list(replay.hold),
                     "hold_seen": list(replay._hold_seen),
                     "fit_trans": sum(t.T - 1 for t in tail)}, f)
    tmp.replace(path.with_suffix(".replay.pkl"))


def _replay_save_due(update: int, mode: str, gate: GateState) -> bool:
    """Replay tail is heavy (~300MB) — persist it only where it matters:
    resume-capable model runs, every 5 updates or once the gate opens."""
    if mode == "ppo":
        return False
    return update % 5 == 0 or gate.admitted or gate.assist_admitted


def load_checkpoint(path: Path, actor, critic, model, vrep, opt_a, opt_c,
                    opt_m, opt_v, gate: GateState, replay,
                    device: torch.device) -> int:
    ck = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    if "model" in ck:
        try:
            model.load_state_dict(ck["model"])
        except (RuntimeError, KeyError):
            pass  # arch changed (GenCoord port) — refit from scratch
    if "vrep" in ck:
        vrep.load_state_dict(ck["vrep"])
    for opt, key in ((opt_a, "opt_a"), (opt_c, "opt_c"),
                     (opt_m, "opt_m"), (opt_v, "opt_v")):
        if key in ck:
            try:
                opt.load_state_dict(ck[key])
            except ValueError:
                pass  # param groups changed — keep weights, reset Adam state
    gate.load_state_dict(ck["gate"])
    rp = path.with_suffix(".replay.pkl")
    if rp.exists():
        with rp.open("rb") as f:
            data = pickle.load(f)
        replay.fit = list(data["fit"])
        replay.hold = list(data["hold"])
        replay._hold_seen = set(data["hold_seen"])
        replay._fit_trans = int(data["fit_trans"])
    rng = np.random.default_rng()
    rng.bit_generator.state = pickle.loads(ck["np_rng"])
    tr = ck["torch_rng"]
    torch.set_rng_state(tr.cpu() if isinstance(tr, torch.Tensor) else tr)
    if ck.get("torch_cuda_rng") is not None and torch.cuda.is_available():
        cr = ck["torch_cuda_rng"]
        torch.cuda.set_rng_state(
            cr.cpu() if isinstance(cr, torch.Tensor) else cr)
    return int(ck["update"]), rng


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ppo", "sidecar", "gated"],
                   required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--episodes-per-update", type=int, default=512)
    p.add_argument("--max-updates", type=int, default=10000)
    p.add_argument("--time-budget-min", type=float, default=0.0,
                   help="0 = unlimited")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--phy-steps", type=int, default=25,
                   help="physics substeps per action (25=50ms, 10=20ms)")
    p.add_argument("--model-hidden", type=int, nargs="+",
                   default=[512, 512, 512])
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--fork-alpha", type=float, default=None,
                   help="override --alpha on resume (P2 fork)")
    p.add_argument("--kl-window", type=int, default=10)
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # separate streams: rng drives job options + ppo shuffle; rng_m is
    # owned by everything model-side so sidecar work can never perturb
    # the collection/update stream shared with the ppo arm.
    rng = np.random.default_rng(args.seed)
    rng_m = np.random.default_rng(args.seed + 1_000_003)
    torch.manual_seed(args.seed)

    env_bp = make_standup_env_bp(
        max_steps=args.max_steps, phy_steps_per_action=args.phy_steps)
    bp_hash = blueprint_hash(env_bp)

    dt = args.phy_steps * 0.002
    sub_coeff = (args.phy_steps + 1) / (2 * args.phy_steps)
    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, device=str(device))
    critic = CriticMLP(obs_dim=SELF_OBS_DIM, hidden_dim=256).to(device)
    model = SelfWorldModel(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, obs_dim=SELF_OBS_DIM,
        hidden=tuple(args.model_hidden), dt=dt,
        substep_coeff=sub_coeff).to(device)
    vrep = RepCritic().to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_m = torch.optim.Adam(model.parameters(), lr=3e-4)
    opt_v = torch.optim.Adam(vrep.parameters(), lr=3e-4)

    ppo_cfg = PPOConfig(alpha=args.alpha, kl_window=args.kl_window)
    mcfg = ModelConfig()
    replay = ModelReplay()
    gate = GateState()
    rollouter = ParallelRollouter(num_workers=args.workers)
    import atexit, signal
    atexit.register(rollouter.close)
    def _sigterm(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    start_u = 0
    if args.resume_from:
        last_u, rng = load_checkpoint(
            Path(args.resume_from), actor, critic, model, vrep,
            opt_a, opt_c, opt_m, opt_v, gate, replay, device)
        start_u = last_u + 1
    if args.fork_alpha is not None:
        ppo_cfg.alpha = args.fork_alpha

    metrics_path = run_dir / "metrics.jsonl"
    t_start = time.time()
    export_dir = run_dir / "policy_export"
    export_dir.mkdir(exist_ok=True)

    header = {
        "run": args.run_name, "mode": args.mode, "seed": args.seed,
        "alpha": ppo_cfg.alpha, "episodes_per_update":
        args.episodes_per_update, "blueprint": bp_hash,
        "git": os.popen("git rev-parse HEAD").read().strip(),
    }
    _jsonl(metrics_path, {"type": "header", **header})

    for u in range(start_u, args.max_updates):
        t0 = time.time()
        rec: Dict = {"type": "update", "u": u, "mode": args.mode}

        # --- 1. collect real episodes with pi_old ---
        policy_bp = actor.to_blueprint(str(export_dir))
        jobs = _build_jobs(env_bp, policy_bp, args.seed + u * 100003,
                           args.episodes_per_update, rng, stochastic=True)
        episodes = rollouter.collect(jobs)
        n_eps = len(episodes)
        trajs = []
        for ep in episodes:
            trajs.extend(extract_aligned(ep, update=u))
        n_fit = n_hold = 0
        for tr in trajs:
            if replay.add(tr) == "fit":
                n_fit += 1
            else:
                n_hold += 1
        rec["episodes"] = n_eps
        rec["traj_fit"] = n_fit
        rec["traj_hold"] = n_hold
        rec["trajs"] = len(trajs)
        rec["replay_trans"] = len(replay)
        rec["hold_eps"] = len(replay.hold)

        # --- 2. real PPO batch (all trajectories) ---
        batch = build_ppo_batch(trajs, critic, device,
                                ppo_cfg.gamma, ppo_cfg.lam)
        rec["n_frames"] = int(batch.obs.shape[0])
        rec["phi_max_train"] = float(
            np.max([tr.phi.max() for tr in trajs]))
        rec["phi_mean_end"] = float(
            np.mean([tr.phi[-1] for tr in trajs]))

        # --- 3. world model sidecar / gate ---
        t_model = time.time()
        if args.mode != "ppo":
            # refresh normalization from a fresh sample
            nw = replay.sample_windows(1024, mcfg.model_k, rng_m,
                                       cur_update=u)
            model.fit_normalization(nw)
            fit_stats = fit_model(model, opt_m, replay, mcfg, rng_m, u,
                                  device)
            rec.update({f"wm_{k}": v for k, v in fit_stats.items()})
            rec["vrep_loss"] = fit_rep_critic(
                vrep, opt_v, replay, model, rng_m, device)
            report = validate_model(
                model, actor, critic, replay, rng_m, device,
                n_windows=mcfg.val_windows, env_bp=env_bp,
                paired_seeds=mcfg.paired_seeds, paired_h=mcfg.paired_h,
                vrep=vrep)
            gate.update(report)
            rec.update(report.flat())
            if report.notes:
                rec["gate_notes"] = report.notes[:8]
        rec["t_model"] = time.time() - t_model

        # --- 4. model aux batch (gated only) ---
        model_batch = None
        t_branch = time.time()
        if (args.mode == "gated" and (gate.admitted or gate.assist_admitted)
                and ppo_cfg.alpha > 0.0):
            n_seeds = min(int(mcfg.max_seeds_frac * rec["n_frames"]),
                          mcfg.max_seeds_abs)
            seed_stages = set(gate.admitted_stages or gate.assist_stages)
            horizon = gate.horizon or gate.assist_horizon
            seeds = replay.sample_seeds(
                n_seeds, rng_m, stages=seed_stages, pool="fit")
            model_batch = gen_model_batch(
                model, actor, critic, seeds, horizon, device,
                ppo_cfg.gamma, ppo_cfg.lam, vrep=vrep)
            if model_batch is not None:
                rec["model_adv_std_raw"] = float(
                    np.asarray(model_batch["adv"]).std())
                model_batch["adv"] = normalize_model_adv(
                    model_batch["adv"], float(batch.adv.std()))
                rec["model_seeds"] = int(len(model_batch["adv"]))
                rec["model_phi_path_mean"] = float(
                    np.asarray(model_batch["phi_path"]).mean())
        rec["t_branch"] = time.time() - t_branch

        # --- 5. PPO update ---
        t_ppo = time.time()
        effective_alpha = (ppo_cfg.alpha
                           if model_batch is not None else 0.0)
        cfg_u = dataclasses.replace(ppo_cfg, alpha=effective_alpha)
        stats = ppo_update(actor, critic, opt_a, opt_c, batch, cfg_u,
                           device, model_batch=model_batch, rng=rng)
        rec.update(stats)
        rec["t_ppo"] = time.time() - t_ppo
        rec["alpha_eff"] = effective_alpha
        rec["gate_open"] = float(gate.admitted)

        # --- 6. eval ---
        if (u + 1) % args.eval_interval == 0 or u == start_u:
            t_ev = time.time()
            ejobs = _build_jobs(
                env_bp, policy_bp,
                args.seed + 999983 + u, args.eval_episodes,
                rng, stochastic=False)
            eval_eps = rollouter.collect(ejobs)
            rec.update({f"eval_{k}": v
                        for k, v in aggregate_metrics(eval_eps).items()})
            rec["t_eval"] = time.time() - t_ev

        rec["t_total"] = time.time() - t0
        rec["elapsed_min"] = (time.time() - t_start) / 60.0
        _jsonl(metrics_path, rec)
        print(f"[u{u:04d}] frames={rec['n_frames']} "
              f"kl={stats['kl_mean']:.4f} "
              f"ev={stats['ev']:.3f} "
              f"eval_max={rec.get('eval_a_max_pot', float('nan')):.3f}/"
              f"{rec.get('eval_b_max_pot', float('nan')):.3f} "
              f"gate={gate.horizon}/{gate.admitted_stages} "
              f"assist={gate.assist_horizon}/{gate.assist_stages} "
              f"t={rec['t_total']:.1f}s", flush=True)

        # --- 7. checkpoint (every update; replay tail per _replay_save_due)
        save_checkpoint(run_dir / "checkpoints" / f"ckpt_u{u:05d}.pt",
                        u, actor, critic, model, vrep,
                        opt_a, opt_c, opt_m, opt_v,
                        gate, replay, rng,
                        torch.get_rng_state(),
                        save_replay=_replay_save_due(u, args.mode, gate))

        if (args.time_budget_min > 0
                and (time.time() - t_start) / 60.0 > args.time_budget_min):
            print(f"[budget] {args.time_budget_min}min reached", flush=True)
            break

    rollouter.close()


if __name__ == "__main__":
    main()
