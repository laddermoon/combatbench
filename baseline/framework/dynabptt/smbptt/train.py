"""SuperMimic-faithful BPTT trainer (self-only standing balance).

Per update u (mirrors HybridRobotics/supermimic runner):
  1. collect real episodes with the stochastic policy -> replay
  2. fit world model on k=8 autoregressive windows
  3. BPTT: imagine H steps from replay seeds with deterministic actions,
     -J backprop into the actor; fit rep-critic on the same rollout
  4. periodic deterministic eval; checkpoint + metrics.jsonl

Usage (from repo root):
  PYTHONPATH=. python3 -B -m baseline.framework.dynabptt.smbptt.train \
      --run-name sm_balance_v1 --phy-steps 10 --max-steps 1500
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from baseline.framework.critic_mlp import CriticMLP  # noqa: F401 (unused)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)
from baseline.framework.rollout.episode import blueprint_hash
from baseline.framework.rollout.job import Job, SamplingSpec
from baseline.framework.rollout.parallel_rollouter import ParallelRollouter

from ..hybrid_ppo.model import (RepCritic, SelfWorldModel, fit_model)
from ..hybrid_ppo.standup import ACTION_DIM, SELF_OBS_DIM, STATE_DIM
from ..hybrid_ppo.transitions import ModelReplay
from .imagine import bptt_update
from .task import extract_balance, make_balance_env_bp

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _jsonl(path: Path, record: Dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def _build_jobs(env_bp, policy_bp, base_seed, n_episodes, rng,
                stochastic, init_distance_min=1.5, init_distance_max=3.5):
    spec = SamplingSpec(explore_factor=0.0)
    return [Job(
        policy_a_bp=policy_bp, policy_b_bp=policy_bp,
        env_bp=env_bp,
        seed=int(base_seed + i),
        episode_options={
            "initial_distance": float(
                rng.uniform(init_distance_min, init_distance_max)),
        },
        sampling_a=spec, sampling_b=spec,
        stochastic=stochastic,
    ) for i in range(n_episodes)]


def aggregate_balance_metrics(episodes) -> Dict[str, float]:
    """Balance eval: survival fraction + episode length + phi-free stats."""
    n = 0
    timeout = 0
    frames = 0
    for ep in episodes:
        for aid in ("robot_a", "robot_b"):
            recs = ep.agent_termination_proposal_records.get(aid, ())
            n += 1
            frames += int(ep.agent_frame_boundary[aid])
            if not recs or recs[0][0] == "timeout":
                timeout += 1
    agents = max(n, 1)
    return {
        "a_survival": timeout / agents,
        "a_mean_len": frames / agents,
        "n_eps": float(len(episodes)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phy-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--episodes-per-update", type=int, default=128)
    p.add_argument("--workers", type=int, default=120)
    p.add_argument("--max-updates", type=int, default=10000)
    p.add_argument("--time-budget-min", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    # model
    p.add_argument("--model-hidden", type=int, nargs="+",
                   default=[1024, 1024, 1024])
    p.add_argument("--model-batches", type=int, default=60)
    p.add_argument("--model-k", type=int, default=8)
    # policy improvement
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--seeds", type=int, default=1024)
    p.add_argument("--imag-per-update", type=int, default=1,
                   help="imagined batches per update")
    p.add_argument("--actor-lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    # eval
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--resume-from", type=str, default=None)
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    rng_m = np.random.default_rng(args.seed + 1_000_003)
    torch.manual_seed(args.seed)

    env_bp = make_balance_env_bp(
        max_steps=args.max_steps, phy_steps_per_action=args.phy_steps)
    bp_hash = blueprint_hash(env_bp)

    dt = args.phy_steps * 0.002
    sub_coeff = (args.phy_steps + 1) / (2 * args.phy_steps)
    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, device=str(device))
    model = SelfWorldModel(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, obs_dim=SELF_OBS_DIM,
        hidden=tuple(args.model_hidden), dt=dt,
        substep_coeff=sub_coeff).to(device)
    vrep = RepCritic().to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.actor_lr,
                             betas=(0.9, 0.95))
    opt_m = torch.optim.Adam(model.parameters(), lr=2e-4)
    opt_v = torch.optim.Adam(vrep.parameters(), lr=2e-4)

    from ..hybrid_ppo.train import ModelConfig
    mcfg = ModelConfig(
        model_fit_batches=args.model_batches, model_k=args.model_k,
        model_tail_frac=0.35)
    replay = ModelReplay()
    rollouter = ParallelRollouter(num_workers=args.workers)
    import atexit, signal
    atexit.register(rollouter.close)
    def _sigterm(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    start_u = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device,
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        model.load_state_dict(ck["model"])
        vrep.load_state_dict(ck["vrep"])
        opt_a.load_state_dict(ck["opt_a"])
        opt_m.load_state_dict(ck["opt_m"])
        opt_v.load_state_dict(ck["opt_v"])
        rp = Path(str(args.resume_from).replace(".pt", ".replay.pkl"))
        if rp.exists():
            data = pickle.load(open(rp, "rb"))
            replay.fit = list(data["fit"])
            replay.hold = list(data["hold"])
            replay._hold_seen = set(data["hold_seen"])
            replay._fit_trans = int(data["fit_trans"])
        start_u = int(ck["update"]) + 1
        rng = np.random.default_rng()
        rng.bit_generator.state = pickle.loads(ck["np_rng"])

    metrics_path = run_dir / "metrics.jsonl"
    t_start = time.time()
    export_dir = run_dir / "policy_export"
    export_dir.mkdir(exist_ok=True)
    _jsonl(metrics_path, {
        "type": "header", "run": args.run_name, "seed": args.seed,
        "phy_steps": args.phy_steps, "max_steps": args.max_steps,
        "horizon": args.horizon, "seeds": args.seeds,
        "blueprint": bp_hash,
        "git": os.popen("git rev-parse HEAD").read().strip(),
    })

    best_survival = -1.0
    for u in range(start_u, args.max_updates):
        t0 = time.time()
        rec: Dict = {"type": "update", "u": u}

        # --- 1. collect real episodes ----------------------------------
        policy_bp = actor.to_blueprint(str(export_dir))
        jobs = _build_jobs(env_bp, policy_bp, args.seed + u * 100003,
                           args.episodes_per_update, rng, stochastic=True)
        episodes = rollouter.collect(jobs)
        n_fit = n_hold = 0
        for ep in episodes:
            for tr in extract_balance(ep, update=u):
                if replay.add(tr) == "fit":
                    n_fit += 1
                else:
                    n_hold += 1
        rec["episodes"] = len(episodes)
        rec["trajs"] = n_fit + n_hold
        rec["replay_trans"] = len(replay)
        rec.update({f"train_{k}": v
                    for k, v in aggregate_balance_metrics(episodes).items()})

        # --- 2. fit world model ----------------------------------------
        t_m = time.time()
        nw = replay.sample_windows(1024, mcfg.model_k, rng_m,
                                   cur_update=u)
        model.fit_normalization(nw)
        with torch.no_grad():
            _cs = torch.as_tensor(
                np.concatenate([nw["s0"], nw["cs"].reshape(-1, 55)], 0),
                device=device)
            _rep, _aux = SelfWorldModel.rep_aux_from_cs(_cs)
            _x = torch.cat([_rep, _aux], -1)
            vrep.x_mean.copy_(_x.mean(0))
            vrep.x_std.copy_(_x.std(0) + 1e-6)
        fstats = fit_model(model, opt_m, replay, mcfg, rng_m, u, device)
        rec.update({f"wm_{k}": v for k, v in fstats.items()})
        rec["t_model"] = time.time() - t_m

        # --- 3. BPTT policy improvement ---------------------------------
        t_p = time.time()
        for _ in range(args.imag_per_update):
            seeds = replay.sample_seeds(args.seeds, rng_m, pool="fit",
                                        tail_frac=0.3, tail_region=40)
            if len(seeds) < 64:
                rec["bptt"] = "skipped:few-seeds"
                break
            cs0 = torch.as_tensor(
                np.stack([tr.cs[t - 1] for tr, t in seeds]),
                device=device)
            o0 = torch.as_tensor(
                np.stack([tr.obs[t] for tr, t in seeds]),
                device=device)
            bstats = bptt_update(
                model, actor, vrep, opt_a, opt_v, cs0, o0,
                horizon=args.horizon, gamma=args.gamma, lam=args.lam)
            rec.update({f"bptt_{k}": v for k, v in bstats.items()})
        rec["t_bptt"] = time.time() - t_p

        # --- 4. eval ----------------------------------------------------
        if (u + 1) % args.eval_interval == 0 or u == start_u:
            t_ev = time.time()
            ejobs = _build_jobs(
                env_bp, policy_bp,
                args.seed + 999983 + u, args.eval_episodes,
                rng, stochastic=False)
            eval_eps = rollouter.collect(ejobs)
            em = aggregate_balance_metrics(eval_eps)
            rec.update({f"eval_{k}": v for k, v in em.items()})
            rec["t_eval"] = time.time() - t_ev
            if em["a_survival"] > best_survival:
                best_survival = em["a_survival"]
                torch.save(actor.state_dict(), run_dir / "best_actor.pt")
                rec["new_best"] = 1.0

        rec["t_total"] = time.time() - t0
        rec["elapsed_min"] = (time.time() - t_start) / 60.0
        _jsonl(metrics_path, rec)
        print(f"[u{u:04d}] trajs={rec['trajs']} "
              f"wm_s={fstats.get('s', 0):.3f} "
              f"J={rec.get('bptt_J', float('nan')):.4f} "
              f"alive={rec.get('bptt_alive_end', float('nan')):.3f} "
              f"surv={rec.get('train_a_survival', float('nan')):.2f}/"
              f"{rec.get('eval_a_survival', float('nan')):.2f} "
              f"t={rec['t_total']:.1f}s", flush=True)

        # --- 5. checkpoint ----------------------------------------------
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "update": u, "actor": actor.state_dict(),
            "model": model.state_dict(), "vrep": vrep.state_dict(),
            "opt_a": opt_a.state_dict(), "opt_m": opt_m.state_dict(),
            "opt_v": opt_v.state_dict(),
            "np_rng": pickle.dumps(rng.bit_generator.state),
        }, ckpt_dir / f"ckpt_u{u:05d}.pt")
        if u % 5 == 0:
            tail = [t for t in replay.fit if t.update >= u - 3]
            with open(ckpt_dir / f"ckpt_u{u:05d}.replay.pkl", "wb") as f:
                pickle.dump({"fit": tail, "hold": list(replay.hold),
                             "hold_seen": list(replay._hold_seen),
                             "fit_trans": sum(t.T - 1 for t in tail)}, f)

        if (args.time_budget_min > 0
                and (time.time() - t_start) / 60.0 > args.time_budget_min):
            print(f"[budget] {args.time_budget_min}min reached", flush=True)
            break

    rollouter.close()


if __name__ == "__main__":
    main()
