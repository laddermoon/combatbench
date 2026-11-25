"""Collect a fresh replay dataset at a chosen control decimation.

Used to test whether a finer control step unlocks model accuracy
(SUPERMIMIC_BOUNDARY.md, migration item T3).  Loads an actor from a
checkpoint, collects episodes through ParallelRollouter, extracts
aligned trajectories and pickles a replay tail that offline_calib can
consume via --replay.

Usage:
    PYTHONPATH=. python3 -m baseline.framework.dynabptt.hybrid_ppo.collect_replay \
        --ckpt .../ckpt_u00019.pt --phy-steps 10 --episodes 400 \
        --max-steps 500 --out /tmp/replay_dt20.pkl
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)
from baseline.framework.rollout.parallel_rollouter import ParallelRollouter

from .standup import ACTION_DIM, SELF_OBS_DIM, make_standup_env_bp
from .train import _build_jobs
from .transitions import ModelReplay, extract_aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phy-steps", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--export-dir", default="/tmp/hp_export")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, device=device)
    actor.load_state_dict(ck["actor"])
    policy_bp = actor.to_blueprint(args.export_dir)

    dt_ms = args.phy_steps * 2
    env_bp = make_standup_env_bp(max_steps=args.max_steps,
                                 phy_steps_per_action=args.phy_steps)
    rng = np.random.default_rng(args.seed)
    rollouter = ParallelRollouter(num_workers=args.workers)

    t0 = time.time()
    replay = ModelReplay()
    n_frames = 0
    for ep in rollouter.collect(
            _build_jobs(env_bp, policy_bp, args.seed,
                        args.episodes, rng, stochastic=True)):
        n_frames += ep.num_frames
        for tr in extract_aligned(ep, update=0):
            replay.add(tr)
    rollouter.close()

    payload = {"fit": list(replay.fit), "hold": list(replay.hold),
               "hold_seen": list(replay._hold_seen),
               "fit_trans": sum(t.T - 1 for t in replay.fit)}
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"collected {n_frames} frames at dt={dt_ms}ms "
          f"in {time.time()-t0:.0f}s -> {args.out} "
          f"({len(replay.fit)} fit trajs / {payload['fit_trans']} trans, "
          f"{len(replay.hold)} hold)")


if __name__ == "__main__":
    main()
