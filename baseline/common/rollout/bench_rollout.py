"""Benchmark rollout speed: random TanhGaussianMLPPolicy + blueprint.

Usage:
    PYTHONPATH=. python3 baseline/common/rollout/bench_rollout.py \
        baseline/humanoid21/blueprints/stage1_env.yaml \
        --episodes 256 --workers 48

    # Single-process debug:
    PYTHONPATH=. python3 baseline/common/rollout/bench_rollout.py \
        baseline/humanoid21/blueprints/stage1_env.yaml \
        --episodes 4 --workers 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

COMBATBENCH = str(Path(__file__).resolve().parents[3])
if COMBATBENCH not in sys.path:
    sys.path.insert(0, COMBATBENCH)

import numpy as np

from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from baseline.common.rollout.episode import Episode
from baseline.common.rollout.episode_recorder import EpisodeRecorder
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


def bench_single(
    env_bp_yaml: Path,
    n_episodes: int,
    obs_dim: int = 96,
    action_dim: int = 21,
) -> None:
    """Single-process benchmark: build once, run N episodes in a loop."""
    from baseline.common.rollout.episode import blueprint_hash as _bp_hash

    env_pb = ParameterizedEnvBlueprint.load(env_bp_yaml)
    env_bp = env_pb.materialize()

    print(f"=== Single-process bench: {n_episodes} episodes ===", flush=True)

    # Build policy (random init)
    t0 = time.perf_counter()
    policy = TanhGaussianMLPPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256,
    )
    policy.eval()
    print(f"  policy build:  {time.perf_counter() - t0:.3f}s", flush=True)

    # Build env (once)
    t0 = time.perf_counter()
    env_hash = _bp_hash(env_bp)
    #recorder = EpisodeRecorder(blueprint_hash=env_hash)
    runtime = env_bp.build() # recorders=[recorder])
    print(f"  env build:     {time.perf_counter() - t0:.3f}s", flush=True)

    runner = EpisodeRunner(runtime=runtime, policy_a=policy, policy_b=policy)

    # Run episodes
    times = []
    for i in range(n_episodes):
        t0 = time.perf_counter()
        runner.run_episode(seed=i + 1000, want_extras=False)
        dt = time.perf_counter() - t0
        #ep = recorder.get_last_episode()
        times.append(dt)
        #print(
        #    f"  ep {i:3d}: {dt:.3f}s  steps={len(ep.observations.get('robot_a', []))}",
        #    flush=True,
        #)
        print(
            f"  ep {i:3d}: {dt:.3f}s",
            flush=True,
        )

    times = np.array(times)
    print(
        f"  ---\n"
        f"  mean={times.mean():.3f}s  median={np.median(times):.3f}s  "
        f"min={times.min():.3f}s  max={times.max():.3f}s\n"
        f"  total={times.sum():.1f}s  throughput={n_episodes / times.sum():.1f} ep/s",
        flush=True,
    )


def bench_parallel(
    env_bp_yaml: Path,
    n_episodes: int,
    n_workers: int,
    obs_dim: int = 96,
    action_dim: int = 21,
) -> None:
    """Multi-process benchmark: ParallelRollouter, fresh env+policy per job."""
    from baseline.common.rollout.parallel_rollouter import ParallelRollouter
    from envs.framework.blueprint import EnvBlueprint
    from envs.framework.policy import PolicyBlueprint

    env_pb = ParameterizedEnvBlueprint.load(env_bp_yaml)
    env_bp = env_pb.materialize()

    print(f"=== Parallel bench: {n_episodes} episodes, {n_workers} workers ===", flush=True)

    # Export a random policy to disk so workers can load it
    actor = TanhGaussianMLPPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256,
    )
    export_dir = Path(__file__).resolve().parent / "_bench_export"
    policy_bp: PolicyBlueprint = actor.to_blueprint(dest_path=str(export_dir))

    # Build jobs (all share the same env blueprint)
    jobs = [
        (policy_bp, policy_bp, env_bp, i + 2000, None)
        for i in range(n_episodes)
    ]

    t0 = time.perf_counter()
    with ParallelRollouter(num_workers=n_workers) as rollouter:
        episodes = rollouter.collect(jobs)
    total = time.perf_counter() - t0

    steps = [len(ep.observations.get("robot_a", [])) for ep in episodes]
    print(
        f"  ---\n"
        f"  total={total:.1f}s  throughput={n_episodes / total:.1f} ep/s\n"
        f"  mean_steps={np.mean(steps):.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blueprint", type=Path, help="Path to env blueprint YAML")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=96)
    parser.add_argument("--action-dim", type=int, default=21)
    args = parser.parse_args()

    if args.workers <= 1:
        bench_single(
            args.blueprint, args.episodes,
            obs_dim=args.obs_dim, action_dim=args.action_dim,
        )
    else:
        bench_parallel(
            args.blueprint, args.episodes, args.workers,
            obs_dim=args.obs_dim, action_dim=args.action_dim,
        )


if __name__ == "__main__":
    main()
