"""Batch driver for policy-rollout motion capture.

Runs N episodes of a push-probe env with a trained policy on each robot,
recording the full physical state via ``TrajectoryRecorder``.  Optionally
renders the first ``--render`` episodes to mp4 for visual review.

Usage (from repo root)::

    PYTHONPATH=. python3 -m mocap.humanoid21.policy_capture.collect \
        --env-blueprint mocap/humanoid21/policy_capture/push_probe_base_env.yaml \
        --policy-a  <blueprint.yaml> \
        --policy-b  <blueprint.yaml>      # default: same as --policy-a
        --episodes 8 --seed-base 1000 \
        --out-dir /data1/dev/policy_capture/<run_name> \
        --render 2

Outputs per episode ``ep_<seed>_{robot_a,robot_b}.npz`` +
``ep_<seed>_meta.json`` (see recorder.py docstring), plus ``ep_<seed>.mp4``
for rendered episodes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from envs.framework.blueprint import EnvBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.policy import PolicyBlueprint
from envs.framework.round_runner import RoundRunner

from mocap.humanoid21.policy_capture.recorder import TrajectoryRecorder


def _parse_range(spec: str) -> tuple[float, float]:
    lo, hi = spec.split(":", 1)
    return float(lo), float(hi)


def _patch_push_plugins(env_bp: EnvBlueprint, args) -> EnvBlueprint:
    """Override RandomPushPlugin knobs without touching the yaml file."""
    if args.push_force is None and args.push_duration is None \
            and args.push_interval is None:
        return env_bp
    plugins = []
    for spec in env_bp.plugins:
        if not spec.cls.endswith("RandomPushPlugin"):
            plugins.append(spec)
            continue
        cfg = dict(spec.config)
        if args.push_force is not None:
            lo, hi = _parse_range(args.push_force)
            cfg["force_magnitude_min"], cfg["force_magnitude_max"] = lo, hi
        if args.push_duration is not None:
            cfg["push_duration_steps"] = int(args.push_duration)
        if args.push_interval is not None:
            lo, hi = _parse_range(args.push_interval)
            cfg["min_interval"], cfg["max_interval"] = int(lo), int(hi)
        plugins.append(dataclasses.replace(spec, config=cfg))
    return dataclasses.replace(env_bp, plugins=tuple(plugins))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-blueprint", required=True)
    ap.add_argument("--policy-a", required=True, help="Policy blueprint yaml for robot_a")
    ap.add_argument("--policy-b", default=None, help="Blueprint yaml for robot_b (default: same as A)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--seed-stride", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--render", type=int, default=0,
                    help="Render mp4 for the first N episodes")
    ap.add_argument("--push-force", default=None, metavar="MIN:MAX",
                    help="Override push force range (N)")
    ap.add_argument("--push-duration", type=int, default=None,
                    help="Override push length in action steps (1 = 50ms)")
    ap.add_argument("--push-interval", default=None, metavar="MIN:MAX",
                    help="Override push interval range (action steps)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_bp = _patch_push_plugins(
        EnvBlueprint.load(args.env_blueprint), args)
    policy_a = PolicyBlueprint.load(args.policy_a).build()
    policy_b = (PolicyBlueprint.load(args.policy_b).build()
                if args.policy_b else policy_a)

    provenance = {
        "env_blueprint": str(Path(args.env_blueprint).resolve()),
        "policy_a": str(Path(args.policy_a).resolve()),
        "policy_b": str(Path(args.policy_b).resolve()) if args.policy_b else None,
        "push_overrides": {
            "force": args.push_force,
            "duration": args.push_duration,
            "interval": args.push_interval,
        },
    }

    for ep in range(args.episodes):
        seed = args.seed_base + ep * args.seed_stride
        video_plugin = None
        if ep < args.render:
            video_plugin = VideoRecorderPlugin(
                fps=30, output_path=str(out_dir / f"ep_{seed}.mp4"))
        recorder = TrajectoryRecorder(
            output_dir=str(out_dir),
            phy_steps_per_action=env_bp.phy_steps_per_action,
            provenance=provenance,
        )
        with RoundRunner(
            blueprint=env_bp,
            policy_a=policy_a,
            policy_b=policy_b,
            video_plugin=video_plugin,
            recorders=[recorder],
        ) as runner:
            result = runner.run(seed=seed)
        print(f"[ep seed={seed}] steps={result['steps']} "
              f"termination={result['termination_reasons']}", flush=True)


if __name__ == "__main__":
    main()
