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
import json
import sys
from pathlib import Path

from envs.framework.blueprint import EnvBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.policy import PolicyBlueprint
from envs.framework.round_runner import RoundRunner

from mocap.humanoid21.policy_capture.recorder import TrajectoryRecorder


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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_bp = EnvBlueprint.load(args.env_blueprint)
    policy_a = PolicyBlueprint.load(args.policy_a).build()
    policy_b = (PolicyBlueprint.load(args.policy_b).build()
                if args.policy_b else policy_a)

    provenance = {
        "env_blueprint": str(Path(args.env_blueprint).resolve()),
        "policy_a": str(Path(args.policy_a).resolve()),
        "policy_b": str(Path(args.policy_b).resolve()) if args.policy_b else None,
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
