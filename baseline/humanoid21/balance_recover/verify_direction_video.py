"""验证 RelativeImpulsePlugin 方向正确性：生成 4 个方向的视频。

对 0°（正面）、90°（右侧）、180°（背面）、270°（左侧）各生成一个视频，
用大力 + 长 duration，让用户目视确认方向是否正确。

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/verify_direction_video.py \
        --policy-export baseline/runs/fixaw_survonly_crossphi2_s42/policy \
        --force 300 --duration 8 \
        --output-dir /data1/dev/verify_direction
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from envs.framework.round_runner import RoundRunner


def main() -> None:
    p = argparse.ArgumentParser(description="Verify RelativeImpulsePlugin direction via video")
    p.add_argument("--policy-export", required=True,
                   help="Path to policy export directory")
    p.add_argument("--blueprint", type=str,
                   default="baseline/humanoid21/balance_recover/relative_impulse_env.yaml",
                   help="Path to relative impulse env blueprint YAML.")
    p.add_argument("--force", type=float, default=300.0,
                   help="Force magnitude (N).")
    p.add_argument("--duration", type=int, default=8,
                   help="Impulse duration (action steps).")
    p.add_argument("--output-dir", type=str, default="/data1/dev/verify_direction",
                   help="Output directory for videos.")
    p.add_argument("--max-steps", type=int, default=400,
                   help="Max action steps per episode.")
    p.add_argument("--agent-id", type=str, default="robot_a")
    args = p.parse_args()

    policy_bp_path = Path(args.policy_export) / "policy_blueprint.yaml"
    policy_path_abs = str(policy_bp_path.resolve())

    env_pb = ParameterizedEnvBlueprint.load(args.blueprint)

    directions = [
        (0.0, "front"),
        (90.0, "right"),
        (180.0, "back"),
        (270.0, "left"),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for angle, label in directions:
        video_path = out_dir / f"impulse_{args.agent_id}_{label}_{int(angle)}deg.mp4"
        print(f"\n=== Direction {angle:.0f}° ({label}) -> {video_path} ===")

        env_bp = env_pb.materialize(
            max_steps=args.max_steps,
            agent_id=args.agent_id,
            tolerance=6,
            policy_blueprint_path=policy_path_abs,
            force_magnitude=args.force,
            duration_action_steps=args.duration,
            direction_angle=angle,
        )

        video_plugin = VideoRecorderPlugin(fps=30, output_path=str(video_path))

        policy_a = PolicyBlueprint.load(policy_bp_path).build()
        policy_b = PolicyBlueprint.load(policy_bp_path).build()

        runner = RoundRunner(
            blueprint=env_bp,
            policy_a=policy_a,
            policy_b=policy_b,
            video_plugin=video_plugin,
        )

        result = runner.run(seed=42)
        print(f"  Steps: {result['steps']}")
        print(f"  Termination: {result['termination_reasons']}")

        del runner

    print(f"\nAll videos saved to {out_dir}/")
    print("Expected behavior (angle = push direction, NOT force origin):")
    print("  0°   (front):  robot pushed FORWARD  (in the direction it faces)")
    print("  90°  (right):  robot pushed to its RIGHT")
    print("  180° (back):   robot pushed BACKWARD (opposite to facing direction)")
    print("  270° (left):   robot pushed to its LEFT")


if __name__ == "__main__":
    main()
