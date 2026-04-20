#!/usr/bin/env python3
"""
Run a combat round between two policies.

This script should be run from the combatbench directory:
    cd /path/to/combatbench
    python -m envs.humanoid21.run_round [OPTIONS]

Usage:
    # Run with default (standing) policies
    python -m envs.humanoid21.run_round --duration 10 --video test.mp4

    # Run with random policy
    python -m envs.humanoid21.run_round --policy-a random --duration 5 --video test.mp4

    # Run with parameters
    python -m envs.humanoid21.run_round \
        --policy-a "random?scale=0.2&seed=42" \
        --duration 15 --video output.mp4
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Set headless render mode if EGL is available
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Add project root to path for imports (when running script directly)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a combat round between two policies (Humanoid21).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--policy-a', type=str, default=None,
                        help='Policy A specification (e.g., random, standing, or path to policy directory)')
    parser.add_argument('--policy-b', type=str, default=None,
                        help='Policy B specification')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Match duration in seconds (default: 30.0)')
    parser.add_argument('--control-frequency', type=int, default=20,
                        help='Control frequency in Hz (default: 20)')
    parser.add_argument('--damage-scale', type=float, default=100.0,
                        help='Damage scaling factor (default: 100.0)')
    parser.add_argument('--video', '--output', type=str, default=None,
                        help='Path to save video (e.g., match.mp4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import policy loading utility
    from policy import load_policy

    # Load policies
    policy_a_spec = args.policy_a or 'noopaction'
    policy_b_spec = args.policy_b or 'noopaction'

    print(f"Loading policy A: {policy_a_spec}")
    policy_a = load_policy(policy_a_spec)
    print(f"  Loaded: {policy_a.__class__.__name__}")

    print(f"Loading policy B: {policy_b_spec}")
    policy_b = load_policy(policy_b_spec)
    print(f"  Loaded: {policy_b.__class__.__name__}")

    # Import framework components
    from envs.humanoid21 import make_env
    from envs.framework.round_runner import RoundRunner
    from envs.framework.common_plugins import VideoRecorderPlugin
    from envs.humanoid21.plugins import CombatScoringPlugin


    from envs.humanoid21.disturbance_plugins import RandomPushPlugin, InitialStatePerturbationPlugin
    '''
    RandomPushPlugin(
    target_robot="robot_a",
    target_body="torso",
    force_magnitude=12.0,
    min_interval=20,
    max_interval=50,
    push_duration_steps=2,
)
RandomPushPlugin(
    target_robot="robot_a",
    target_body="torso",
    force_magnitude=30.0,
    min_interval=15,
    max_interval=40,
    push_duration_steps=2,
)
RandomPushPlugin(
    target_robot="robot_a",
    target_body="torso",
    force_magnitude=60.0,
    min_interval=10,
    max_interval=30,
    push_duration_steps=2,
)


RandomPushPlugin(
            target_robot="robot_a",
            target_body="torso",
            force_magnitude=2.0,
            min_interval=10,
            max_interval=40,
            push_duration_steps=10,
        )


        RandomPushPlugin(
            target_robot="robot_a",
            target_body="torso",
            force_magnitude=30.0,
            min_interval=20,
            max_interval=50,
            push_duration_steps=10,
            random_seed=42,
        ),
        RandomPushPlugin(
            target_robot="robot_b",
            target_body="torso",
            force_magnitude=2.0,
            min_interval=10,
            max_interval=40,
            push_duration_steps=10,
            random_seed=42,
        ),
    '''

    # Prepare plugins
    plugins = [
        CombatScoringPlugin(damage_scale=args.damage_scale),
        InitialStatePerturbationPlugin(
            target_robot="robot_a",
            joint_pos_delta_max=0.05,
            joint_vel_delta_max=0.05,
            root_xy_offset_max=0.05,
            root_tilt_deg_max=10.0,
            root_linear_velocity_delta_max=[0.5, 0.5, 0.0],
            root_angular_velocity_delta_max=[0.5, 0.5, 0.2],
            random_seed=42,
        ),
    ]
    if args.video:
        plugins.append(VideoRecorderPlugin(fps=30, output_path=args.video))

    runtime = make_env(
        match_duration=args.duration,
        control_frequency=args.control_frequency,
        plugins=plugins,
    )

    # Run round
    runner = RoundRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        runtime=runtime,
        verbose=not args.quiet,
    )

    result = runner.run(seed=None)

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Round Summary")
        print("=" * 60)
        print(f"Policy A: {policy_a.__class__.__name__}")
        print(f"Policy B: {policy_b.__class__.__name__}")
        print(f"Winner: {result['winner']}")
        print(f"Steps: {result['steps']}")
        print(f"Final HP: A={result['final_health'].get('robot_a', 0):.1f}, B={result['final_health'].get('robot_b', 0):.1f}")
        if args.video:
            print(f"Video saved to: {args.video}")
        print("=" * 60)


if __name__ == "__main__":
    main()
