#!/usr/bin/env python3
"""
Run a combat round between two policies.

This script should be run from the combatbench directory:
    cd /path/to/combatbench
    python -m envs.t800.run_round [OPTIONS]

Usage:
    # Run with default random policies
    python -m envs.t800.run_round --duration 10 --video test.mp4

    # Run with random policy
    python -m envs.t800.run_round --policy-a "random?action_dim=25" --duration 5 --video test.mp4

    # Run with parameters
    python -m envs.t800.run_round \
        --policy-a "random?scale=0.2&seed=42&action_dim=25" \
        --duration 15 --video output.mp4
"""

import argparse
import os
import sys
from pathlib import Path
# Set headless render mode
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a combat round between two policies (T800).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--policy-a', type=str, default=None,
                        help='Policy A specification (path to policy dir or "random")')
    parser.add_argument('--policy-b', type=str, default=None,
                        help='Policy B specification')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Match duration in seconds (default: 30.0)')
    parser.add_argument('--control-frequency', type=int, default=20,
                        help='Control frequency in Hz (default: 20)')
    parser.add_argument('--damage-scale', type=float, default=100.0,
                        help='Damage scaling factor (default: 100.0)')
    parser.add_argument('--video', '--output', type=str, default=None,
                        help='Path to save video (e.g. t800_battle.mp4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--freeze-a', action='store_true',
                        help='Freeze robot_a at initial pose')
    parser.add_argument('--freeze-b', action='store_true',
                        help='Freeze robot_b at initial pose')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import policy loading utility
    from policy import load_policy

    # Load policies
    # NOTE: T800 requires 25-dim actions, so default random policy sets action_dim=25.
    policy_a_spec = args.policy_a or 'random?action_dim=25'
    policy_b_spec = args.policy_b or 'random?action_dim=25'

    print(f"Loading policy A: {policy_a_spec}")
    policy_a = load_policy(policy_a_spec)
    print(f"  Loaded: {policy_a.__class__.__name__}")

    print(f"Loading policy B: {policy_b_spec}")
    policy_b = load_policy(policy_b_spec)
    print(f"  Loaded: {policy_b.__class__.__name__}")

    # Import T800 components
    from envs.t800 import make_env, T800CombatScoringPlugin, FrozenRobotPlugin
    from envs.framework.round_runner import RoundRunner
    from envs.framework.common_plugins import VideoRecorderPlugin

    # Prepare plugins (T800 scoring + optional video)
    plugins = [
        T800CombatScoringPlugin(damage_scale=args.damage_scale),
    ]
    if args.freeze_a:
        plugins.append(FrozenRobotPlugin(frozen_robot_id='robot_a'))
    if args.freeze_b:
        plugins.append(FrozenRobotPlugin(frozen_robot_id='robot_b'))
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
        print(f"Winner: {result.get('winner', 'Unknown')}")
        print(f"Steps: {result.get('steps', 0)}")
        print(f"Final HP: A={result.get('final_health', {}).get('robot_a', 0):.1f}, "
              f"B={result.get('final_health', {}).get('robot_b', 0):.1f}")
        if args.video:
            print(f"Video saved to: {args.video}")
        print("=" * 60)


if __name__ == "__main__":
    main()
