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
    python -m envs.humanoid21.run_round --policy-a policy.RandomCombatPolicy --duration 5 --video test.mp4

    # Run with parameters
    python -m envs.humanoid21.run_round \
        --policy-a "policy.RandomCombatPolicy?scale=0.2&seed=42" \
        --duration 15 --video output.mp4

    # Run with config file
    python -m envs.humanoid21.run_round \
        --policy-a "@configs/policy_a.json" \
        --policy-b "@configs/policy_b.json" \
        --video match.mp4
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Set headless render mode if EGL is available
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Add project root to path for imports (when running script directly)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _parse_policy_spec(policy_spec: str) -> tuple:
    from urllib.parse import parse_qs, urlencode

    if policy_spec.startswith('@'):
        config_path = Path(policy_spec[1:])
        if not config_path.exists():
            raise FileNotFoundError(f"Policy config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        policy_type = config.get('type')
        params = config.get('params', {})

        if policy_type is None:
            raise ValueError(f"Config file missing 'type' field: {config_path}")

        return _parse_policy_spec(f"{policy_type}?{urlencode(params)}" if params else policy_type)

    if '?' in policy_spec:
        base_spec, query_string = policy_spec.split('?', 1)
        params = parse_qs(query_string, keep_blank_values=True)
        kwargs = {}
        for key, values in params.items():
            if len(values) != 1:
                raise ValueError(f"Parameter '{key}' specified multiple times")
            value = values[0]
            try:
                kwargs[key] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key] = value
    else:
        base_spec = policy_spec
        kwargs = {}

    if ':' in base_spec:
        module_path, class_name = base_spec.split(':', 1)
    else:
        module_path, class_name = base_spec.rsplit('.', 1)

    return module_path, class_name, kwargs


def load_policy(policy_spec: Optional[str], device: str = 'auto') -> Any:
    if not policy_spec:
        from policy.standing.policy import StandingCombatPolicy
        return StandingCombatPolicy()

    module_path, class_name, kwargs = _parse_policy_spec(policy_spec)

    # Convert common kwargs
    if 'device' not in kwargs and device != 'auto':
        kwargs['device'] = device

    if module_path.endswith('.py'):
        path = Path(module_path)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    policy_cls = getattr(module, class_name)
    return policy_cls(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a combat round between two policies (Humanoid21).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--policy-a', type=str, default=None,
                        help='Policy A specification (e.g., policy.RandomCombatPolicy)')
    parser.add_argument('--policy-b', type=str, default=None,
                        help='Policy B specification')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Match duration in seconds (default: 30.0)')
    parser.add_argument('--control-frequency', type=int, default=20,
                        help='Control frequency in Hz (default: 20)')
    parser.add_argument('--initial-distance', type=float, default=2.0,
                        help='Initial distance between robots (default: 2.0)')
    parser.add_argument('--non-fall-mode', action='store_true',
                        help='Enable non-fall mode (keep robots upright)')
    parser.add_argument('--non-fall-pitch-limit-deg', type=float, default=5.0,
                        help='Pitch limit for non-fall mode in degrees (default: 5.0)')
    parser.add_argument('--non-fall-roll-limit-deg', type=float, default=5.0,
                        help='Roll limit for non-fall mode in degrees (default: 5.0)')
    parser.add_argument('--damage-scale', type=float, default=100.0,
                        help='Damage scaling factor (default: 100.0)')
    parser.add_argument('--video', '--output', type=str, default=None,
                        help='Path to save video (e.g., match.mp4)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for policy inference (auto/cpu/cuda)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load policies
    print(f"Loading policy A: {args.policy_a or 'StandingCombatPolicy (default)'}")
    policy_a = load_policy(args.policy_a, device=args.device)
    print(f"  Loaded: {policy_a.__class__.__name__}")

    print(f"Loading policy B: {args.policy_b or 'StandingCombatPolicy (default)'}")
    policy_b = load_policy(args.policy_b, device=args.device)
    print(f"  Loaded: {policy_b.__class__.__name__}")

    # Import framework components
    from envs.humanoid21 import make_env
    from envs.framework.round_runner import RoundRunner
    from envs.framework.common_plugins import VideoRecorderPlugin

    # Prepare plugins
    plugins = []
    if args.video:
        # Set video path via class variable for plugin override
        VideoRecorderPlugin.set_videosave_path(args.video)
        plugins.append(VideoRecorderPlugin(fps=30))

    runtime = make_env(
        match_duration=args.duration,
        control_frequency=args.control_frequency,
        non_fall_mode=args.non_fall_mode,
        non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
        non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
        damage_scale=args.damage_scale,
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
