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

    # Inject additional runtime plugins (repeatable)
    python -m envs.humanoid21.run_round \
        --plugin "baseline.humanoid21.common:ImbalanceTerminationPlugin?agent_id=robot_a&grace_steps=2"
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl

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
                        help='Round duration in seconds (default: 30.0)')
    parser.add_argument('--control-frequency', type=int, default=20,
                        help='Control frequency in Hz (default: 20)')
    parser.add_argument('--damage-scale', type=float, default=100.0,
                        help='Damage scaling factor (default: 100.0)')
    parser.add_argument('--video', '--output', type=str, default=None,
                        help='Path to save video (e.g., match.mp4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument(
        '--plugin',
        action='append',
        default=[],
        metavar='SPEC',
        help=(
            "Inject runtime plugin, repeatable. "
            "Format: module.path:ClassName?key=value&key2=value2"
        ),
    )

    return parser.parse_args()


def _coerce_cli_value(raw: str) -> Any:
    text = str(raw)
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_plugin_spec(spec: str) -> tuple[str, str, Dict[str, Any]]:
    module_and_class, sep, query = str(spec).partition("?")
    if ":" not in module_and_class:
        raise ValueError(
            f"Invalid --plugin spec: {spec!r}. "
            "Expected format module.path:ClassName?key=value"
        )
    module_path, class_name = module_and_class.split(":", 1)
    module_path = module_path.strip()
    class_name = class_name.strip()
    if not module_path or not class_name:
        raise ValueError(
            f"Invalid --plugin spec: {spec!r}. "
            "module.path and ClassName must be non-empty."
        )

    kwargs: Dict[str, Any] = {}
    if sep and query:
        for key, value in parse_qsl(query, keep_blank_values=True):
            k = key.strip()
            if not k:
                continue
            kwargs[k] = _coerce_cli_value(value)
    return module_path, class_name, kwargs


def _load_plugin_from_spec(spec: str) -> Any:
    module_path, class_name, kwargs = _parse_plugin_spec(spec)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Plugin class {class_name!r} not found in module {module_path!r}"
        )
    if not callable(cls):
        raise TypeError(
            f"Plugin target {module_path}:{class_name} is not callable"
        )
    return cls(**kwargs)


def main() -> None:
    args = parse_args()

    # Import policy loading utility
    from policy import load_policy

    # Load policies
    policy_a_spec = args.policy_a or 'random'
    policy_b_spec = args.policy_b or 'random'

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

    # Prepare plugins
    plugins = [
        CombatScoringPlugin(damage_scale=args.damage_scale),
    ]
    if args.video:
        plugins.append(VideoRecorderPlugin(fps=30, output_path=args.video))
    for plugin_spec in args.plugin:
        plugin_obj = _load_plugin_from_spec(plugin_spec)
        plugins.append(plugin_obj)
        if not args.quiet:
            print(f"Injected plugin: {plugin_obj.__class__.__name__} ({plugin_spec})")

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
