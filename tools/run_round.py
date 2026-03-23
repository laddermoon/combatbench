"""
CombatBench Round Runner CLI

A unified script to run combat rounds between two policies.
All policies are loaded using a consistent specification format that supports constructor parameters.

Usage:
    # Run with Python module policies
    python tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy \
                             --policy-b combatbench.policy.StandingCombatPolicy

    # Run with SB3 model (using unified format)
    python tools/run_round.py \
        --policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=model.zip" \
        --video output.mp4

    # Run with parameters (query string format)
    python tools/run_round.py --policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2"

    # Run with config file
    python tools/run_round.py --policy-a "@policy_config.json"

    # Run with no policies (both standing)
    python tools/run_round.py --duration 10 --video test.mp4
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

# Add parent directory to path to import local modules
# The script is in combatbench/tools/, so we need to add combatbench/../ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _parse_policy_spec(policy_spec: str) -> tuple:
    """
    Parse policy specification into (module_path, class_name, kwargs).

    Supports formats:
    - "module.path.ClassName" -> (module.path, ClassName, {})
    - "module.path.ClassName?key=value&foo=bar" -> (module.path, ClassName, {key: value, foo: bar})
    - "path/to/file.py:ClassName" -> (path/to/file.py, ClassName, {})
    - "path/to/file.py:ClassName?key=value" -> (path/to/file.py, ClassName, {key: value})
    - "@config.json" -> (from config file, from config file, from config file)

    Args:
        policy_spec: Policy specification string

    Returns:
        tuple: (module_path_or_file, class_name, kwargs_dict)
    """
    from urllib.parse import parse_qs, urlencode

    # Check if it's a config file
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

    # Split into base spec and query string
    if '?' in policy_spec:
        base_spec, query_string = policy_spec.split('?', 1)
        params = parse_qs(query_string, keep_blank_values=True)
        # Parse values (handle booleans, numbers, strings)
        kwargs = {}
        for key, values in params.items():
            if len(values) != 1:
                raise ValueError(f"Parameter '{key}' specified multiple times")
            value = values[0]
            # Try to parse as JSON (handles numbers, booleans, null, arrays, objects)
            try:
                kwargs[key] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key] = value  # Keep as string
    else:
        base_spec = policy_spec
        kwargs = {}

    # Check if it's a file path with class name
    if ':' in base_spec and '.py:' in base_spec:
        file_path, class_name = base_spec.rsplit(':', 1)
        return file_path, class_name, kwargs

    # Check if it's a Python file path
    if Path(base_spec).exists() and Path(base_spec).suffix == '.py':
        return base_spec, None, kwargs

    # Assume it's a module path
    parts = base_spec.rsplit('.', 1)
    if len(parts) >= 2:
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]
        return module_path, class_name, kwargs

    raise ValueError(f"Invalid policy specification: {policy_spec}")


def load_policy(policy_spec: Optional[str], device: str = 'auto') -> Any:
    """
    Load a policy from various sources using a unified specification format.

    Args:
        policy_spec: Policy specification which can be:
            - None: Returns StandingCombatPolicy (default)
            - Python module path: e.g., "combatbench.policy.RandomCombatPolicy"
            - With parameters: "combatbench.policy.RandomCombatPolicy?scale=0.2&seed=42"
            - Python file path: e.g., "/path/to/policy.py:ClassName"
            - Config file: e.g., "@policy_config.json"
        device: Device for model inference (passed as device parameter if policy supports it)

    Returns:
        Policy instance with act(obs, info) -> np.ndarray method

    Examples:
        # No policy (standing)
        load_policy(None)

        # Simple module path
        load_policy("combatbench.policy.RandomCombatPolicy")

        # SB3 model with parameters
        load_policy("combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=model.zip&device=cuda")

        # With parameters (query string format)
        load_policy("combatbench.policy.RandomCombatPolicy?scale=0.2")
        load_policy("combatbench.policy.RandomCombatPolicy?scale=0.2&seed=42")

        # With complex parameters (use JSON values)
        load_policy("combatbench.policy.CustomPolicy?model_path=model.zip&noise=true&scale=0.5")

        # Python file with class
        load_policy("path/to/policy.py:MyPolicy")
        load_policy("path/to/policy.py:MyPolicy?param=value")

        # Config file
        load_policy("@policy_config.json")
    """
    from combatbench.policy import StandingCombatPolicy

    if policy_spec is None:
        return StandingCombatPolicy()

    # Parse the specification
    module_path, class_name, kwargs = _parse_policy_spec(policy_spec)

    # If it's a file path
    if module_path.endswith('.py'):
        spec = importlib.util.spec_from_file_location("policy_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if class_name is None:
            # Try to find a policy class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name.endswith('Policy'):
                    return attr(**kwargs)
            raise ImportError(f"No policy class found in {module_path}")
        else:
            policy_class = getattr(module, class_name)
            return policy_class(**kwargs)

    # Import from module path
    try:
        module = importlib.import_module(module_path)
        policy_class = getattr(module, class_name)
        return policy_class(**kwargs)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import policy '{policy_spec}': {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a CombatBench round between two policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Policy arguments
    parser.add_argument(
        '--policy-a', '--policy', '--model-a', '--model',
        type=str, default=None,
        help='Policy for robot A (red). Format: module.path.ClassName?param=value, or @config.json'
    )
    parser.add_argument(
        '--policy-b', '--model-b',
        type=str, default=None,
        help='Policy for robot B (blue). Format: module.path.ClassName?param=value, or @config.json'
    )

    # Environment arguments
    parser.add_argument(
        '--duration', '--match-duration',
        type=float, default=30.0,
        help='Round duration in seconds (default: 30.0)'
    )
    parser.add_argument(
        '--control-frequency', '--fps',
        type=int, default=20,
        help='Control frequency in Hz (default: 20)'
    )
    parser.add_argument(
        '--initial-distance',
        type=float, default=2.0,
        help='Initial distance between robots in meters (default: 2.0)'
    )
    parser.add_argument(
        '--phase',
        type=str, default=None,
        choices=['stand', 'fight', 'fight_attacker', 'fight_attacker_approach'],
        help='Training phase for controller configuration (default: None)'
    )

    # Non-fall mode arguments
    parser.add_argument(
        '--non-fall-mode', action='store_true',
        help='Enable non-fall mode (clamp root pitch/roll to prevent falling)'
    )
    parser.add_argument(
        '--non-fall-pitch-limit-deg',
        type=float, default=5.0,
        help='Pitch limit in degrees for non-fall mode (default: 5.0)'
    )
    parser.add_argument(
        '--non-fall-roll-limit-deg',
        type=float, default=5.0,
        help='Roll limit in degrees for non-fall mode (default: 5.0)'
    )
    parser.add_argument(
        '--damage-scale',
        type=float, default=100.0,
        help='Damage scaling factor (default: 100.0)'
    )

    # Output arguments
    parser.add_argument(
        '--video', '--output',
        type=str, default=None,
        help='Path to save video (e.g., round.mp4). If not specified, no video is saved.'
    )

    # Inference arguments
    parser.add_argument(
        '--device',
        type=str, default='auto',
        help='Device for policy inference (default: auto)'
    )

    # Verbosity
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Import RoundRunner after path setup
    from combatbench.envs import RoundRunner

    # Load policies
    print(f"Loading policy A: {args.policy_a or 'StandingCombatPolicy (default)'}")
    policy_a = load_policy(args.policy_a, device=args.device)
    print(f"  Loaded: {policy_a.__class__.__name__}")

    print(f"Loading policy B: {args.policy_b or 'StandingCombatPolicy (default)'}")
    policy_b = load_policy(args.policy_b, device=args.device)
    print(f"  Loaded: {policy_b.__class__.__name__}")

    # Determine render mode
    render_mode = "rgb_array" if args.video else None

    # Create round runner
    runner = RoundRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        render_mode=render_mode,
        match_duration=args.duration,
        control_frequency=args.control_frequency,
        initial_distance=args.initial_distance,
        phase=args.phase,
        non_fall_mode=args.non_fall_mode,
        non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
        non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
        damage_scale=args.damage_scale,
        verbose=not args.quiet,
    )

    # Run the round
    result = runner.run(save_video_path=args.video)

    # Print final summary
    if not args.quiet:
        print()
        print("=" * 60)
        print("Round Summary")
        print("=" * 60)
        print(f"Policy A: {policy_a.__class__.__name__}")
        print(f"Policy B: {policy_b.__class__.__name__}")
        print(f"Winner: {result.winner or 'draw'}")
        print(f"Steps: {result.steps}")
        print(f"Final HP: A={result.scores['robot_a']:.1f}, B={result.scores['robot_b']:.1f}")
        print(f"Damage dealt: A={result.damage_dealt['robot_a']:.1f}, B={result.damage_dealt['robot_b']:.1f}")
        if args.video:
            print(f"Video: {args.video} ({result.video_frames} frames)")
        print("=" * 60)


if __name__ == "__main__":
    main()
