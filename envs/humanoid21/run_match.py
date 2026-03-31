#!/usr/bin/env python3
"""
Run a multi-round match between two policies.

This script should be run from the combatbench directory:
    cd /path/to/combatbench
    python -m envs.humanoid21.run_match [OPTIONS]

Usage:
    # Run 6 rounds with default (standing) policies
    python -m envs.humanoid21.run_match --duration 10 --video-dir videos/

    # Run 3 rounds with random policy
    python -m envs.humanoid21.run_match \
        --policy-a policy.RandomCombatPolicy \
        --rounds 3 \
        --video-dir videos/

    # Run with parameters
    python -m envs.humanoid21.run_match \
        --policy-a "policy.RandomCombatPolicy?scale=0.2&seed=42" \
        --policy-b "policy.RandomCombatPolicy?scale=0.1&seed=43" \
        --duration 15 \
        --rounds 3 \
        --video-dir videos/

    # Run with config file
    python -m envs.humanoid21.run_match \
        --policy-a "@configs/policy_a.json" \
        --policy-b "@configs/policy_b.json" \
        --rounds 6 \
        --video-dir videos/

Match Rules:
    1. Initial HP: 100 for each robot
    2. KO Condition: Reduce opponent's HP to 0
    3. Time Decision: Higher HP wins when time runs out
    4. Tie Decision: Draw if HP is equal
    5. Round Duration: Each round is 30 seconds (default)
    6. Total Rounds: 6 rounds (default)
    7. HP Continuation: HP carries over between rounds
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Set headless render mode BEFORE any imports
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')


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
        description='Run a multi-round match between two policies (Humanoid21).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--policy-a', type=str, default=None,
                        help='Policy A specification (e.g., policy.RandomCombatPolicy)')
    parser.add_argument('--policy-b', type=str, default=None,
                        help='Policy B specification')
    parser.add_argument('--rounds', type=int, default=6,
                        help='Total number of rounds (default: 6)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Match duration per round in seconds (default: 30.0)')
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
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Directory to save round videos (e.g., videos/)')
    parser.add_argument('--result-file', type=str, default=None,
                        help='Path to save match result as JSON (e.g., result.json)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for policy inference (auto/cpu/cuda)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    return parser.parse_args()


def serialize_match_result(result) -> dict:
    """
    将 MatchResult 转换为 JSON 可序列化的字典

    Args:
        result: MatchResult 数据类

    Returns:
        JSON 可序列化的字典
    """
    import numpy as np
    from dataclasses import asdict

    def serialize_value(val):
        """递归序列化值，处理 numpy 数组和 dataclass"""
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, dict):
            return {k: serialize_value(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return type(val)(serialize_value(v) for v in val)
        elif hasattr(val, '__dataclass_fields__'):
            return {k: serialize_value(getattr(val, k)) for k in val.__dataclass_fields__}
        else:
            return val

    return serialize_value(asdict(result))


def save_match_result(result: dict, filepath: str, policy_a_name: str, policy_b_name: str) -> None:
    """
    保存比赛结果到 JSON 文件

    Args:
        result: 序列化后的比赛结果字典
        filepath: 保存路径
        policy_a_name: 策略 A 名称
        policy_b_name: 策略 B 名称
    """
    # 添加元数据
    output = {
        'metadata': {
            'policy_a': policy_a_name,
            'policy_b': policy_b_name,
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        },
        'result': result
    }

    # 确保目录存在
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Match result saved to: {filepath}")


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
    from envs.framework.match_runner import MatchRunner
    from envs.framework.common_plugins import VideoRecorderPlugin

    # Create environment factory function
    def env_factory(initial_health_a: float = 100.0, initial_health_b: float = 100.0):
        plugins = []
        # Always add video plugin when video_dir is specified
        if args.video_dir is not None:
            plugins.append(VideoRecorderPlugin(fps=30))

        return make_env(
            match_duration=args.duration,
            control_frequency=args.control_frequency,
            non_fall_mode=args.non_fall_mode,
            non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
            non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
            damage_scale=args.damage_scale,
            initial_health_a=initial_health_a,
            initial_health_b=initial_health_b,
            plugins=plugins,
        )

    # Run match
    runner = MatchRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        env_factory=env_factory,
        total_rounds=args.rounds,
        verbose=not args.quiet,
    )

    # Run with video directory if specified
    result = runner.run(seed=None, video_dir=args.video_dir)

    # Save result to JSON if specified
    if args.result_file:
        serialized_result = serialize_match_result(result)
        save_match_result(
            serialized_result,
            args.result_file,
            policy_a.__class__.__name__,
            policy_b.__class__.__name__
        )

    # Print final summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Match Final Summary")
        print("=" * 60)
        print(f"Policy A: {policy_a.__class__.__name__}")
        print(f"Policy B: {policy_b.__class__.__name__}")
        print(f"Total Rounds: {result.rounds_completed}/{result.total_rounds}")
        print(f"Final Winner: {result.final_winner}")
        if result.ko_winner:
            print(f"KO Winner: {result.ko_winner}")
        print(f"Total Score: A={result.total_score['robot_a']}, B={result.total_score['robot_b']}")
        if args.video_dir:
            print(f"Videos saved to: {args.video_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
