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
import imageio.v2 as imageio
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
    parser.add_argument('--sample-every-steps', type=int, default=5,
                        help='Save one observer visualization image every N control steps (default: 5)')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory used to save sampled observer visualization images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional environment seed')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    return parser.parse_args()


def _normalize_observer_output(output: Any) -> tuple[Any, Dict[str, Any]]:
    if output is None:
        return None, {}
    if isinstance(output, tuple) and len(output) == 2:
        payload, info = output
        if isinstance(info, dict):
            return payload, dict(info)
        return payload, {"observer_output": info}
    if isinstance(output, dict) and ("obs" in output or "observation" in output):
        payload = output.get("obs", output.get("observation"))
        info: Dict[str, Any] = {}
        raw_info = output.get("info")
        if isinstance(raw_info, dict):
            info.update(raw_info)
        elif raw_info is not None:
            info["observer_info"] = raw_info
        for key, value in output.items():
            if key not in {"obs", "observation", "info"}:
                info[key] = value
        return payload, info
    return output, {}


def _collect_runtime_view(runtime: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    obs: Dict[str, Any] = {}
    info: Dict[str, Any] = {
        "robot_a": {},
        "robot_b": {},
    }
    for agent_id in ("robot_a", "robot_b"):
        payload, agent_info = _normalize_observer_output(
            runtime.get_observer_output(f"{agent_id}_obs")
        )
        obs[agent_id] = payload
        info[agent_id].update(agent_info)
    return obs, info


def _build_policy_info(info: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    opponent_id = "robot_b" if agent_id == "robot_a" else "robot_a"
    shared_info = info["shared"]
    agent_info = info[agent_id]
    opponent_info = info[opponent_id]
    policy_info = dict(shared_info)
    policy_info["shared"] = shared_info
    policy_info["self"] = agent_info
    policy_info["opponent"] = opponent_info
    policy_info.update(agent_info)
    return policy_info


def _resolve_action_dim(runtime: Any, policy_a: Any, policy_b: Any) -> int:
    action_space = getattr(runtime, "action_space", None)
    if action_space is not None and hasattr(action_space, "spaces") and "robot_a" in action_space.spaces:
        return int(action_space.spaces["robot_a"].shape[0])
    for policy in (policy_a, policy_b):
        if hasattr(policy, "ACTION_DIM"):
            return int(policy.ACTION_DIM)
    return 21


def _normalize_action(action: Any) -> Optional[np.ndarray]:
    if action is None:
        return None
    return np.asarray(action, dtype=np.float32)


def _resolve_winner(shared_info: Dict[str, Any], final_health: Dict[str, float]) -> str:
    if isinstance(shared_info.get("winner"), str):
        return shared_info["winner"]
    health_a = float(final_health.get("robot_a", 0.0))
    health_b = float(final_health.get("robot_b", 0.0))
    if health_a <= 0.0 and health_b <= 0.0:
        return "draw"
    if health_a <= 0.0:
        return "robot_b"
    if health_b <= 0.0:
        return "robot_a"
    if health_a > health_b:
        return "robot_a"
    if health_b > health_a:
        return "robot_b"
    return "draw"


def _resolve_image_dir(args: argparse.Namespace) -> Path:
    if args.image_dir:
        return Path(args.image_dir).expanduser().resolve()
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        return video_path.parent / f"{video_path.stem}_observer_frames"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "verify_env_observer_frames" / timestamp


def _save_observer_image(balance_observer: Any, image_dir: Path, step_index: int, quiet: bool) -> Path:
    image = balance_observer.get_visualization_image()
    image_path = image_dir / f"step_{step_index:05d}.png"
    imageio.imwrite(str(image_path), image)
    if not quiet:
        print(f"Saved observer image: {image_path}")
    return image_path


def main() -> None:
    args = parse_args()

    from policy import load_policy

    policy_a_spec = args.policy_a or 'random'
    policy_b_spec = args.policy_b or 'random'

    print(f"Loading policy A: {policy_a_spec}")
    policy_a = load_policy(policy_a_spec)
    print(f"  Loaded: {policy_a.__class__.__name__}")

    print(f"Loading policy B: {policy_b_spec}")
    policy_b = load_policy(policy_b_spec)
    print(f"  Loaded: {policy_b.__class__.__name__}")

    from envs.humanoid21 import make_env
    from envs.framework.common_plugins import VideoRecorderPlugin
    from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver
    from envs.humanoid21.plugins import CombatScoringPlugin

    plugins = [
        CombatScoringPlugin(damage_scale=args.damage_scale),
    ]
    if args.video:
        plugins.append(VideoRecorderPlugin(fps=30, output_path=args.video))

    balance_observers = {
        "robot_a": Humanoid21BalanceAnalysisObserver("robot_a"),
        "robot_b": Humanoid21BalanceAnalysisObserver("robot_b"),
    }
    observer_plugins = {
        "robot_a_balance": balance_observers["robot_a"],
        "robot_b_balance": balance_observers["robot_b"],
    }

    image_dir_root: Optional[Path] = None
    image_dirs: Dict[str, Path] = {}
    if args.sample_every_steps > 0:
        image_dir_root = _resolve_image_dir(args)
        image_dir_root.mkdir(parents=True, exist_ok=True)
        for agent_id in balance_observers:
            agent_dir = image_dir_root / agent_id
            agent_dir.mkdir(parents=True, exist_ok=True)
            image_dirs[agent_id] = agent_dir

    runtime = make_env(
        match_duration=args.duration,
        control_frequency=args.control_frequency,
        plugins=plugins,
        observer_plugins=observer_plugins,
    )
    try:
        runtime.reset(seed=args.seed)
        obs, info = _collect_runtime_view(runtime)

        if hasattr(policy_a, "reset"):
            policy_a.reset()
        if hasattr(policy_b, "reset"):
            policy_b.reset()

        action_dim = _resolve_action_dim(runtime, policy_a, policy_b)
        step_count = 0
        last_saved_step: Optional[int] = None

        if image_dirs:
            for agent_id, balance_observer in balance_observers.items():
                _save_observer_image(balance_observer, image_dirs[agent_id], step_count, args.quiet)
            last_saved_step = step_count

        while True:
            try:
                act_a = policy_a.act(
                    obs["robot_a"],
                    _build_policy_info(info, "robot_a"),
                ) if hasattr(policy_a, "act") else np.zeros(action_dim, dtype=np.float32)
            except Exception as exc:
                if not args.quiet:
                    print(f"Warning: Policy A failed: {exc}")
                act_a = np.zeros(action_dim, dtype=np.float32)

            try:
                act_b = policy_b.act(
                    obs["robot_b"],
                    _build_policy_info(info, "robot_b"),
                ) if hasattr(policy_b, "act") else np.zeros(action_dim, dtype=np.float32)
            except Exception as exc:
                if not args.quiet:
                    print(f"Warning: Policy B failed: {exc}")
                act_b = np.zeros(action_dim, dtype=np.float32)

            runtime.step(_normalize_action(act_a), _normalize_action(act_b))
            step_count += 1
            obs, info = _collect_runtime_view(runtime)

            if image_dirs and step_count % args.sample_every_steps == 0:
                for agent_id, balance_observer in balance_observers.items():
                    _save_observer_image(balance_observer, image_dirs[agent_id], step_count, args.quiet)
                last_saved_step = step_count

            terminated, truncated = runtime.get_termination_flags()
            if terminated or truncated:
                break

        if image_dirs and last_saved_step != step_count:
            for agent_id, balance_observer in balance_observers.items():
                _save_observer_image(balance_observer, image_dirs[agent_id], step_count, args.quiet)

        shared_info = info["shared"]
        final_health = dict(shared_info.get("health", {}))
        result = {
            "steps": step_count,
            "winner": _resolve_winner(shared_info, final_health),
            "final_health": final_health,
            "damage_taken": dict(shared_info.get("damage_taken", {})),
            "termination_reasons": shared_info.get("termination_reasons", []),
        }

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
            if image_dir_root is not None:
                print(f"Observer images saved to: {image_dir_root}")
                print("Balance observer agents: robot_a, robot_b")
                print(f"Sample every steps: {args.sample_every_steps}")
            print("=" * 60)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
