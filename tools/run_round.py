"""
CombatBench Round Runner CLI

A unified script to run combat rounds between two policies.
All policies are loaded using a consistent specification format that supports constructor parameters.

Usage:
    # Run with Python module policies
    python -m combatbench.tools.run_round --policy-a combatbench.policy.RandomCombatPolicy \
                             --policy-b combatbench.policy.StandingCombatPolicy

    # Run with SB3 model (using unified format)
    python -m combatbench.tools.run_round \
        --policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=model.zip" \
        --video output.mp4

    # Run with parameters (query string format)
    python -m combatbench.tools.run_round --policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2"

    # Run with config file
    python -m combatbench.tools.run_round --policy-a "@policy_config.json"

    # Run with no policies (both standing)
    python -m combatbench.tools.run_round --duration 10 --video test.mp4
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Set headless render mode if EGL is available
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Add parent directory to path to import local modules
# The script is in combatbench/tools/, so we need to add combatbench/../ to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RoundResult:
    """
    Result of a combat round.

    Attributes:
        steps: Total number of steps taken in the round
        end_reason: Reason why the round ended
        winner: Which robot won ('robot_a', 'robot_b', or 'draw')
        scores: Final HP scores for both robots
        initial_scores: Initial HP scores (usually 100 each)
        damage_dealt: Total damage dealt by each robot
        total_reward: Total shaped reward accumulated (if rewards were computed)
        video_frames: Number of video frames captured (if video was enabled)
    """
    steps: int
    end_reason: str
    winner: Optional[str]
    scores: Dict[str, float]
    initial_scores: Dict[str, float] = field(default_factory=lambda: {"robot_a": 100.0, "robot_b": 100.0})
    damage_dealt: Dict[str, float] = field(default_factory=lambda: {"robot_a": 0.0, "robot_b": 0.0})
    total_reward: Dict[str, float] = field(default_factory=lambda: {"robot_a": 0.0, "robot_b": 0.0})
    video_frames: int = 0


class RoundRunner:
    """
    Runs a complete combat round between two policies.

    This class handles:
        - Environment creation and reset
        - Policy execution for both robots
        - Step-by-step simulation
        - Result collection and statistics

    Example:
        >>> from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy
        >>> runner = RoundRunner(
        ...     policy_a=RandomCombatPolicy(),
        ...     policy_b=StandingCombatPolicy(),
        ...     render_mode="rgb_array",
        ... )
        >>> result = runner.run()
        >>> print(f"Winner: {result.winner}, Steps: {result.steps}")
    """

    def __init__(
        self,
        policy_a: Any,
        policy_b: Any,
        render_mode: Optional[str] = None,
        match_duration: float = 30.0,
        control_frequency: int = 20,
        initial_distance: float = 2.0,
        non_fall_mode: bool = True,
        enable_fall_detection: bool = False,
        video_fps: int = 30,
        verbose: bool = True,
    ):
        """
        Initialize the round runner.

        Args:
            policy_a: Policy for robot A (red)
                     Must implement act(obs, info) -> np.ndarray and reset()
            policy_b: Policy for robot B (blue)
                     Must implement act(obs, info) -> np.ndarray and reset()
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            match_duration: Round duration in seconds
            control_frequency: Control frequency in Hz
            initial_distance: Initial distance between robots
            non_fall_mode: Use Humanoid21NonFallEnv (True) or Humanoid21FallEnv (False)
            enable_fall_detection: Enable fall detection (only for FallEnv)
            video_fps: Video frame rate
            verbose: Print round progress
        """
        from combatbench.envs import Humanoid21NonFallEnv, Humanoid21FallEnv

        self.policy_a = policy_a
        self.policy_b = policy_b
        self.verbose = verbose
        self.non_fall_mode = non_fall_mode

        # Create environment
        if non_fall_mode:
            self.env = Humanoid21NonFallEnv(
                render_mode=render_mode,
                match_duration=match_duration,
                control_frequency=control_frequency,
                initial_distance=initial_distance,
            )
        else:
            self.env = Humanoid21FallEnv(
                render_mode=render_mode,
                match_duration=match_duration,
                control_frequency=control_frequency,
                initial_distance=initial_distance,
                enable_fall_detection=enable_fall_detection,
            )

        # Set video recording
        self.video_fps = video_fps

        # Statistics tracking
        self._total_reward = {"robot_a": 0.0, "robot_b": 0.0}
        self._damage_dealt = {"robot_a": 0.0, "robot_b": 0.0}

    def _print_header(self) -> None:
        """Print round start header."""
        if not self.verbose:
            return

        print("=" * 60)
        print("CombatBench Round Started")
        print(f"Environment: {self.env.__class__.__name__}")
        print(f"Duration: {self.env.match_duration}s")
        print(f"Control Frequency: {self.env.control_frequency}Hz")
        print(f"Initial Distance: {self.env.simulator.initial_distance}m")
        print("=" * 60)

    def _print_step_info(self, step: int, info: Dict[str, Any]) -> None:
        """Print periodic step information."""
        if not self.verbose:
            return

        if step % 100 == 0:
            if 'relative_metrics' in info:
                distance = info['relative_metrics']['robot_a']['distance']
            else:
                distance = 0.0
            scores = info.get('scores', {})
            print(
                f"Step {step:03d} - HP: {scores} - "
                f"Distance: {distance:.2f}m"
            )

    def _print_result(self, result: RoundResult) -> None:
        """Print round result."""
        if not self.verbose:
            return

        print("-" * 60)
        print(f"Round ended. Total steps: {result.steps}")
        print(f"Reason: {result.end_reason}")
        print(f"Final HP: robot_a={result.scores['robot_a']:.1f}, "
              f"robot_b={result.scores['robot_b']:.1f}")
        print(f"Damage dealt: robot_a={result.damage_dealt['robot_a']:.1f}, "
              f"robot_b={result.damage_dealt['robot_b']:.1f}")
        print(f"Winner: {result.winner or 'draw'}")
        print("-" * 60)

    def run(
        self,
        save_video_path: Optional[str] = None,
        action_callback: Optional[Callable[[Any, int], Dict[str, np.ndarray]]] = None,
        seed: Optional[int] = None,
    ) -> RoundResult:
        """
        Run a complete round.

        Args:
            save_video_path: Path to save video (mp4). If None, video is not saved.
            action_callback: Optional callback for per-step action override
                Function(env, step_index) -> {'robot_a': action_a, 'robot_b': action_b}

        Returns:
            RoundResult containing round statistics
        """
        # Enable video recording if needed
        if save_video_path:
            self.env.video_enabled = True

        # Reset environment
        obs, info = self.env.reset(seed=seed)

        # Reset policies
        if hasattr(self.policy_a, "reset"):
            self.policy_a.reset()
        if hasattr(self.policy_b, "reset"):
            self.policy_b.reset()

        # Initialize tracking
        self._total_reward = {"robot_a": 0.0, "robot_b": 0.0}
        self._damage_dealt = {"robot_a": 0.0, "robot_b": 0.0}
        initial_scores = info.get('scores', {"robot_a": 100.0, "robot_b": 100.0})

        self._print_header()

        # Main loop
        step_count = 0
        action_dim = self.env.action_space["robot_a"].shape[0]

        while True:
            # Get actions from policies
            try:
                act_a = self.policy_a.act(obs["robot_a_obs"], info)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Policy A failed: {e}, using zero action")
                act_a = np.zeros(action_dim, dtype=np.float32)

            try:
                act_b = self.policy_b.act(obs["robot_b_obs"], info)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Policy B failed: {e}, using zero action")
                act_b = np.zeros(action_dim, dtype=np.float32)

            action = {
                "robot_a": np.asarray(act_a, dtype=np.float32),
                "robot_b": np.asarray(act_b, dtype=np.float32),
            }

            # Allow action callback to override
            if action_callback is not None:
                action = action_callback(self.env, step_count)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1

            # Track rewards
            self._total_reward["robot_a"] += float(reward.get("robot_a", 0))
            self._total_reward["robot_b"] += float(reward.get("robot_b", 0))

            # Track damage dealt
            current_scores = info.get('scores', {"robot_a": 100.0, "robot_b": 100.0})
            for attacker, defender in [("robot_a", "robot_b"), ("robot_b", "robot_a")]:
                damage = max(0.0, initial_scores[defender] - current_scores[defender])
                self._damage_dealt[attacker] = damage

            # Print progress
            self._print_step_info(step_count, info)

            # Check termination
            if terminated or truncated:
                break

        # Save video
        video_frames = len(self.env.get_video_buffer())
        if save_video_path and video_frames > 0:
            if self.verbose:
                print(f"\nSaving video to {save_video_path}...")
            self.env.save_video(str(save_video_path), fps=self.video_fps)

        # Build result
        result = RoundResult(
            steps=step_count,
            end_reason=info.get("end_reason", "time_limit"),
            winner=info.get("winner"),
            scores=current_scores,
            initial_scores=initial_scores,
            damage_dealt=dict(self._damage_dealt),
            total_reward=dict(self._total_reward),
            video_frames=video_frames,
        )

        self._print_result(result)
        self.env.close()

        return result

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def run_round(
    policy_a: Any,
    policy_b: Any,
    render_mode: Optional[str] = None,
    match_duration: float = 30.0,
    control_frequency: int = 20,
    initial_distance: float = 2.0,
    non_fall_mode: bool = True,
    enable_fall_detection: bool = False,
    video_fps: int = 30,
    env_kwargs: Optional[Dict[str, Any]] = None,
    save_video_path: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> RoundResult:
    """
    Convenience function to run a single round.

    Args:
        policy_a: Policy for robot A
        policy_b: Policy for robot B
        render_mode: Rendering mode
        match_duration: Round duration in seconds
        control_frequency: Control frequency in Hz
        initial_distance: Initial distance between robots
        non_fall_mode: Use Humanoid21NonFallEnv (True) or Humanoid21FallEnv (False)
        enable_fall_detection: Enable fall detection (only for FallEnv)
        video_fps: Video frame rate
        env_kwargs: Additional keyword arguments (for future compatibility)
        save_video_path: Path to save video
        verbose: Print progress

    Returns:
        RoundResult containing round statistics

    Example:
        >>> from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy
        >>> result = run_round(
        ...     policy_a=RandomCombatPolicy(),
        ...     policy_b=StandingCombatPolicy(),
        ...     save_video_path="round.mp4",
        ... )
    """
    # Handle None policies - use default StandingCombatPolicy
    if policy_a is None:
        policy_a = load_policy(None)
    if policy_b is None:
        policy_b = load_policy(None)

    runner = RoundRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        render_mode=render_mode,
        match_duration=match_duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
        non_fall_mode=non_fall_mode,
        enable_fall_detection=enable_fall_detection,
        video_fps=video_fps,
        verbose=verbose,
    )
    return runner.run(save_video_path=save_video_path, seed=seed)


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

    # Add device to kwargs if specified
    if device != 'auto':
        kwargs['device'] = device

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

    # Non-fall mode arguments
    parser.add_argument(
        '--non-fall-mode',
        type=lambda x: x.lower() not in ['false', '0', 'no'],
        default=True,
        help='Use Humanoid21NonFallEnv (True) or Humanoid21FallEnv (False) (default: True)'
    )
    parser.add_argument(
        '--enable-fall-detection',
        action='store_true',
        help='Enable fall detection (only for FallEnv)'
    )
    parser.add_argument(
        '--video-fps',
        type=int,
        default=30,
        help='Video frame rate (default: 30)'
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
        non_fall_mode=args.non_fall_mode,
        enable_fall_detection=args.enable_fall_detection,
        video_fps=args.video_fps,
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
