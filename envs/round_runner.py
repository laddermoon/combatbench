"""
Round Runner Module

Provides a clean interface for running a complete combat round between two policies.
This module integrates CombatGymEnv with policies to execute full episodes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .combat_gym import CombatGymEnv


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
        phase: Optional[str] = None,
        non_fall_mode: bool = False,
        damage_scale: float = 100.0,
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
            phase: Training phase for controller config ('stand', 'fight', etc.)
            non_fall_mode: Enable non-fall mode (orientation clamping)
            damage_scale: Damage scaling factor
            verbose: Print round progress
        """
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.phase = phase
        self.verbose = verbose

        # Create environment
        self.env = CombatGymEnv(
            render_mode=render_mode,
            match_duration=match_duration,
            control_frequency=control_frequency,
            initial_distance=initial_distance,
            non_fall_mode=non_fall_mode,
            damage_scale=damage_scale,
        )

        # Statistics tracking
        self._total_reward = {"robot_a": 0.0, "robot_b": 0.0}
        self._damage_dealt = {"robot_a": 0.0, "robot_b": 0.0}

    def _configure_phase(self) -> None:
        """Apply phase-specific controller configuration."""
        if self.phase is None:
            return

        # Import phase configuration functions
        try:
            from combatbench.baseline.sb3.selfplay_env import (
                configure_base_env_for_fight,
                configure_base_env_for_fight_attacker,
                configure_base_env_for_stand,
            )
        except ImportError:
            # If baseline not available, skip phase configuration
            return

        if self.phase == "stand":
            configure_base_env_for_stand(self.env)
        elif self.phase == "fight":
            configure_base_env_for_fight(self.env)
        elif self.phase in ("fight_attacker", "fight_attacker_approach"):
            configure_base_env_for_fight_attacker(self.env)

        # Re-fetch observations after configuration
        mujoco.mj_forward(self.env.physics.model, self.env.physics.data)

    def _print_header(self) -> None:
        """Print round start header."""
        if not self.verbose:
            return

        print("=" * 60)
        print("CombatBench Round Started")
        print(f"Phase: {self.phase or 'default'}")
        print(f"Duration: {self.env.match_duration}s")
        print(f"Control Frequency: {self.env.control_frequency}Hz")
        print(f"Initial Distance: {self.env.initial_distance}m")
        print("=" * 60)

    def _print_step_info(self, step: int, info: Dict[str, Any]) -> None:
        """Print periodic step information."""
        if not self.verbose:
            return

        if step % 100 == 0:
            distance = np.linalg.norm(
                info["positions"]["robot_a"] - info["positions"]["robot_b"]
            )
            print(
                f"Step {step:03d} - HP: {info['scores']} - "
                f"Distance: {distance:.2f}m"
            )

    def _print_hit_info(self, step: int, info: Dict[str, Any]) -> None:
        """Print hit information."""
        if not self.verbose:
            return

        for robot_id in ("robot_a", "robot_b"):
            if info["hit_records"][robot_id]:
                icon = "🔴" if robot_id == "robot_a" else "🔵"
                name = "Robot A" if robot_id == "robot_a" else "Robot B"
                print(f"[Step {step}] {icon} {name} hit! "
                      f"{info['hit_records'][robot_id]}")

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
        # Reset environment
        obs, info = self.env.reset()
        self._configure_phase()
        obs = self.env._get_obs()
        info = self.env._build_info()

        # Reset policies
        if hasattr(self.policy_a, "reset"):
            self.policy_a.reset()
        if hasattr(self.policy_b, "reset"):
            self.policy_b.reset()

        # Initialize tracking
        self._total_reward = {"robot_a": 0.0, "robot_b": 0.0}
        self._damage_dealt = {"robot_a": 0.0, "robot_b": 0.0}
        initial_scores = dict(info["scores"])

        self._print_header()

        # Main loop
        step_count = 0
        action_dim = self.env.robot_a.ACTION_DIM

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

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(
                action_dict=action,
                action_callback=action_callback,
            )
            step_count += 1

            # Track rewards and damage
            self._total_reward["robot_a"] += float(reward.get("robot_a", 0))
            self._total_reward["robot_b"] += float(reward.get("robot_b", 0))

            # Track damage dealt
            current_scores = info["scores"]
            for attacker, defender in [("robot_a", "robot_b"), ("robot_b", "robot_a")]:
                damage = max(0.0, initial_scores[defender] - current_scores[defender])
                self._damage_dealt[attacker] = damage

            # Print progress
            self._print_step_info(step_count, info)
            self._print_hit_info(step_count, info)

            # Check termination
            if terminated or truncated:
                break

        # Save video
        video_frames = len(self.env.get_video_buffer())
        if save_video_path and video_frames > 0:
            if self.verbose:
                print(f"\nSaving video to {save_video_path}...")
            self.env.save_video(str(save_video_path), fps=self.env.video_sample_frequency)

        # Build result
        result = RoundResult(
            steps=step_count,
            end_reason=info.get("end_reason", "unknown"),
            winner=info.get("winner"),
            scores=info["scores"],
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
    phase: Optional[str] = None,
    save_video_path: Optional[str] = None,
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
        phase: Training phase for controller config
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
    runner = RoundRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        render_mode=render_mode,
        match_duration=match_duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
        phase=phase,
        verbose=verbose,
    )
    return runner.run(save_video_path=save_video_path)


# Import mujoco for configuration
try:
    import mujoco
except ImportError:
    mujoco = None
