"""
Humanoid21 Match Environment Example

This script demonstrates the usage of Humanoid21MatchEnv.
This is a dual-agent competition environment suitable for matches.
"""

import os
import sys
from pathlib import Path

# Set headless render mode
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from combatbench.envs.humanoid21 import Humanoid21MatchEnv


def run_match_example():
    """Run match environment example with video recording."""

    print("=" * 60)
    print("Humanoid21 Match Environment Example")
    print("=" * 60)

    # Create environment with video recording
    env = Humanoid21MatchEnv(
        render_mode="rgb_array",
        match_duration=10.0,
        control_frequency=20.0,
        enable_nonfall=True,  # Use upright constraints
        enable_fall_detection=True,
    )

    # Enable video recording
    env.video_enabled = True

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"Mode: Competition match (dual agent)")
    print(f"Features: Non-fall mode enabled")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run episodes
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n--- Match {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()

        episode_reward_a = 0.0
        episode_reward_b = 0.0
        step_count = 0
        done = False

        while not done:
            # Random actions for both robots
            action = {
                'robot_a': np.random.uniform(-0.8, 0.8, size=21),
                'robot_b': np.random.uniform(-0.8, 0.8, size=21),
            }

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward_a += reward['robot_a']
            episode_reward_b += reward['robot_b']
            step_count += 1

            done = terminated or truncated

            if step_count % 50 == 0:
                scores = info.get('scores', {})
                print(f"  Step {step_count}: "
                      f"HP A={scores.get('robot_a', 100):.1f}, "
                      f"HP B={scores.get('robot_b', 100):.1f}")

        # Determine winner
        scores = info.get('scores', {})
        hp_a = scores.get('robot_a', 100)
        hp_b = scores.get('robot_b', 100)

        if hp_a > hp_b:
            winner = "robot_a"
        elif hp_b > hp_a:
            winner = "robot_b"
        else:
            winner = "draw"

        print(f"Match {episode + 1} finished!")
        print(f"  Steps: {step_count}")
        print(f"  Final HP: A={hp_a:.1f}, B={hp_b:.1f}")
        print(f"  Winner: {winner}")

    # Save video
    video_path = Path(__file__).parent / "videos" / "humanoid21_match_example.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving video to {video_path}...")
    success = env.save_video(str(video_path), fps=30)

    if success:
        print(f"Video saved successfully! ({video_path})")
    else:
        print("Failed to save video.")

    env.close()
    print("\n" + "=" * 60)
    print("Match example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_match_example()
