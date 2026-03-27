"""
Humanoid21 Vs Policy Environment Example

This script demonstrates the usage of Humanoid21VsPolicyEnv.
The opponent uses a custom policy.
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
from combatbench.envs.humanoid21 import Humanoid21VsPolicyEnv
from combatbench.policy import RandomCombatPolicy


def run_vs_policy_example():
    """Run vs policy environment example with video recording."""

    print("=" * 60)
    print("Humanoid21 Vs Policy Environment Example")
    print("=" * 60)

    # Create opponent policy
    opponent_policy = RandomCombatPolicy(scale=0.3, seed=42)

    # Create environment with video recording
    env = Humanoid21VsPolicyEnv(
        opponent_policy=opponent_policy,
        render_mode="rgb_array",
        match_duration=10.0,
        control_frequency=20.0,
        enable_fall_detection=True,
    )

    # Enable video recording
    env.video_enabled = True

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"Opponent type: POLICY (RandomCombatPolicy)")
    print(f"Opponent scale: 0.3")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run episodes
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            # Random action
            action = np.random.uniform(-0.5, 0.5, size=env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

            if step_count % 50 == 0:
                print(f"  Step {step_count}: reward={reward:.3f}")

        print(f"Episode {episode + 1} finished!")
        print(f"  Steps: {step_count}, Total reward: {episode_reward:.3f}")

    # Save video
    video_path = Path(__file__).parent / "videos" / "humanoid21_vs_policy_example.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving video to {video_path}...")
    success = env.save_video(str(video_path), fps=30)

    if success:
        print(f"Video saved successfully! ({video_path})")
    else:
        print("Failed to save video.")

    env.close()
    print("\nExample completed!")


if __name__ == "__main__":
    run_vs_policy_example()
