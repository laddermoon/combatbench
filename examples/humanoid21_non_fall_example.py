"""
Humanoid21 Non-Fall Environment Example

This script demonstrates the usage of Humanoid21NonFallEnv.
This environment has upright constraints to prevent robots from falling.
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
from combatbench.envs.humanoid21 import Humanoid21NonFallEnv


def run_non_fall_example():
    """Run non-fall environment example with video recording."""

    print("=" * 60)
    print("Humanoid21 Non-Fall Environment Example")
    print("=" * 60)

    # Create environment with video recording
    env = Humanoid21NonFallEnv(
        render_mode="rgb_array",
        match_duration=10.0,
        control_frequency=20.0,
        enable_fall_detection=True,  # Still track fall status
    )

    # Enable video recording
    env.video_enabled = True

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"Features: Upright constraint enabled (robots won't fall)")
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
            # Random action (can be more aggressive since robots won't fall)
            action = np.random.uniform(-0.8, 0.8, size=env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

            if step_count % 50 == 0:
                fallen = info.get('fallen', {})
                print(f"  Step {step_count}: reward={reward:.3f}, fallen={fallen}")

        print(f"Episode {episode + 1} finished!")
        print(f"  Steps: {step_count}, Total reward: {episode_reward:.3f}")
        print(f"  Any robots fell? {any(info.get('fallen', {}).values())}")

    # Save video
    video_path = Path(__file__).parent / "videos" / "humanoid21_non_fall_example.mp4"
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
    run_non_fall_example()
