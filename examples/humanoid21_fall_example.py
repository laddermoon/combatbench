"""
Humanoid21 Fall Environment Example

This script demonstrates the usage of Humanoid21FallEnv.
This environment allows robots to fall and tracks fall status.
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
from combatbench.envs.humanoid21 import Humanoid21FallEnv


def run_fall_example():
    """Run fall environment example with video recording."""

    print("=" * 60)
    print("Humanoid21 Fall Environment Example")
    print("=" * 60)

    # Create environment with video recording
    env = Humanoid21FallEnv(
        render_mode="rgb_array",
        match_duration=10.0,
        control_frequency=20.0,
    )

    # Enable video recording
    env.video_enabled = True

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"Features: Fall detection enabled (robots can fall)")
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
        any_fell = False

        while not done:
            # Random action
            action = np.random.uniform(-0.5, 0.5, size=env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

            # Check if any robot fell
            fallen = info.get('fallen', {})
            if any(fallen.values()):
                any_fell = True

            if step_count % 50 == 0:
                print(f"  Step {step_count}: reward={reward:.3f}, fallen={fallen}")

        print(f"Episode {episode + 1} finished!")
        print(f"  Steps: {step_count}, Total reward: {episode_reward:.3f}")
        print(f"  Any robots fell? {any_fell}")
        print(f"  Final fallen status: {info.get('fallen', {})}")

    # Save video
    video_path = Path(__file__).parent / "videos" / "humanoid21_fall_example.mp4"
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
    run_fall_example()
