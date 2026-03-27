"""
Humanoid21 Single Agent Environment Example

This script demonstrates the usage of Humanoid21SingleAgentEnv.
It runs a few episodes with random actions and saves a video.
"""

import os
import sys
from pathlib import Path

# Set headless render mode
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Add parent directory to path
# The script is in things/combatbench/examples/, so we need to add things/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from combatbench.envs.humanoid21 import Humanoid21SingleAgentEnv


def run_single_agent_example():
    """Run single agent environment example with video recording."""

    print("=" * 60)
    print("Humanoid21 Single Agent Environment Example")
    print("=" * 60)

    # Create environment with video recording
    env = Humanoid21SingleAgentEnv(
        render_mode="rgb_array",
        match_duration=10.0,  # Short duration for demo
        control_frequency=20.0,
        opponent_type='standing',  # Opponent stands still
        enable_fall_detection=True,
    )

    # Enable video recording
    env.video_enabled = True

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Match duration: {env._env.match_duration}s")
    print(f"Control frequency: {env._env.control_frequency}Hz")
    print(f"Opponent type: {env.opponent_type}")

    # Run multiple episodes
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()
        print(f"Initial obs shape: {obs.shape}")
        print(f"Initial info keys: {list(info.keys())}")

        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            # Random action
            action = np.random.uniform(-1, 1, size=env.action_space.shape)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Check if episode is done
            done = terminated or truncated

            # Print progress every 50 steps
            if step_count % 50 == 0:
                fallen = info.get('fallen', {})
                scores = info.get('scores', {})
                print(f"  Step {step_count}: reward={reward:.3f}, "
                      f"fallen={fallen}, scores={scores}")

        print(f"Episode {episode + 1} finished!")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Final scores: {info.get('scores', {})}")
        print(f"  Video frames: {len(env.get_video_buffer())}")

    # Save video
    video_path = Path(__file__).parent / "videos" / "humanoid21_single_agent_example.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving video to {video_path}...")
    success = env.save_video(str(video_path), fps=30)

    if success:
        print(f"Video saved successfully! ({video_path})")
    else:
        print("Failed to save video.")

    env.close()
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_single_agent_example()
