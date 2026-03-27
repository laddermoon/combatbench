"""
Humanoid21 Dual Agent Environment Example

This script demonstrates the usage of Humanoid21DualAgentEnv.
Both robots are controlled independently.
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
from combatbench.envs.humanoid21 import Humanoid21DualAgentEnv


def run_dual_agent_example():
    """Run dual agent environment example with video recording."""

    print("=" * 60)
    print("Humanoid21 Dual Agent Environment Example")
    print("=" * 60)

    # Create environment with video recording
    env = Humanoid21DualAgentEnv(
        render_mode="rgb_array",
        match_duration=10.0,
        control_frequency=20.0,
        enable_fall_detection=True,
    )

    # Enable video recording
    env.video_enabled = True

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"Mode: Dual agent (both robots controlled)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run episodes
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()
        print(f"Initial obs keys: {list(obs.keys())}")

        episode_reward_a = 0.0
        episode_reward_b = 0.0
        step_count = 0
        done = False

        while not done:
            # Random actions for both robots
            action = {
                'robot_a': np.random.uniform(-0.5, 0.5, size=21),
                'robot_b': np.random.uniform(-0.5, 0.5, size=21),
            }

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward_a += reward['robot_a']
            episode_reward_b += reward['robot_b']
            step_count += 1

            done = terminated or truncated

            if step_count % 50 == 0:
                fallen = info.get('fallen', {})
                print(f"  Step {step_count}: reward_a={reward['robot_a']:.3f}, "
                      f"reward_b={reward['robot_b']:.3f}, fallen={fallen}")

        print(f"Episode {episode + 1} finished!")
        print(f"  Steps: {step_count}")
        print(f"  Total reward A: {episode_reward_a:.3f}")
        print(f"  Total reward B: {episode_reward_b:.3f}")
        print(f"  Final fallen status: {info.get('fallen', {})}")

    # Save video
    video_path = Path(__file__).parent / "videos" / "humanoid21_dual_agent_example.mp4"
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
    run_dual_agent_example()
