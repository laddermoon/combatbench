"""
Test entry script: Run a complete Episode (30s) without policy
"""

import sys
import os
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import numpy as np

from envs.combat_gym import CombatGymEnv

def main():
    print("=" * 60)
    print("CombatBench 21DOF Test Run (No Policy)")
    print("=" * 60)

    # 1. Create environment, enable rgb_array render mode
    env = CombatGymEnv(
        render_mode="rgb_array", 
        match_duration=30.0,
        control_frequency=20
    )
    
    # 2. Reset environment
    obs, info = env.reset()
    print(f"Environment reset, Initial position: {info['positions']}")
    print(f"Initial HP: {info['scores']}")
    print("-" * 60)

    max_steps = int(env.match_duration * env.control_frequency)
    step_count = 0
    
    # No policy: Keep all joints at zero (upright) or apply tiny random perturbations
    # Here we apply tiny noise
    action_dim = env.robot_a.ACTION_DIM
    
    print(f"Starting run, estimated max steps: {max_steps} steps (30seconds)")
    
    while True:
        # Apply tiny random force so robots are not completely stiff
        action = {
            'robot_a': np.random.uniform(-0.1, 0.1, action_dim),
            'robot_b': np.random.uniform(-0.1, 0.1, action_dim)
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # Simple status output (every 100 steps)
        if step_count % 100 == 0:
            print(f"Step {step_count:03d} - HP: {info['scores']}, " 
                  f"Distance: {np.linalg.norm(info['positions']['robot_a'] - info['positions']['robot_b']):.2f}m")
            
        # Check for valid attacks (HP deduction)
        if info['hit_records']['robot_a']:
            print(f"[Step {step_count}] 🔴 Robot A hit! Details: {info['hit_records']['robot_a']}")
        if info['hit_records']['robot_b']:
            print(f"[Step {step_count}] 🔵 Robot B hit! Details: {info['hit_records']['robot_b']}")

        if terminated or truncated:
            break

    print("-" * 60)
    print(f"Episode ended. Total steps: {step_count}")
    print(f"Final result: {info['end_reason']}")
    print(f"Final HP: {info['scores']}")
    
    # 3. Save video
    print("\nSaving video...")
    video_path = Path(__file__).parent / 'no_policy_test.mp4'
    env.save_video(str(video_path), fps=env.video_sample_frequency)
    print(f"Video saved to: {video_path}")
    print(f"Total recorded frames: {len(env.get_video_buffer())}")

    env.close()

if __name__ == '__main__':
    main()
