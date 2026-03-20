"""
Test entry script: Run a complete Episode (30s) without policy
"""

import sys
import os
os.environ['MUJOCO_GL'] = 'egl'

import argparse
from pathlib import Path
import numpy as np

from envs.combat_gym import CombatGymEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Run combat simulation without policy")
    parser.add_argument("--match-duration", type=float, default=30.0, help="Match duration in seconds")
    parser.add_argument("--control-frequency", type=int, default=20, help="Control frequency (Hz)")
    parser.add_argument("--non-fall-mode", action="store_true", help="Enable non-fall mode (clamp root pitch/roll)")
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=15.0, help="Pitch limit in degrees for non-fall mode")
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=10.0, help="Roll limit in degrees for non-fall mode")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: no_policy_test.mp4 or no_policy_test_nonfall.mp4)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("CombatBench 21DOF Test Run (No Policy)")
    if args.non_fall_mode:
        print(f"Non-fall mode: ENABLED (pitch: ±{args.non_fall_pitch_limit_deg}°, roll: ±{args.non_fall_roll_limit_deg}°)")
    else:
        print("Non-fall mode: DISABLED")
    print("=" * 60)

    # 1. Create environment, enable rgb_array render mode
    env = CombatGymEnv(
        render_mode="rgb_array", 
        match_duration=args.match_duration,
        control_frequency=args.control_frequency,
        non_fall_mode=args.non_fall_mode,
        non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
        non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
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
    if args.output:
        video_path = Path(args.output)
    else:
        filename = 'no_policy_test_nonfall.mp4' if args.non_fall_mode else 'no_policy_test.mp4'
        video_path = Path(__file__).parent / filename
    env.save_video(str(video_path), fps=env.video_sample_frequency)
    print(f"Video saved to: {video_path}")
    print(f"Total recorded frames: {len(env.get_video_buffer())}")

    env.close()

if __name__ == '__main__':
    main()
