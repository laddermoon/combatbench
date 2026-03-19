import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Set headless render mode if EGL is available
os.environ['MUJOCO_GL'] = 'egl'

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.combat_gym import CombatGymEnv

class MatchRunner:
    """
    A utility class to run a single match between two policies.
    """
    def __init__(self, policy_a=None, policy_b=None, render_mode="rgb_array", match_duration=30.0, control_frequency=20):
        self.policy_a = policy_a
        self.policy_b = policy_b
        
        self.env = CombatGymEnv(
            render_mode=render_mode,
            match_duration=match_duration,
            control_frequency=control_frequency
        )

    def run(self, save_video_path=None):
        """
        Execute one complete episode/match.
        Returns the final result dict.
        """
        obs, info = self.env.reset()
        print("=" * 60)
        print("CombatBench Match Started")
        print(f"Initial position: {info['positions']}")
        print(f"Initial HP: {info['scores']}")
        print("=" * 60)

        step_count = 0
        action_dim = self.env.robot_a.ACTION_DIM

        # Call policy reset if available
        if hasattr(self.policy_a, 'reset'):
            self.policy_a.reset()
        if hasattr(self.policy_b, 'reset'):
            self.policy_b.reset()

        while True:
            # Get actions from policies, fallback to zero actions if no policy provided
            act_a = self.policy_a.act(obs['robot_a_obs'], info) if self.policy_a else np.zeros(action_dim)
            act_b = self.policy_b.act(obs['robot_b_obs'], info) if self.policy_b else np.zeros(action_dim)

            action = {
                'robot_a': act_a,
                'robot_b': act_b
            }

            obs, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1

            # Simple logging
            if step_count % 100 == 0:
                print(f"Step {step_count:03d} - HP: {info['scores']}, "
                      f"Distance: {np.linalg.norm(info['positions']['robot_a'] - info['positions']['robot_b']):.2f}m")

            if info['hit_records']['robot_a']:
                print(f"[Step {step_count}] 🔴 Robot A hit! Details: {info['hit_records']['robot_a']}")
            if info['hit_records']['robot_b']:
                print(f"[Step {step_count}] 🔵 Robot B hit! Details: {info['hit_records']['robot_b']}")

            if terminated or truncated:
                break

        print("-" * 60)
        print(f"Match ended. Total steps: {step_count}")
        print(f"Final result: {info['end_reason']}")
        print(f"Final HP: {info['scores']}")
        print("-" * 60)

        if save_video_path:
            print("\nSaving match video...")
            self.env.save_video(str(save_video_path), fps=self.env.video_sample_frequency)
            print(f"Video saved to: {save_video_path}")

        self.env.close()

        # Build return result
        result = {
            'steps': step_count,
            'end_reason': info['end_reason'],
            'scores': info['scores'],
            'positions': info['positions']
        }
        return result

# Simple dummy policy for testing CLI
class DummyPolicy:
    def __init__(self, action_dim=21, noise_scale=0.1):
        self.action_dim = action_dim
        self.noise_scale = noise_scale
        
    def act(self, obs, info=None):
        return np.random.uniform(-self.noise_scale, self.noise_scale, self.action_dim)

    def reset(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a CombatBench match between two policies.")
    parser.add_argument('--video', type=str, default=None, help='Path to save the match video (e.g., match.mp4)')
    parser.add_argument('--duration', type=float, default=30.0, help='Match duration in seconds')
    
    args = parser.parse_args()

    # Create dummy policies for CLI standalone run
    # In real usage, this class should be imported and policies injected into constructor
    policy_a = DummyPolicy()
    policy_b = DummyPolicy()

    runner = MatchRunner(
        policy_a=policy_a, 
        policy_b=policy_b,
        match_duration=args.duration
    )
    
    runner.run(save_video_path=args.video)
