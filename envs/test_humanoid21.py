import os
import sys
from pathlib import Path

# Set correct paths
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from envs.humanoid21 import make_env
import numpy as np

def test_humanoid21_env():
    print("Testing Humanoid21 new architecture...")
    
    # 1. Create environment
    env = make_env(
        match_duration=2.0,  # Run a short 2-second match for testing
        non_fall_mode=True
    )
    
    # 2. Reset
    obs, info = env.reset()
    print("Reset successful.")
    print(f"Observation shapes: Robot A={obs['robot_a_obs'].shape}, Robot B={obs['robot_b_obs'].shape}")
    print(f"Initial Health: {info['health']}")
    
    # 3. Step loop
    done = False
    step_count = 0
    while not done:
        # random actions
        action = {
            "robot_a": env.action_space.spaces["robot_a"].sample(),
            "robot_b": env.action_space.spaces["robot_b"].sample(),
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
    print(f"Episode finished after {step_count} steps.")
    print(f"Termination reasons: {info.get('termination_reasons', [])}")
    print(f"Final Health: {info['health']}")
    print(f"Winner: {info.get('winner')}")
    
    env.close()
    print("Test passed!")

if __name__ == "__main__":
    test_humanoid21_env()
