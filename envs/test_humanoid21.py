import os
import sys
from pathlib import Path

# Set correct paths
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from envs.humanoid21 import make_env

def test_humanoid21_env():
    print("Testing Humanoid21 new architecture...")
    
    runtime = make_env(
        match_duration=2.0,  # Run a short 2-second match for testing
        non_fall_mode=True
    )
    
    result = runtime.reset()
    obs = result["obs"]
    info = result["info"]
    print("Reset successful.")
    print(f"Observation shapes: Robot A={obs['robot_a'].shape}, Robot B={obs['robot_b'].shape}")
    print(f"Initial Health: {info['shared']['health']}")
    
    done = False
    step_count = 0
    while not done:
        action_a = runtime.action_space.spaces["robot_a"].sample()
        action_b = runtime.action_space.spaces["robot_b"].sample()
        
        result = runtime.step(action_a, action_b)
        obs = result["obs"]
        info = result["info"]
        terminated = result["terminated"]
        truncated = result["truncated"]
        done = terminated or truncated
        step_count += 1
        
    print(f"Episode finished after {step_count} steps.")
    print(f"Termination reasons: {info['shared'].get('termination_reasons', [])}")
    print(f"Final Health: {info['shared']['health']}")
    print(f"Winner: {info['shared'].get('winner')}")
    
    runtime.close()
    print("Test passed!")

if __name__ == "__main__":
    test_humanoid21_env()
