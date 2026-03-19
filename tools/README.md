# CombatBench Tools

This directory contains utility scripts to aid in running, evaluating, and working with the CombatBench environment.

## 1. Match Runner (`run_match.py`)

A utility tool designed to simulate a single head-to-head match (episode) between two policies. 

### Usage in Code

You can import the `MatchRunner` class into your own evaluation or training loops to easily orchestrate a fight:

```python
from tools.run_match import MatchRunner

# Assume policy_a and policy_b are your trained policy objects
# Policies must implement: `act(obs, info)` and `reset()`
runner = MatchRunner(
    policy_a=my_red_policy,
    policy_b=my_blue_policy,
    match_duration=30.0 # seconds
)

# Run the match and optionally save the output video
result = runner.run(save_video_path="match_output.mp4")

print(f"Winner: {result['end_reason']}")
print(f"Final HP - Red: {result['scores']['robot_a']}, Blue: {result['scores']['robot_b']}")
```

### CLI Usage

You can also run it directly via the command line to test the simulation using random actions (dummy policies):

```bash
python tools/run_match.py --video output_video.mp4 --duration 10.0
```

**Arguments:**
- `--video`: Path where the MP4 replay will be saved (e.g. `match.mp4`). If omitted, no video is saved.
- `--duration`: Maximum match time limit in seconds (default is 30.0).
