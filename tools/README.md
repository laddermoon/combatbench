# CombatBench Tools

This directory contains utility scripts to aid in running, evaluating, and working with the CombatBench environment.

## 1. Round Runner (`run_round.py`)

A unified script to run combat rounds between two policies. All policies are loaded using a consistent specification format that supports constructor parameters.

### Policy Specification Formats

#### 1. Python Module Paths
```bash
# Simple (no parameters)
--policy-a combatbench.policy.RandomCombatPolicy

# With parameters (query string format)
--policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2&seed=42"

# SB3 model
--policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=model.zip&device=cuda"
```

#### 2. Python File Paths
```bash
# With class name
--policy-a "path/to/policy.py:MyPolicy"

# With class name and parameters
--policy-a "path/to/policy.py:MyPolicy?scale=0.5"
```

#### 3. Config Files (JSON)
```bash
--policy-a "@policy_config.json"
```

Config file format (`policy_config.json`):
```json
{
  "type": "combatbench.policy.RandomCombatPolicy",
  "params": {
    "scale": 0.2,
    "seed": 42
  }
}
```

Or for SB3 models:
```json
{
  "type": "combatbench.baseline.sb3.policies.SB3CombatPolicy",
  "params": {
    "model_path": "runs/stand_v1/model_final.zip",
    "device": "cuda"
  }
}
```

#### 4. Default (Standing Policy)
```bash
# If --policy-a is omitted, uses StandingCombatPolicy
python tools/run_round.py --duration 10 --video test.mp4
```

### Parameter Type Support

The query string format supports automatic type conversion:
- **Numbers**: `?scale=0.5`, `?count=10`
- **Booleans**: `?enabled=true`, `?debug=false`
- **Strings**: `?name=my_policy`, `?model_path=model.zip`
- **JSON values**: For complex types, use JSON-encoded values:
  - Lists: `?list=[1,2,3]`
  - Objects: `?config={"key":"value"}`
  - Null: `?optional=null`

### Using the RoundRunner Class

You can import the `RoundRunner` class from `combatbench.envs` into your own evaluation or training loops:

```python
from combatbench.envs import RoundRunner
from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy

# Create runner with two policies
runner = RoundRunner(
    policy_a=RandomCombatPolicy(scale=0.1),
    policy_b=StandingCombatPolicy(),
    match_duration=30.0,  # seconds
    render_mode="rgb_array",
)

# Run the round and optionally save the output video
result = runner.run(save_video_path="round_output.mp4")

# Access results via RoundResult dataclass
print(f"Winner: {result.winner}")
print(f"Steps: {result.steps}")
print(f"Final HP - Red: {result.scores['robot_a']}, Blue: {result.scores['robot_b']}")
print(f"Damage dealt - A: {result.damage_dealt['robot_a']}, B: {result.damage_dealt['robot_b']}")
```

### CLI Usage Examples

Run with no policies (both use StandingCombatPolicy):
```bash
python tools/run_round.py --duration 10 --video test.mp4
```

Run with Python module policies:
```bash
python tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy \
                         --policy-b combatbench.policy.StandingCombatPolicy
```

Run with SB3 model:
```bash
python tools/run_round.py \
  --policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=runs/stand_v1/model_final.zip" \
  --video match.mp4
```

Run with parameters:
```bash
# Random policy with custom scale
python tools/run_round.py --policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2"

# Custom policy with multiple parameters
python tools/run_round.py --policy-a "mypolicy.MyPolicy?model_path=model.zip&noise=true"
```

Run with config file:
```bash
python tools/run_round.py --policy-a "@configs/policy_a.json" --policy-b "@configs/policy_b.json"
```

**CLI Arguments:**

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--policy-a` | `--model-a` | Policy for robot A (red) | `StandingCombatPolicy` |
| `--policy-b` | `--model-b` | Policy for robot B (blue) | `StandingCombatPolicy` |
| `--duration` | `--match-duration` | Round duration in seconds | `30.0` |
| `--control-frequency` | `--fps` | Control frequency in Hz | `20` |
| `--initial-distance` | | Initial distance between robots (m) | `2.0` |
| `--phase` | | Training phase for controller config | `None` |
| `--non-fall-mode` | | Enable orientation clamping | `False` |
| `--non-fall-pitch-limit-deg` | | Pitch limit for non-fall mode | `15.0` |
| `--non-fall-roll-limit-deg` | | Roll limit for non-fall mode | `10.0` |
| `--damage-scale` | | Damage scaling factor | `100.0` |
| `--video` | `--output` | Path to save video | `None` (no video) |
| `--device` | | Device for policy inference | `auto` |
| `--quiet` | `-q` | Suppress progress output | `False` |

### Policy Interface

Policies must implement:
```python
def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
    """Return action array with shape (21,), values in [-1, 1]"""
    pass

def reset(self) -> None:
    """Reset internal state at episode start (optional)"""
    pass
```

### RoundResult Fields

- `steps`: Total number of steps taken
- `end_reason`: Reason why the round ended
- `winner`: Which robot won ('robot_a', 'robot_b', or 'draw')
- `scores`: Final HP scores for both robots
- `initial_scores`: Initial HP scores (usually 100 each)
- `damage_dealt`: Total damage dealt by each robot
- `total_reward`: Total shaped reward accumulated
- `video_frames`: Number of video frames captured
