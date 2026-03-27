# CombatBench Environment Examples

This directory contains integration test scripts for all Humanoid21 environments.

Each example script runs a few episodes with random actions and saves a video to `examples/videos/`.

## Available Examples

| Example | Environment | Description |
|---------|-------------|-------------|
| `humanoid21_single_agent_example.py` | Humanoid21SingleAgentEnv | Single-agent training environment |
| `humanoid21_vs_frozen_example.py` | Humanoid21VsFrozenEnv | Opponent is frozen (immobile) |
| `humanoid21_vs_standing_example.py` | Humanoid21VsStandingEnv | Opponent stands (can be knocked down) |
| `humanoid21_vs_policy_example.py` | Humanoid21VsPolicyEnv | Opponent uses a policy (RandomCombatPolicy) |
| `humanoid21_non_fall_example.py` | Humanoid21NonFallEnv | Upright constraint enabled |
| `humanoid21_fall_example.py` | Humanoid21FallEnv | Fall detection enabled |
| `humanoid21_dual_agent_example.py` | Humanoid21DualAgentEnv | Both robots controlled |
| `humanoid21_match_example.py` | Humanoid21MatchEnv | Competition match environment |

## Running the Examples

### Run a Single Example

```bash
# From the combatbench root directory
python examples/humanoid21_single_agent_example.py
```

### Run All Examples

```bash
# From the combatbench root directory
python examples/humanoid21_single_agent_example.py
python examples/humanoid21_vs_frozen_example.py
python examples/humanoid21_vs_standing_example.py
python examples/humanoid21_vs_policy_example.py
python examples/humanoid21_non_fall_example.py
python examples/humanoid21_fall_example.py
python examples/humanoid21_dual_agent_example.py
python examples/humanoid21_match_example.py
```

### Run Using Python Module

```bash
# Run single agent example
python -m combatbench.examples.humanoid21_single_agent_example

# Run match example
python -m combatbench.examples.humanoid21_match_example
```

## Output

Each example script will:
1. Print environment information
2. Run 3 episodes with random actions
3. Save a video to `examples/videos/`
4. Print episode statistics

The videos are saved in MP4 format with 30 FPS.

## Example Output

```
============================================================
Humanoid21 Single Agent Environment Example
============================================================

Environment: Humanoid21SingleAgentEnv
Observation space: Box(-inf, inf, (127,), float32)
Action space: Box(-1.0, 1.0, (21,), float32)
Match duration: 10.0s
Control frequency: 20.0Hz
Opponent type: standing

--- Episode 1/3 ---
Initial obs shape: (127,)
Initial info keys: ['step', 'torso_position', 'opponent_position', ...]
  Step 50: reward=0.000, fallen={'robot_a': False, 'robot_b': False}
Episode 1 finished!
  Steps: 201
  Total reward: 0.000
  Final scores: {'robot_a': 100.0, 'robot_b': 100.0}
  Video frames: 300

Saving video to examples/videos/humanoid21_single_agent_example.mp4...
Video saved successfully!

============================================================
Example completed successfully!
============================================================
```

## Environment Features

### Single Agent Environments

- **Input**: Action as numpy array with shape (21,)
- **Output**: Observation as numpy array with shape (127,)
- **Opponent**: Controlled by hook (frozen, standing, or policy)

### Dual Agent Environments

- **Input**: Action as dict `{'robot_a': array(21,), 'robot_b': array(21,)}`
- **Output**: Observation as dict `{'robot_a_obs': array(127,), 'robot_b_obs': array(127,)}`
- **Control**: Both robots controlled independently

## Customization

You can modify the example scripts to:

- Change the match duration
- Use different policies (e.g., `RandomCombatPolicy`, `StandingCombatPolicy`)
- Adjust action ranges
- Enable/disable features like fall detection or upright constraints

For example:

```python
# Use a custom opponent policy
from combatbench.policy import RandomCombatPolicy

opponent = RandomCombatPolicy(scale=0.5, seed=123)
env = Humanoid21VsPolicyEnv(
    opponent_policy=opponent,
    match_duration=30.0,  # Longer match
    control_frequency=20.0,
)
```

## Troubleshooting

### Video Not Saving

If videos are not saved, ensure:
1. EGL is available for headless rendering
2. The `videos/` directory is writable
3. `render_mode="rgb_array"` is set

### Import Errors

If you get import errors, run the script from the `combatbench` root directory:

```bash
cd /path/to/combatbench
python examples/humanoid21_single_agent_example.py
```
