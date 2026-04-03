# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CombatBench is a MuJoCo-based humanoid robot combat simulation environment. It provides a Gymnasium-compatible RL environment where two 21-DOF humanoid robots fight each other. The project includes a physics engine, collision detection, scoring system, and baseline training implementations using Stable-Baselines3.

## Project Structure

- `assets/` - MuJoCo XML models, textures, meshes (arena: `battle_v1.xml`)
- `envs/` - Environment implementations
  - `framework/` - Core framework interfaces (BasePlugin, SimContext, etc.)
  - `humanoid21/` - 21-DOF humanoid robot environment
    - `simulator.py` - Main simulator (`Humanoid21Simulator`)
    - `plugins.py` - Combat plugins (scoring, non-fall constraint, frozen robot)
    - `observer_plugins.py` - Gymnasium observation plugin
    - `disturbance_plugins.py` - External disturbance plugins
    - `run_round.py` - `RoundRunner` class for running complete rounds
    - `DATASPEC.md` - Data interface specification
    - `OBSERVATION_zh.md` - Observation space documentation (96-dim)
- `policy/` - Policy interface and reference implementations
  - `base.py` - `BaseCombatPolicy` abstract interface
  - `load_util.py` - Policy loading utility (`load_policy()`)
  - `random/` - RandomCombatPolicy directory (with policy.py)
  - `standing/` - StandingCombatPolicy directory (with policy.py)
- `tools/` - Utilities (`run_round.py` for CLI usage)
- `docs/` - Rules, environment specs, robot details
- `baseline/` - Training implementations (Stable-Baselines3, GRPO)

## Common Commands

### Installation
```bash
pip install mujoco gymnasium numpy opencv-python imageio egl torch stable-baselines3 scipy
```

### Quick Start (Standing Policy)
```bash
# Run with no policies (both use StandingCombatPolicy - no movement)
python3 envs/humanoid21/run_round.py --duration 10 --video test.mp4

# Run with random policy
python3 envs/humanoid21/run_round.py --policy-a random --duration 5 --video test.mp4
```

### Running Rounds (Evaluation & Video)

**Using the unified round runner CLI:**
```bash
# Run with no policies (both standing)
python envs/humanoid21/run_round.py --duration 10 --video test.mp4

# Run with policy directories
python envs/humanoid21/run_round.py \
  --policy-a random \
  --policy-b standing \
  --video match.mp4

# Run with parameters
python envs/humanoid21/run_round.py \
  --policy-a "random?scale=0.2&seed=42" \
  --duration 15 --video output.mp4
```

**Using RoundRunner in Python code:**
```python
from combatbench.envs.humanoid21 import Humanoid21Simulator, RoundRunner
from combatbench.policy import load_policy

policy_a = load_policy("random?scale=0.1&seed=42")
policy_b = load_policy("standing")

runner = RoundRunner(
    simulator=Humanoid21Simulator(),
    policy_a=policy_a,
    policy_b=policy_b,
    match_duration=30.0,
    render_mode="rgb_array",
)
result = runner.run(save_video_path="output.mp4")
print(f"Winner: {result.winner}, Steps: {result.steps}")
```

## Policy Loading

All policies use a directory-based structure with `policy.py` containing a `BaseCombatPolicy` implementation.

### Directory Structure

Each policy is a directory with:
- **`policy.py`** (required) - Contains a class inheriting `BaseCombatPolicy`
- **`requirements.txt`** (optional) - Additional dependencies

Example:
```
my_policy/
├── policy.py            # Must contain BaseCombatPolicy implementation
└── requirements.txt     # Optional dependencies
```

### Loading Formats

1. **Directory path** (auto-detects first BaseCombatPolicy):
   ```python
   policy = load_policy("my_policy")
   ```

2. **Module path with class**:
   ```python
   policy = load_policy("my_policy.policy.MyCombatPolicy")
   ```

3. **With parameters** (query string):
   ```python
   policy = load_policy("my_policy?scale=0.2&seed=42")
   ```

### Parameter Type Support

- Numbers: `?scale=0.5`, `?count=10`
- Booleans: `?enabled=true`
- Strings: `?model_path=model.zip`
- JSON values: `?list=[1,2,3]`, `?config={"key":"value"}`

### Policy Interface

All policies must inherit from `BaseCombatPolicy` and implement:

```python
from combatbench.policy import BaseCombatPolicy
import numpy as np

class MyCombatPolicy(BaseCombatPolicy):
    ACTION_DIM = 21  # Action space dimension

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your initialization

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """
        Return action array with shape (21,), values in [-1, 1]

        Args:
            obs: 96-dimensional observation array
            info: Optional environment info dict
        """
        # Your action computation
        return action

    def reset(self) -> None:
        """Reset internal state (optional)"""
        pass
```

## Observation Space (96-dim)

The observation space consists of 4 modules:

1. **Proprioception** (42维): `joint_pos_norm` (21) + `joint_vel_norm` (21)
2. **Root State** (13维): height (1) + local_orientation (6) + local linear_vel (3) + local angular_vel (3)
3. **Tactile** (2维): `feet_forces` - force sensors on both feet
4. **Opponent** (39维):
   - Basic pose (7): opponent root position (3) + facing direction (3) + height (1)
   - Keypoint positions (18): 6 keypoints × 3 coordinates (local frame)
   - Keypoint velocities (14): 6 keypoints × 3 velocities - 4 (FaceVector compressed)

See `envs/humanoid21/OBSERVATION_zh.md` for detailed documentation.

## Data Interface Specification

All data access follows the structured format defined in `DATASPEC.md`:

```python
# Core state (robot positions, velocities, joints)
core_state = sim.accessor.get_core_state()
# Returns: {'robot_a': {...}, 'robot_b': {...}}

# Derived state (observations, contacts, etc.)
derived_state = sim.accessor.get_derived_state()
# Returns: {'robot_a': {'observation': np.ndarray(96), ...}, ...}

# Static data (robot info, normalization params)
static_data = sim.accessor.get_static_data()
```

## Documentation

- [`docs/ROBOT.md`](docs/ROBOT.md) - 21-DOF robot design rationale
- [`docs/RULE.md`](docs/RULE.md) - Combat rules and HP system
- [`docs/OBSERVATION.md`](docs/OBSERVATION.md) - Observation space design
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) - Arena and simulation environment
- [`envs/humanoid21/DATASPEC.md`](envs/humanoid21/DATASPEC.md) - Data interface specification
- [`envs/humanoid21/OBSERVATION_zh.md`](envs/humanoid21/OBSERVATION_zh.md) - 96-dim observation details
- [`policy/README.md`](policy/README.md) - Policy implementation guide

## Important Notes

- **Observation dimension**: 96 (not 127 - old version)
- **Policy structure**: Directory-based with `policy.py` (not single-file policies)
- **Data format**: Structured by `robot_id` (robot_a/robot_b), not flat arrays
- **Plugin system**: Uses `SimContext` with accessor/mutator pattern
