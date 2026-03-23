# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CombatBench is a MuJoCo-based humanoid robot combat simulation environment. It provides a Gymnasium-compatible RL environment where two 21-DOF humanoid robots fight each other. The project includes a physics engine, collision detection, scoring system, and baseline training implementations using Stable-Baselines3.

## Project Structure

- `assets/` - MuJoCo XML models, textures, meshes (arena: `battle_v1.xml`)
- `core/` - Core engine components
  - `physics.py` - MuJoCo physics wrapper (`PhysicsEngine`)
  - `humanoid_robot.py` - 21-DOF humanoid robot (`HumanoidRobot`)
  - `base_robot.py` - Abstract base class for robots
  - `collision.py` - Collision detection and hit judgment
  - `scoring.py` - HP-based scoring (100 HP initial, head=-3, torso=-1)
- `envs/` - Gymnasium environment wrapper
  - `combat_gym.py` - Main `CombatGymEnv` (127-dim obs, 21-dim action per robot)
  - `round_runner.py` - `RoundRunner` class for running complete rounds between two policies
- `policy/` - Policy interface and reference implementations
  - `base.py` - `BaseCombatPolicy` abstract interface
  - `random.py` - `RandomCombatPolicy` for testing
  - `standing.py` - `StandingCombatPolicy` (no movement)
- `baseline/sb3/` - Stable-Baselines3 PPO baseline
- `baseline/selfplay_hp/` - PyTorch PPO implementation for HP-only self-play
- `tools/` - Utilities and round runner (`run_round.py`)
- `docs/` - Rules, environment specs, robot details

## Common Commands

### Installation
```bash
pip install mujoco gymnasium numpy opencv-python imageio egl torch stable-baselines3 scipy
```

### Quick Start (Standing Policy)
```bash
# Run with no policies (both use StandingCombatPolicy - no movement)
python3 tools/run_round.py --duration 10 --video test.mp4

# Run with random policy
python3 tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy --duration 5 --video test.mp4
```

### Running Rounds (Evaluation & Video)

**Using the unified round runner CLI:**
```bash
# Run with no policies (both standing)
python tools/run_round.py --duration 10 --video test.mp4

# Run with Python module policies
python tools/run_round.py \
  --policy-a combatbench.policy.RandomCombatPolicy \
  --policy-b combatbench.policy.StandingCombatPolicy \
  --video match.mp4

# Run with parameters
python tools/run_round.py \
  --policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2&seed=42" \
  --duration 15 --video output.mp4

# Run with config file
python tools/run_round.py \
  --policy-a "@configs/policy_a.json" \
  --policy-b "@configs/policy_b.json" \
  --video match.mp4
```

**Using RoundRunner in Python code:**
```python
from combatbench.envs import RoundRunner
from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy

runner = RoundRunner(
    policy_a=RandomCombatPolicy(scale=0.1),
    policy_b=StandingCombatPolicy(),
    match_duration=30.0,
    render_mode="rgb_array",
)
result = runner.run(save_video_path="output.mp4")
print(f"Winner: {result.winner}, Steps: {result.steps}")
```

## Policy Specification Format

All policies use a unified specification format that supports constructor parameters.

### Formats

1. **Python module path** (no parameters):
   ```
   combatbench.policy.RandomCombatPolicy
   ```

2. **Python module path with parameters** (query string):
   ```
   combatbench.policy.RandomCombatPolicy?scale=0.2&seed=42
   ```

3. **Python file with class**:
   ```
   path/to/policy.py:MyPolicy
   path/to/policy.py:MyPolicy?param=value
   ```

4. **Config file** (JSON):
   ```bash
   @policy_config.json
   ```

   Config file format:
   ```json
   {
     "type": "combatbench.policy.RandomCombatPolicy",
     "params": {
       "scale": 0.2,
       "seed": 42
     }
   }
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

class MyPolicy(BaseCombatPolicy):
    def __init__(self, observation_space=None, action_space=None, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        # Your initialization

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """Return action array with shape (21,), values in [-1, 1]"""
        # Your action computation
        return action

    def reset(self) -> None:
        """Reset internal state (optional)"""
        pass
```

### RoundResult Dataclass

Returned by `RoundRunner.run()`:

```python
@dataclass
class RoundResult:
    steps: int                    # Total steps taken
    end_reason: str               # Why round ended
    winner: Optional[str]         # 'robot_a', 'robot_b', or 'draw'
    scores: Dict[str, float]      # Final HP for both robots
    initial_scores: Dict[str, float]  # Initial HP (usually 100)
    damage_dealt: Dict[str, float]    # Total damage by each robot
    total_reward: Dict[str, float]    # Accumulated shaped reward
    video_frames: int             # Number of video frames captured
```

## Documentation

- [`docs/ROBOT.md`](docs/ROBOT.md) - 21-DOF robot design rationale
- [`docs/RULE.md`](docs/RULE.md) - Combat rules and HP system
- [`docs/OBSERVATION.md`](docs/OBSERVATION.md) - Observation space design
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) - Arena and simulation environment

## Important Notes
