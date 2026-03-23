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
python tools/run_round.py --duration 10 --video test.mp4

# Run with random policy
python tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy --duration 5 --video test.mp4
```

### Environment Validation
```bash
python3 -m combatbench.baseline.sb3.validate_env
```

### Training Commands

**Phase 1: Standing Pre-training**
```bash
python3 -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 1000000 \
  --run-name stand_v1
```

**Phase 2: Fight Fine-tuning** (initialized from standing model)
```bash
python3 -m combatbench.baseline.sb3.train \
  --phase fight \
  --timesteps 2000000 \
  --run-name fight_v1 \
  --init-model combatbench/baseline/sb3/runs/stand_v1/model_final.zip
```

**Phase 3: Attacker Approach** (requires opponent model)
```bash
python3 -m combatbench.baseline.sb3.train \
  --phase fight_attacker_approach \
  --timesteps 2000000 \
  --run-name attacker_approach_v1 \
  --opponent-model combatbench/baseline/sb3/runs/stand_v1/model_final.zip
```

**Phase 4: Attacker Combat**
```bash
python3 -m combatbench.baseline.sb3.train \
  --phase fight_attacker \
  --timesteps 2000000 \
  --run-name attacker_v1 \
  --opponent-model combatbench/baseline/sb3/runs/stand_v1/model_final.zip \
  --attacker-base-model combatbench/baseline/sb3/runs/attacker_approach_v1/model_final.zip
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

# Run with SB3 model (unified format)
python tools/run_round.py \
  --policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=runs/stand_v1/model_final.zip" \
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

## Architecture Details

### Observation Space (127 dims per robot)
- Joint positions (21) + velocities (21) = 42
- Root state: height (1) + local orientation (6) + linear/angular velocity (6) = 13
- Tactile: feet contact (2) + external forces (6) = 8
- Opponent observation (64): relative position/velocity, orientation, 9 keypoints (pos+vel)

See `HumanoidRobot.OBSERVATION_SLICES` for exact indices.

### Action Space (21 dims per robot)
Controlled joints:
- Abdomen: `abdomen_z`, `abdomen_y`, `abdomen_x`
- Right leg: `hip_x_right`, `hip_z_right`, `hip_y_right`, `knee_right`, `ankle_y_right`, `ankle_x_right`
- Left leg: `hip_x_left`, `hip_z_left`, `hip_y_left`, `knee_left`, `ankle_y_left`, `ankle_x_left`
- Right arm: `shoulder1_right`, `shoulder2_right`, `elbow_right`
- Left arm: `shoulder1_left`, `shoulder2_left`, `elbow_left`

Actions are normalized [-1, 1] and scaled by `action_scale` before PD control.

### Robot Naming Conventions
- Robot A (red): uses `_red` suffix in MuJoCo XML
- Robot B (blue): uses `_blue` suffix in MuJoCo XML
- Code uses `robot_a` / `robot_b` IDs

### Physics Configuration
- Default timestep: 0.002s (500Hz)
- Control frequency: 20Hz (50 physics steps per action)
- Match duration: 30s (default) or 10-15s for training
- Video FPS: 30 (default)

### Combat Rules
- Initial HP: 100 per robot
- Attacking parts: hand, forearm, elbow, upper arm, foot, shin, knee, thigh
- Damage targets: head (-3), torso (-1)
- Match ends when HP reaches 0 or time limit expires

### Training Phases & Reward Configurations

See `baseline/sb3/rewards.py` for `RewardConfig` dataclass:
- `STANDING_REWARD_CONFIG` - emphasizes height, uprightness, feet contact
- `FIGHT_REWARD_CONFIG` - adds distance, facing, damage rewards
- `ATTACKER_APPROACH_REWARD_CONFIG` - approach curriculum, progress rewards
- `ATTACKER_REWARD_CONFIG` - full attacker with engagement bonuses

### Controller Configuration

The environment uses a **PD controller with reference positions and action scaling**:
- Reference positions: nominal joint pose (default: standing pose)
- Action scale: per-joint multipliers for how much [-1,1] action affects joints
- Controller gains: `kp` (proportional), `kd` (derivative)

Key functions:
- `configure_base_env_for_stand()` - standing configuration
- `configure_base_env_for_fight()` - symmetric fight configuration
- `configure_base_env_for_fight_attacker()` - attacker vs standing opponent

Action interpretation:
- Direct: `target_pos = reference + action_scale * action`
- Attacker base residual: `target_pos = base_action + residual_scale * residual_action`

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

## Important Notes

- Always set `MUJOCO_GL=egl` for headless rendering (GPU server)
- The project uses both SB3 and custom PyTorch PPO implementations
- Training follows a curriculum: stand → fight → attacker_approach → attacker
- Model checkpoints are saved to `baseline/sb3/runs/<run_name>/`
- Video rendering requires EGL; set environment variables before importing mujoco
