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
- `baseline/sb3/` - Stable-Baselines3 PPO baseline
- `baseline/selfplay_hp/` - PyTorch PPO implementation for HP-only self-play
- `tools/` - Utilities and match runner (`run_match.py`)
- `docs/` - Rules, environment specs, robot details

## Common Commands

### Installation
```bash
pip install mujoco gymnasium numpy opencv-python imageio egl torch stable-baselines3 scipy
```

### Quick Start (Random Policy)
```bash
python run_without_policy.py
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

### Evaluation & Video Export

**Self-play evaluation:**
```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode selfplay \
  --model combatbench/baseline/sb3/runs/fight_v1/model_final.zip \
  --phase fight \
  --episodes 5
```

**Match video (same model vs itself):**
```bash
python run_policy_video.py \
  --mode shared_env \
  --model combatbench/baseline/sb3/runs/stand_v1/best_model/best_model.zip \
  --phase stand \
  --duration 10 \
  --video output.mp4
```

**Match video (two different models):**
```bash
python run_policy_video.py \
  --mode match \
  --model path/to/model_a.zip \
  --model-b path/to/model_b.zip \
  --phase fight \
  --duration 15 \
  --video match.mp4
```

**Run a match via CLI:**
```bash
python tools/run_match.py \
  --video match.mp4 \
  --duration 30 \
  --model path/to/model.zip \
  --model-b path/to/other_model.zip
```

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

### Self-Play Environment Wrappers

`SymmetricSelfPlayEnv` wraps `CombatGymEnv` for SB3 training:
- Returns normalized `robot_a_obs` (127 dims)
- Broadcasts shared action to both robots
- Returns average shaped reward from both robots

`AttackerStandingOpponentEnv` for asymmetric training:
- Robot A (attacker): trainable residual policy on top of base
- Robot B (defender): frozen standing policy

### Observation Normalization

`ObservationNormalizer` scales observations to [-1, 1] for training stability.
Normalizes based on target height and various velocity/position scales.

### Policy Interface

Policies must implement:
```python
def act(self, obs, info=None) -> np.ndarray:  # Returns action in [-1, 1]
def reset(self):  # Optional reset
```

`SB3CombatPolicy` loads trained SB3 models and handles:
- Normalization
- Opponent feature masking (for standing phase)
- Approach base actions (lean/step forward)
- Residual action composition

## Important Notes

- Always set `MUJOCO_GL=egl` for headless rendering (GPU server)
- The project uses both SB3 and custom PyTorch PPO implementations
- Training follows a curriculum: stand → fight → attacker_approach → attacker
- Model checkpoints are saved to `baseline/sb3/runs/<run_name>/`
- Video rendering requires EGL; set environment variables before importing mujoco
