# CombatBench Environments

This directory contains Gymnasium-compatible environments for humanoid robot combat simulation.

## Files

| File | Description |
|------|-------------|
| `combat_gym.py` | Main `CombatGymEnv` - dual robot combat environment |
| `round_runner.py` | `RoundRunner` - runs complete rounds between two policies |
| `__init__.py` | Module exports (`CombatGymEnv`, `RoundRunner`, `RoundResult`, `run_round`) |

## CombatGymEnv

The `CombatGymEnv` is a Gymnasium-compatible environment that simulates combat between two 21-DOF humanoid robots.

### Key Features

- **Dual Robot Combat**: Two robots (Robot A in red, Robot B in blue) fight each other
- **HP-based Scoring**: Initial 100 HP, head hits = -3, torso hits = -1
- **Gymnasium Interface**: Standard `reset()` and `step()` API
- **Video Recording**: Built-in video capture and MP4 export
- **Non-fall Mode**: Optional root orientation clamping for training stability

### Observation Space (127 dims per robot)

```
robot_a_obs: Box(-inf, inf, (127,), float32)
robot_b_obs: Box(-inf, inf, (127,), float32)
```

Breakdown:
- **Proprioception** (42): Joint positions (21) + velocities (21)
- **Root State** (13): Height (1) + local orientation (6) + linear/angular velocity (6)
- **Tactile & Force** (8): Feet contact (2) + external forces (6)
- **Opponent Observation** (64): Relative position/velocity, orientation, 9 keypoints

See [`docs/OBSERVATION.md`](../docs/OBSERVATION.md) for detailed specification.

### Action Space (21 dims per robot)

```
action: {
    "robot_a": Box(-1.0, 1.0, (21,), float32),
    "robot_b": Box(-1.0, 1.0, (21,), float32),
}
```

Controlled joints:
- Abdomen (3): `abdomen_z`, `abdomen_y`, `abdomen_x`
- Right Leg (6): `hip_x_right`, `hip_z_right`, `hip_y_right`, `knee_right`, `ankle_y_right`, `ankle_x_right`
- Left Leg (6): `hip_x_left`, `hip_z_left`, `hip_y_left`, `knee_left`, `ankle_y_left`, `ankle_x_left`
- Right Arm (3): `shoulder1_right`, `shoulder2_right`, `elbow_right`
- Left Arm (3): `shoulder1_left`, `shoulder2_left`, `elbow_left`

### Initialization Parameters

```python
CombatGymEnv(
    render_mode=None,           # "human" or "rgb_array"
    arena_xml=None,             # Path to arena XML (default: assets/battle_v1.xml)
    dt=0.002,                   # Physics timestep (500Hz)
    initial_distance=2.0,       # Initial distance between robots (meters)
    control_frequency=20,       # Control frequency (Hz)
    video_sample_frequency=30,  # Video FPS (default: 30)
    match_duration=30.0,        # Match duration (seconds)
    non_fall_mode=False,        # Enable orientation clamping
    non_fall_pitch_limit_deg=15.0,  # Pitch limit for non-fall mode
    non_fall_roll_limit_deg=10.0,   # Roll limit for non-fall mode
    damage_scale=100.0,         # Damage scaling factor
)
```

### Usage Example

```python
from combatbench.envs import CombatGymEnv

# Create environment
env = CombatGymEnv(render_mode="rgb_array", match_duration=30.0)

# Reset
obs, info = env.reset()

# Step
action = {
    "robot_a": random_action_a,  # Shape (21,), range [-1, 1]
    "robot_b": random_action_b,  # Shape (21,), range [-1, 1]
}
obs, reward, terminated, truncated, info = env.step(action)

# Access results
print(f"Scores: {info['scores']}")  # HP for both robots
print(f"Winner: {info['winner']}")
print(f"End reason: {info['end_reason']}")

# Save video
env.save_video("match.mp4", fps=env.video_sample_frequency)
```

### Controller Configuration

The environment uses PD control with configurable reference positions and action scales:

```python
# Set reference positions (e.g., standing pose)
env.set_controller_reference_positions({
    "robot_a": stand_pose,
    "robot_b": stand_pose,
})

# Set action scale per joint
env.set_controller_action_scale({
    "robot_a": action_scale,
    "robot_b": action_scale,
})

# Set PD gains
env.set_controller_gains(kp=4.0, kd=0.4)

# Direct joint position control
env.set_robot_joint_positions({
    "robot_a": {"abdomen_z": 0.1, "knee_right": -0.5},
    "robot_b": {"abdomen_z": -0.1, "knee_left": -0.5},
})
```

### Info Dictionary Structure

The `info` dict returned by `step()` contains:

```python
{
    # Scores
    "scores": {"robot_a": 100.0, "robot_b": 95.0},

    # Positions
    "positions": {"robot_a": np.array([...]), "robot_b": np.array([...])},
    "torso_positions": {...},

    # Robot States
    "robot_states": {
        "robot_a": {...},  # Full state summary
        "robot_b": {...},
    },

    # Relative Metrics
    "relative_metrics": {
        "robot_a": {
            "distance": 2.0,
            "horizontal_distance": 2.0,
            "relative_position": np.array([...]),
            "direction_to_opponent": np.array([...]),
            "facing_opponent": 1.0,
        },
        "robot_b": {...},
    },

    # Hit Records
    "hit_records": {
        "robot_a": [
            {
                "hit_part": "torso",
                "damage_part": "hand",
                "damage": -1.0,
                "velocity": 2.5,
                "force": 50.0,
                "impulse": 0.5,
                "contact_count": 1,
            },
            ...
        ],
        "robot_b": [...],
    },

    # Match State
    "winner": "robot_a",  # or "robot_b", or None
    "end_reason": "Time limit reached (30.0s), robot_a wins by health",
    "current_step": 600,
    "physics_step_count": 15000,

    # Controller State
    "controller_state": {
        "robot_a": {
            "reference_positions": np.array([...]),
            "target_positions": np.array([...]),
            "action_scale": np.array([...]),
        },
        "robot_b": {...},
    },

    # Non-fall Mode Settings
    "non_fall_mode": {
        "enabled": False,
        "pitch_limit_deg": 15.0,
        "roll_limit_deg": 10.0,
    },

    # Observation Slices
    "observation_slices": {...},  # Index ranges for observation components
}
```

### Video Recording

```python
# Enable video during initialization
env = CombatGymEnv(render_mode="rgb_array")

# After running episode, save video
env.save_video("match.mp4", fps=30)

# Or manually access frames
frames = env.get_video_buffer()
env.clear_video_buffer()
```

### Non-Fall Mode

When `non_fall_mode=True`, the environment clamps root orientation to prevent robots from falling:

```python
env = CombatGymEnv(
    non_fall_mode=True,
    non_fall_pitch_limit_deg=15.0,  # Max pitch deviation
    non_fall_roll_limit_deg=10.0,   # Max roll deviation
)
```

This is useful during training to maintain stability and prevent "physics explosions."

### Combat Rules

See [`docs/RULE.md`](../docs/RULE.md) for complete rules:

- Initial HP: 100 per robot
- Valid attacks: Hands, forearms, elbows, upper arms, feet, shins, knees, thighs
- Valid targets: Head (-3 HP), Torso (-1 HP)
- Win conditions: KO (HP=0), time limit (higher HP wins), draw (equal HP)

## RoundRunner

The `RoundRunner` class provides a high-level interface for running complete rounds between two policies.

### Usage

```python
from combatbench.envs import RoundRunner
from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy

runner = RoundRunner(
    policy_a=RandomCombatPolicy(scale=0.1),
    policy_b=StandingCombatPolicy(),
    match_duration=30.0,
    render_mode="rgb_array",
)
result = runner.run(save_video_path="match.mp4")

print(f"Winner: {result.winner}")
print(f"Steps: {result.steps}")
print(f"Final HP: {result.scores}")
```

Or use the convenience function:

```python
from combatbench.envs import run_round

result = run_round(
    policy_a="combatbench.policy.RandomCombatPolicy",
    policy_b="combatbench.policy.StandingCombatPolicy",
    save_video_path="match.mp4",
)
```

### RoundResult

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

## Design Decisions

### 21-DOF vs 17-DOF

CombatBench uses a 21-DOF humanoid model (with ankle joints) instead of the 17-DOF model (without ankles). See [`docs/ROBOT.md`](../docs/ROBOT.md) for the detailed rationale.

### Physics Configuration

- **Timestep**: 0.002s (500Hz physics)
- **Control Frequency**: 20Hz (50 physics steps per action)
- **Video FPS**: 30 (configurable via `video_sample_frequency`)

These parameters were chosen to balance:
- Simulation accuracy (higher Hz = more accurate)
- Training efficiency (lower Hz = faster episodes)
- Video quality (30 FPS for smooth playback)

### Action Normalization

Actions are normalized to [-1, 1] and then:
1. Multiplied by per-joint `action_scale`
2. Added to `reference_positions`
3. Clipped to joint limits
4. Converted to torque via PD control

This allows policies to output normalized actions while enabling per-joint tuning.

### Camera System

The `get_broadcast_view()` method implements an intelligent camera that:
- Tracks the midpoint between robots
- Positions at the side of the arena
- Smooths movements (EMA on position/angle)
- Respects room boundaries

This provides a consistent broadcast view for training and evaluation.
