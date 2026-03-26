# CombatBench Environments

This directory contains Gymnasium-compatible environments for humanoid robot combat simulation.

## Files

| File | Description |
|------|-------------|
| `combat_gym.py` | Main `CombatGymEnv` - dual robot combat environment |
| `round_runner.py` | `RoundRunner` - runs complete rounds between two policies |
| `resetters.py` | Reset plugins for initial pose sampling |
| `constraints.py` | Constraint plugins for physical constraints (e.g., non-fall) |
| `disturbances.py` | Disturbance plugins for external perturbations |
| `control_modes.py` | Control mode plugins for action resolution |
| `metrics.py` | Metric collector plugins for extensible diagnostics |
| `__init__.py` | Module exports |

## CombatGymEnv

The `CombatGymEnv` is a Gymnasium-compatible environment that simulates combat between two 21-DOF humanoid robots.

### Key Features

- **Dual Robot Combat**: Two robots (Robot A in red, Robot B in blue) fight each other
- **HP-based Scoring**: Initial 100 HP, head hits = -3, torso hits = -1
- **Gymnasium Interface**: Standard `reset()` and `step()` API
- **Video Recording**: Built-in video capture and MP4 export
- **Plugin-based Runtime**: Modular resetters, constraints, disturbances, control modes, and metrics

### Design Philosophy

`CombatGymEnv` follows a **plugin-based architecture** to maximize extensibility without environment class bloat:

- **Resetters**: Control initial pose sampling (symmetric stand, randomized, custom)
- **Constraints**: Apply physical constraints (non-fall orientation clamp, safety limits)
- **Disturbances**: Inject external perturbations (random pushes, scheduled forces)
- **Control Modes**: Resolve robot actions (policy-driven, zero-action, fixed-action, callback)
- **Metric Collectors**: Gather extensible diagnostic data (core metrics, constraint stats, disturbance events)

The environment core remains minimal and generic, delegating specialized behaviors to composable plugins.

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
    damage_scale=100.0,         # Damage scaling factor
    
    # Plugin-based runtime modules
    resetter=None,              # BaseResetter instance (default: SymmetricStandResetter)
    constraints=None,           # Sequence of BaseConstraint instances
    disturbances=None,          # Sequence of BaseDisturbance instances
    control_modes=None,         # Dict[robot_id, BaseControlMode] (default: PolicyControlMode for both)
    metric_collectors=None,     # Sequence of BaseMetricCollector instances
    add_default_metric_collectors=True,  # Auto-add CoreMetricCollector, ConstraintMetricCollector, DisturbanceMetricCollector
)
```

### Usage Example

#### Basic Usage

```python
from combatbench.envs import CombatGymEnv

# Create environment with default plugins
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

#### Plugin-based Customization

```python
from combatbench.envs import (
    CombatGymEnv,
    SymmetricStandResetter,
    NonFallOrientationClamp,
    RandomPushDisturbance,
    ZeroActionControlMode,
)

# Create environment with custom plugins
env = CombatGymEnv(
    render_mode="rgb_array",
    match_duration=30.0,
    
    # Custom initial pose with randomization
    resetter=SymmetricStandResetter(
        initial_distance=2.5,
        yaw_jitter_deg=5.0,
        lateral_jitter=0.1,
    ),
    
    # Enable non-fall constraint
    constraints=[
        NonFallOrientationClamp(
            pitch_limit_deg=10.0,
            roll_limit_deg=10.0,
        ),
    ],
    
    # Add random push disturbances
    disturbances=[
        RandomPushDisturbance(
            probability_per_substep=0.001,
            force_range=(50.0, 200.0),
        ),
    ],
    
    # Make robot_b a static target
    control_modes={
        "robot_b": ZeroActionControlMode(),
    },
)

obs, info = env.reset()
print(f"Reset type: {info['reset']['type']}")
print(f"Control modes: {info['control_modes']}")
print(f"Metrics: {list(info['metrics'].keys())}")
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

    # Plugin Runtime Info
    "control_modes": {
        "robot_a": "policy",
        "robot_b": "zero_action",
    },
    "reset": {
        "type": "symmetric_stand",
        "initial_distance": 2.0,
        ...
    },
    "constraints": {
        "non_fall_orientation_clamp": {
            "name": "non_fall_orientation_clamp",
            "enabled": True,
            "pitch_limit_deg": 10.0,
            "roll_limit_deg": 10.0,
            "clamped": {"robot_a": False, "robot_b": False},
            "current_step": {"robot_a": 0, "robot_b": 0},
            "episode": {"robot_a": 5, "robot_b": 3},
        },
    },
    "disturbances": [
        {
            "type": "random_push",
            "robot_id": "robot_a",
            "body_name": "torso",
            "force": [120.0, 0.0, 0.0],
            ...
        },
    ],
    "metrics": {
        "core": {...},
        "constraints": {...},
        "disturbances": {...},
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

### Plugin Modules

#### Resetters

Control initial pose sampling:

```python
from combatbench.envs import SymmetricStandResetter, RandomizedSymmetricStandResetter

# Symmetric stand with optional jitter
resetter = SymmetricStandResetter(
    initial_distance=2.0,
    root_height=1.282,
    yaw_jitter_deg=5.0,
    lateral_jitter=0.1,
)

# Fully randomized initial poses
resetter = RandomizedSymmetricStandResetter(
    distance_range=(1.5, 3.0),
    height_range=(1.2, 1.4),
    yaw_range=(-10.0, 10.0),
)
```

#### Constraints

Apply physical constraints during simulation:

```python
from combatbench.envs import NonFallOrientationClamp

# Prevent robots from falling over
constraint = NonFallOrientationClamp(
    pitch_limit_deg=10.0,
    roll_limit_deg=10.0,
)
```

#### Disturbances

Inject external perturbations:

```python
from combatbench.envs import RandomPushDisturbance, ScheduledPushDisturbance

# Random pushes during episode
disturbance = RandomPushDisturbance(
    probability_per_substep=0.001,
    force_range=(50.0, 200.0),
    target_bodies=["torso", "pelvis"],
)

# Scheduled push at specific step
disturbance = ScheduledPushDisturbance(
    trigger_step=100,
    robot_id="robot_a",
    body_name="torso",
    force=[200.0, 0.0, 0.0],
)
```

#### Control Modes

Resolve robot actions:

```python
from combatbench.envs import (
    PolicyControlMode,      # Normal policy-driven control
    ZeroActionControlMode,  # Robot stays in reference pose
    FixedActionControlMode, # Robot executes fixed action
    CallbackControlMode,    # Custom action callback
)

# Make robot_b a static target
env = CombatGymEnv(
    control_modes={
        "robot_b": ZeroActionControlMode(),
    },
)

# Custom action callback
def my_action_callback(env, robot_id, action):
    return np.zeros(21, dtype=np.float32)  # Custom logic

env = CombatGymEnv(
    control_modes={
        "robot_b": CallbackControlMode(my_action_callback),
    },
)
```

#### Metric Collectors

Gather extensible diagnostic data:

```python
from combatbench.envs import BaseMetricCollector

class CustomMetricCollector(BaseMetricCollector):
    name = "custom"
    
    def collect(self, env, observation, info, *, terminated, truncated):
        return {
            "my_metric": 42.0,
        }

env = CombatGymEnv(
    metric_collectors=[CustomMetricCollector()],
    add_default_metric_collectors=True,  # Keep core/constraint/disturbance metrics
)
```

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
