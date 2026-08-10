# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CombatBench is a MuJoCo-based humanoid robot combat simulation environment. It provides a Gymnasium-compatible RL environment where two 21-DOF humanoid robots fight each other. The project includes a physics engine, collision detection, scoring system, and a custom multi-critic PPO/SAC training framework with curriculum learning support.

## Project Structure

- `assets/` - MuJoCo XML models, textures, meshes (arena: `battle_v1.xml`)
- `envs/` - Environment implementations
  - `framework/` - Core framework interfaces (BasePlugin, SimContext, etc.)
  - `humanoid21/` - 21-DOF humanoid robot environment
    - `simulator.py` - Main simulator (`Humanoid21Simulator`)
    - `plugins.py` - Combat plugins (scoring, non-fall constraint, frozen robot)
    - `observer_plugins.py` - Gymnasium observation plugin
    - `disturbance_plugins.py` - External disturbance plugins
    - `DATASPEC.md` - Data interface specification
    - `OBSERVATION_zh.md` - Observation space documentation (96-dim)
  - `t800/` - T800 robot environment (larger DOF robot)
  - `framework/round_runner.py` - `RoundRunner` class and CLI for running complete rounds
  - `framework/match_runner.py` - `MatchRunner` for multi-round matches
- `policy/` - Policy reference implementations
  - `random/` - RandomCombatPolicy directory (with policy.py)
  - `humanoid21/standing/` - StandingCombatPolicy directory (with policy.py, model.pt)
  - `blueprints/` - Policy blueprint YAML files
  - `examples/` - User-submitted competition policies
- `envs/framework/policy.py` - `Policy` ABC, `PolicyBlueprint`, `ParameterizedPolicyBlueprint`
- `docs/` - Rules, environment specs, robot details
- `baseline/` - Training implementations
  - `framework/` - **Unified training framework** (PPO/SAC, code snapshot, experiment base)
    - `train.py` - Main training CLI (supports `--background`, `--smoke`, `--resume-from`, `--no-snapshot`, `--run-dir`)
    - `experiment.py` - `Experiment` ABC + `CommonParams`, `PPOParams`, `SACParams` dataclasses
    - `ppo_loop.py` / `ppo_trainer.py` - PPO training loop + update logic (multi-critic, confidence weighting, plateau detection)
    - `sac_loop.py` / `sac_trainer.py` - SAC training loop + update logic (multi-critic Q, auto-alpha)
    - `code_snapshot.py` - Git-based code snapshot for experiment reproducibility
    - `analyze_training.py` - Training log analysis & visualization
  - `humanoid21/` - Humanoid21-specific training code
    - `standing.py` - Standing baseline (PPO with risk-aware exploration)
    - `rewards/` - Reward observer implementations (standup, balance, etc.)
    - `plugins/` - Custom termination/disturbance plugins
    - `blueprints/` - Environment blueprint YAML files
    - `curriculum/` - Curriculum learning experiments
      - `experiments/` - Auto-discovered experiment registry (`exp_*.py` files)
        - `base.py` - `CombatExperimentBase` (default impls for PPO + SAC)
        - `__init__.py` - Registry: `get_experiment()`, `list_experiments()`
      - `framework/` → renamed to `legacy_framework/` (to be deleted)
  - `common/` - Shared utilities (policies, rollout, replay buffer)
  - `runs/` - Training run outputs (gitignored, can be very large)

## Framework Architecture

The `envs/framework/` directory contains the core framework that provides:

- **Backend abstraction** - Physics engine agnostic interfaces (MuJoCo/PyBullet/IsaacGym)
- **Plugin system** - World rules and observation/reward computation
- **Runtime management** - Episode lifecycle and timing control
- **Data contracts** - Standardized accessor/mutator pattern

### Directory Structure

```
envs/framework/
├── backend.py          # IDataAccessor, IDataMutator, BaseSimulator interfaces
├── context.py          # SimContext, ReadOnlySimContext, TerminationReason
├── plugin.py           # BasePlugin for world rules
├── runtime_plugin.py   # BaseObserverPlugin for observations/rewards
├── env_runtime.py      # EnvRuntime (main public API)
├── common_plugins.py   # TimeoutPlugin, VideoRecorderPlugin
├── round_runner.py     # RoundRunner for single-round evaluation
├── match_runner.py     # MatchRunner for multi-round matches
├── DESIGN.md           # Architecture design specification (Chinese)
└── README.md           # Framework usage documentation
```

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Algorithm/Adapter Layer (PPO/SAC/IL/Gym wrappers)      │
├─────────────────────────────────────────────────────────┤
│  Policy Runtime Layer (EnvRuntime + ObserverPlugins)    │
├─────────────────────────────────────────────────────────┤
│  Physical Sandbox Layer (_RuntimeCore + WorldPlugins)   │
├─────────────────────────────────────────────────────────┤
│  Backend Layer (BaseSimulator - MuJoCo/PyBullet/...)    │
└─────────────────────────────────────────────────────────┘
```

### Core Interfaces

#### IDataAccessor (`backend.py`)
Read-only data access interface:
- `get_static_data()` - Episode-invariant config (robot info, indices)
- `get_core_state()` - Raw physics state (qpos, qvel, root pose)
- `get_derived_state()` - Computed state (contacts, kinematics)
- `get_sensor_data()` - Sensor readings (IMU, touch, force)
- `get_broadcastview_image()` - Rendered image

#### IDataMutator (`backend.py`)
Write access interface (selectively granted):
- `set_core_state(state)` - Override physics state
- `set_action(action)` - Set control action
- `apply_external_force(body, force, torque, robot_id)` - Apply disturbances

#### BaseSimulator (`backend.py`)
Abstract physics backend inheriting both Accessor and Mutator:
- `reset(seed, options)` - Reset physics engine
- `physical_step()` - Single fine-grained physics step
- `get_physical_frequency()` - Physics frequency in Hz

### SimContext (`context.py`)

Blackboard pattern for cross-plugin communication:

```python
class SimContext:
    accessor: IDataAccessor              # Always available (read-only)
    mutator: Optional[IDataMutator]      # Conditionally granted
    metrics: Dict[str, Any]              # Shared metrics (HP, damage, counts)
    events: List[Any]                    # Instantaneous events (hits, fouls)
    termination_proposals: List[str]     # Episode termination requests
    episode_step: int                    # Action-level step counter
    physics_step: int                    # Physics-level step counter

    def request_termination(self, reason: str) -> None:
        """Propose episode end with reason"""
```

**ReadOnlySimContext** - Frozen, read-only view for observer plugins (no mutator access).

**TerminationReason** - Standard constants: `TIMEOUT`, `KO`, `FOUL`, `OUT_OF_BOUNDS`, `CUSTOM`

### Plugin System

#### BasePlugin (`plugin.py`)

World rule plugin with lifecycle hooks:

| Hook | Timing | Mutator | Use Cases |
|------|--------|---------|-----------|
| `on_pre_episode` | After reset | ✓ | Resetter, initialization |
| `on_pre_action_step` | Before action | ✓ | Action mapping, clamping |
| `on_pre_phy_step` | Before physics step | ✓ | External disturbances |
| `on_post_phy_step` | After physics step | ✓ | State constraints |
| `on_post_action_step` | After action step | ✗ | Metrics, rewards, termination |
| `on_post_episode` | After episode ends | ✗ | Logging, aggregation |

**Properties:**
- `name: str` - Plugin identifier
- `priority: int` - Execution order (higher first)
- `require_mutator: bool` - Request write permission

#### BaseObserverPlugin (`runtime_plugin.py`)

Read-only observation/reward computation:
- `on_pre_episode(ctx)` - Called after reset (same name as BasePlugin)
- `on_post_action_step(ctx)` - Called after each action step (same name as BasePlugin)
- `get_output()` - Return cached output

### EnvRuntime (`env_runtime.py`)

Main public API for policy execution:

```python
from combatbench.envs.framework import EnvRuntime

runtime = EnvRuntime(simulator, world_plugins=[], observer_plugins={})

# Episode management
runtime.reset(seed=42, options={})
runtime.step(action_a, action_b)

# Data access
obs_a, obs_b = runtime.get_observation()  # Get per-agent observations
reward = runtime.get_observer_output("robot_a_reward")  # Get observer plugin output
terminated, truncated = runtime.get_termination_flags()

# Plugin management
runtime.attach_plugin(plugin)
runtime.attach_observer_plugin("name", observer_plugin)
```

### Design Patterns

1. **Accessor/Mutator Pattern** - Capability-based security (read always available, write selectively granted)
2. **Blackboard Pattern** - SimContext for cross-plugin communication
3. **Plugin Architecture** - World plugins modify simulation; observers only read
4. **Dispatcher Pattern** - Centralized observer management with batch processing
5. **Factory Pattern** - `make_env()` functions create configured instances

### Common Plugins

**TimeoutPlugin** - Terminates episode after max_steps
**VideoRecorderPlugin** - Records broadcast view to MP4 at specified FPS

## Training Framework

### Architecture

The training framework in `baseline/framework/` is a custom multi-critic PPO/SAC implementation with curriculum learning support. It does **not** use Stable-Baselines3.

Key design decisions:
- **Multi-critic**: One value/Q critic per reward component (e.g. `r_fall`, `r_cross`, `r_joint`, ...)
- **Curriculum scheduling**: Each experiment defines `initial_weights()` and `next_weights()` to control reward component weighting across training
- **Experiment registry**: Auto-discovers `exp_*.py` files in `baseline/humanoid21/curriculum/experiments/`
- **Parallel rollout**: Multi-process episode collection via `ParallelRollouter`

### Training CLI

> **CRITICAL — PYTHONPATH requirement**: All training commands must be run from the `things/combatbench/` directory with `PYTHONPATH` set to that directory. Otherwise `import baseline` will fail silently (especially in `--background` mode where stderr is redirected to the log file and errors are invisible).
>
> ```bash
> # Correct way to launch background training:
> cd /data1/mono/things/combatbench
> PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=<N> \
>   python3 -B baseline/framework/train.py --experiment <name> --algo ppo --background
> ```
>
> **Do NOT wrap `--background` with `nohup ... &`** — `--background` already forks + setsid. Using nohup on top can cause the child process to be killed when the parent shell exits, and hides startup errors.

```bash
# List available experiments
PYTHONPATH=. python3 baseline/framework/train.py --list-experiments

# Smoke test (2 updates, 8 episodes, fast sanity check)
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --algo ppo --smoke

# Full training (foreground, logs to console + run_dir/train.log)
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --algo ppo

# Background training (forks + setsid, logs to run_dir/train.log only)
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --algo ppo --background

# Resume from checkpoint
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --algo ppo \
  --resume-from baseline/runs/train_basic_balance_ppo_20260101_120000/checkpoints/checkpoint_u01000.pt

# Custom run name
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --algo ppo --run-name my_exp_v1

# Disable git code snapshot
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --algo ppo --no-snapshot
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--experiment` | (required) | Experiment name from registry |
| `--algo` | `ppo` | Algorithm: `ppo` or `sac` |
| `--smoke` | off | Short run (2 updates, 8 episodes) for sanity check |
| `--resume-from` | None | Checkpoint path to resume from |
| `--run-name` | auto-generated | `train_<exp>_<algo>_<timestamp>` |
| `--run-dir` | auto | Explicit run output directory |
| `--background` | off | Run in background (fork + setsid) |
| `--no-snapshot` | off | Skip git code snapshot |
| `--no-confidence` | off | Disable EV-based confidence weighting (PPO) |
| `--list-experiments` | off | List experiments and exit |

### Run Directory Structure

```
baseline/runs/<run_name>/
├── config.json              # Experiment config snapshot
├── train.log                # Full training log (stdout + stderr)
├── pid                      # PID file (background mode only)
├── code_snapshot.json       # Git branch + commit info
├── REPRODUCE.md             # Reproduction commands
├── checkpoints/             # Periodic checkpoints (checkpoint_uNNNNN.pt)
├── policy/                  # Best-of-run exported policy
├── policy_exports/          # Per-update policy blueprints
└── videos/                  # Evaluation videos (uNNNNN.mp4)
```

### Code Snapshot & Reproducibility

Every training run (unless `--no-snapshot`) creates a git branch `exp/<run_name>_<timestamp>` capturing the exact code state, without disturbing the working tree. The run directory contains:
- `code_snapshot.json` — branch name, commit hash, base commit
- `REPRODUCE.md` — copy-paste commands to reproduce via `git worktree`

### Adding a New Experiment

1. Create `baseline/humanoid21/curriculum/experiments/exp_<name>.py`
2. Define a class inheriting `CombatExperimentBase`
3. Set `name`, `reward_keys`, `gammas`, `BLUEPRINT`
4. Implement: `extract_rewards()`, `compute_episode_metrics()`, `initial_weights()`, `next_weights()`, `build_rollout_jobs()`, `build_eval_jobs()`, `compare_eval()`, `scheduler_info()`
5. Export singleton: `EXPERIMENT = MyExperimentConfig()`
6. The registry auto-discovers it on next `--list-experiments`

### Background Mode

`--background` forks the process, detaches from terminal (`setsid`), and redirects all output to `run_dir/train.log`. Prints run info to console before exiting:
```
[run] started in background
[run] dir: /data1/mono/things/combatbench/baseline/runs/train_xxx_ppo_...
[run] log: .../train.log
[run] pid: 12345
[run] monitor: tail -f .../train.log
[run] stop: kill 12345
```

### Logging

All output (Python `print()`, C-level stdout/stderr) is tee'd to `run_dir/train.log`. In foreground mode, output also goes to the console. The log includes human-readable training progress and machine-readable `__RAW_STATS__` JSON lines for monitoring scripts.

## Common Commands

### Installation
```bash
pip install -e .
```
Or install dependencies manually:
```bash
pip install -r requirements.txt
```

### Running Rounds (Evaluation & Video)

**Using the round runner CLI:**
```bash
# Run a round with blueprint files
python -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint policy/blueprints/random.yaml \
  --policy-b-blueprint policy/blueprints/humanoid21/standing.yaml \
  --video match.mp4
```

**Using RoundRunner with BaseFrameRecorder (video + per-step data):**

The `--recorder` flag injects a `PostActionRecorder` via spec string `module.path:ClassName?key=value`.
`BaseFrameRecorder` saves per-step images (PNG) and JSON (observer outputs, core/derived state, actions)
for replay and debugging.

```bash
# Run a round with video + recorder, same policy for both robots
PYTHONPATH=. python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/standup_4stage_env.yaml \
  --policy-a-blueprint <policy_dir>/policy_blueprint.yaml \
  --policy-b-blueprint <policy_dir>/policy_blueprint.yaml \
  --video /data1/dev/replay/round.mp4 \
  --recorder "envs.framework.recorder:BaseFrameRecorder?output_dir=/data1/dev/replay/rec" \
  --seed 1000
```

Output structure:
```
/data1/dev/replay/
├── round.mp4                      # Video file
└── rec/
    ├── index.json                 # Episode index
    └── episode_00000/
        ├── manifest.json          # Step list + base_seed
        ├── static.json            # Episode-invariant data
        ├── step_00000.png         # Broadcast view image
        ├── step_00000.json        # observer_outputs + core_state + derived_state
        └── ...
```

**Using recorder_viewer to inspect recordings interactively:**

```bash
# Start the web viewer (serves recording dir + opens browser)
PYTHONPATH=. python3 -m envs.framework.recorder_viewer /data1/dev/replay/rec

# Custom port, no auto-browser
PYTHONPATH=. python3 -m envs.framework.recorder_viewer /data1/dev/replay/rec --port 9000 --no-browser
```

The viewer serves at `http://localhost:8765/viewer.html` (default port 8765).
It fetches `index.json` for the episode list, then per-step PNG images and JSON
data as you navigate. Observer outputs (stage, potential, reward) are visible
in the per-step JSON panel.

**Using RoundRunner in Python code:**
```python
from combatbench.envs.framework import EnvBlueprint, PolicyBlueprint, RoundRunner, VideoRecorderPlugin

blueprint = EnvBlueprint.load("envs/humanoid21/blueprint.yaml")
policy_a = PolicyBlueprint.load("policy/blueprints/random.yaml").build()
policy_b = PolicyBlueprint.load("policy/blueprints/humanoid21/standing.yaml").build()

video = VideoRecorderPlugin(fps=30, output_path="output.mp4")
with RoundRunner(
    blueprint=blueprint,
    policy_a=policy_a,
    policy_b=policy_b,
    video_plugin=video,
) as runner:
    result = runner.run(seed=42)
print(f"Steps: {result['steps']}, Termination: {result['termination_reasons']}")
```

## Policy System

All policies use a directory-based structure with `policy.py` containing a `Policy` implementation.

### Directory Structure

Each policy is a directory with:
- **`policy.py`** (required) - Contains a class inheriting `Policy` from `envs.framework.policy`
- **`policy_blueprint.yaml`** (optional) - Blueprint for loading the policy
- **`model.pt`** (optional) - Trained model weights

Example:
```
my_policy/
├── policy.py            # Must contain Policy implementation
├── policy_blueprint.yaml # Blueprint with cls and config
└── model.pt             # Optional trained weights
```

### Loading via PolicyBlueprint

Policies are loaded via `PolicyBlueprint` (YAML/JSON):

```python
from combatbench.envs.framework import PolicyBlueprint

# Load from YAML file
policy = PolicyBlueprint.load("my_policy/policy_blueprint.yaml").build()

# Load with parameter overrides
policy = PolicyBlueprint.load("my_policy/policy_blueprint.yaml").build(scale=0.2)
```

### Policy Interface

All policies must inherit from `Policy` (defined in `envs/framework/policy.py`) and implement:

```python
from envs.framework.policy import Policy
import numpy as np

class MyCombatPolicy(Policy):
    def __init__(self, **kwargs):
        super().__init__()
        # Your initialization

    def act(self, obs: np.ndarray, want_extra: bool = False) -> tuple:
        """
        Return (action, extra) tuple.

        Args:
            obs: observation from the observer plugin
            want_extra: if True, return auxiliary payload (log_prob, value, etc.)

        Returns:
            action: np.ndarray with shape (21,), values in [-1, 1]
            extra: dict or None
        """
        action = np.zeros(21)
        return action, None

    def reset(self, seed=None) -> None:
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

- [`docs/RULE.md`](docs/RULE.md) - Combat rules and HP system
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) - Arena and simulation environment
- [`envs/humanoid21/DATASPEC.md`](envs/humanoid21/DATASPEC.md) - Data interface specification
- [`envs/humanoid21/OBSERVATION_zh.md`](envs/humanoid21/OBSERVATION_zh.md) - 96-dim observation details
- [`policy/README.md`](policy/README.md) - Policy implementation guide

## Important Notes

- **Observation dimension**: 96 (not 127 - old version)
- **Policy structure**: Directory-based with `policy.py` (not single-file policies)
- **Data format**: Structured by `robot_id` (robot_a/robot_b), not flat arrays
- **Plugin system**: Uses `SimContext` with accessor/mutator pattern
- **Training framework**: Custom PPO/SAC in `baseline/framework/`, not Stable-Baselines3
- **Experiment registry**: Auto-discovers `exp_*.py` in `baseline/humanoid21/curriculum/experiments/`
- **Run directories**: `baseline/runs/` is gitignored, can be very large (checkpoints + videos)
- **Legacy code**: `baseline/humanoid21/curriculum/legacy_framework/` is deprecated, all imports now point to `baseline/framework/`
- **Code snapshot**: Training runs create git branches `exp/*` for reproducibility; clean up with `git branch -D exp/...` when no longer needed
