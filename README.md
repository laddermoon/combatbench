# CombatBench: Humanoid Robot Combat Benchmark

![CombatBench Hero](assets/images/hero.png)

CombatBench is an open-source humanoid combat simulation stack built on MuJoCo. It is not just a single benchmark scene or a thin Gym wrapper: it is a reusable **environment runtime framework** for building robot-vs-robot tasks, and `humanoid21` is the first complete implementation living on top of it.

If you want a sandbox where you can iterate on control, observation design, world rules, training adapters, and evaluation protocols without rewriting the whole environment every time, CombatBench is designed for that.

## Why CombatBench

- **Framework first**: the `envs/framework` layer separates physics backend, world rules, runtime scheduling, and observer outputs.
- **Robot-ready contracts**: state, action, observation, and derived signals are explicit interfaces rather than ad-hoc arrays.
- **Built for experimentation**: the same runtime can support RL training, scripted baselines, ablations, evaluation matches, and future new robots.
- **Humanoid combat as a stress test**: balance, contact, disturbance, asymmetric tactics, and self-play all show up in one environment.
- **Practical media loop**: headless EGL rendering makes it straightforward to generate replay videos and debugging views.

## The `envs/` System: What Makes It Interesting

CombatBench’s most valuable asset is the `envs/` system.

It is designed around a simple idea: **keep the physics sandbox pure, and make everything else composable**.

### `framework`: the reusable core

The `envs/framework` layer is the main reason CombatBench is more than a one-off benchmark.

- **`BaseSimulator`** keeps the backend focused on physics stepping, state read/write, and action application.
- **World plugins** handle objective world logic such as constraints, adjudication, events, and metrics.
- **Observer plugins** build task-facing outputs such as observation, reward, debug views, or analysis features.
- **`EnvRuntime`** acts as the stable public runtime entrance, orchestrating both-sided actions and the full episode lifecycle.

For you as a user, this opens up a lot of room:

- **Train different algorithms on the same runtime** without rewriting the environment core.
- **Swap observation definitions** while keeping the same combat world and simulator.
- **Add new rule plugins** for non-fall, hit-point systems, disturbances, or curriculum phases.
- **Build evaluators and visualization tools** directly on top of the standard data access interfaces.
- **Port to future robots or backends** while preserving most of the surrounding runtime logic.

This means CombatBench can be used as:

- **an RL benchmark**
- **a robotics control sandbox**
- **a self-play experimentation platform**
- **a match evaluation and replay system**

## `humanoid21`: the first complete environment implementation

`envs/humanoid21` is the current flagship implementation.

It packages a 21-DOF humanoid combat environment around the new framework contracts, with a design that is intentionally friendly to both learning systems and environment engineering.

- **Normalized position control** keeps the action interface stable and bounded.
- **Structured data contracts** separate static data, core physical state, and derived learning-facing signals.
- **Ego-centric state design** makes the observation space more reusable across tactics and spawn layouts.
- **Plugin-oriented runtime** keeps combat logic, observation logic, and simulator mechanics decoupled.
- **Dual-agent setting from day one** makes it natural to support self-play and evaluator-vs-policy workflows.

In practical terms, `humanoid21` gives you a place to imagine and build:

- **standing / recovery / anti-fall controllers**
- **contact-aware locomotion and striking policies**
- **self-play curricula from survival to aggressive combat**
- **centralized critic or decentralized actor setups**
- **future vision-only or partial-observation variants**

## Baseline: a starting point, not a ceiling

The `baseline/` directory is where CombatBench becomes immediately usable.

The current `baseline/humanoid21` track provides a concrete GRPO-based starting path for training humanoid policies, beginning from the most fundamental capability: **standing**, then moving toward **disturbance-robust standing**.

This is useful both as:

- **a sanity check** for the environment stack
- **a reference implementation** for training integration
- **a launchpad** for stronger combat-oriented policies

You do not need to adopt the baseline as-is. The point is that the framework and the baseline already meet in a way that makes your next experiment cheaper to start.

## Project Structure

- `assets/`: MuJoCo XML models, textures, meshes, and media assets.
- `envs/`: Environment runtime framework and concrete environments.
  - `framework/`: backend contracts, runtime orchestration, plugin system.
  - `humanoid21/`: current 21-DOF humanoid implementation.
- `policy/`: policy interface and reference policies.
- `baseline/`: training baselines and reproducible starting points.
- `docs/`: benchmark rules and supporting design documents.

## Installation

### Requirements

- Python 3.8+
- MuJoCo 3.x
- Gymnasium
- NumPy
- OpenCV (cv2)

### Setup

```bash
# Clone the repository
# git clone https://github.com/your-org/combatbench.git
# cd combatbench

# Install dependencies (ensure you have mujoco installed)
pip install mujoco gymnasium numpy opencv-python imageio egl
```

## Quick Start

Run a round in the current `humanoid21` environment and save a video:

```bash
# Run with no explicit policies (default standing behavior)
python envs/humanoid21/run_round.py --duration 10 --video test.mp4

# Run with a random policy
python envs/humanoid21/run_round.py --policy-a random --duration 5 --video test.mp4

# Run two different policies
python envs/humanoid21/run_round.py \
  --policy-a random \
  --policy-b standing \
  --duration 15 --video match.mp4
```

## Key Documentation

If you want the design contracts instead of a README summary, go straight to these documents:

- **Framework architecture**: [`envs/framework/DESIGN.md`](envs/framework/DESIGN.md)
- **Humanoid21 observation design**: [`envs/humanoid21/OBSERVATION_zh.md`](envs/humanoid21/OBSERVATION_zh.md)
- **Humanoid21 data contract**: [`envs/humanoid21/DATASPEC.md`](envs/humanoid21/DATASPEC.md)
- **Humanoid21 control contract**: [`envs/humanoid21/CONTROLSPEC.md`](envs/humanoid21/CONTROLSPEC.md)
- **Humanoid21 baseline guide**: [`baseline/humanoid21/README.md`](baseline/humanoid21/README.md)

Additional project documents:

- [Combat Rules](docs/RULE.md) / [中文规则](docs/RULE_zh.md)
- [Environment Details](docs/ENVIRONMENT.md) / [中文环境](docs/ENVIRONMENT_zh.md)
- [Robot Specifications](docs/ROBOT.md) / [中文机器人](docs/ROBOT_zh.md)
- [Policy Submission Guide](docs/SUBMISSION_GUIDE.md) / [中文提交指南](docs/SUBMISSION_GUIDE_zh.md)

## Policy Interface

All combat policies must inherit from `BaseCombatPolicy` and implement the `act()` method:

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
```

See [`policy/base.py`](policy/base.py) for the complete interface definition.

## Contributing

We welcome contributions! Please follow standard open-source pull request workflows.
