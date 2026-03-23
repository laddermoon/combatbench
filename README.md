# CombatBench: Humanoid Robot Combat Benchmark

![CombatBench Hero](assets/images/hero.png)

CombatBench is the open-source simulation environment for humanoid robot combat. It provides a standardized MuJoCo-based environment where two 21-DOF humanoid robots can fight against each other.

## Features

- **21-DOF Humanoid Robots**: High-fidelity robots with ankle joints for realistic combat movements.
- **Official Combat Arena**: Standardized 6.1m x 6.1m closed room with proper lighting and camera setups.
- **Gymnasium Interface**: Standard RL environment interface (`reset`, `step`, etc.).
- **Headless Rendering**: EGL-based fast rendering for generating combat replay videos.
- **Extensibility**: Designed to support future robots (like Unitree G1) and pure vision-based RL observation spaces.


## Project Structure

- `assets/`: Simulation XML models, textures, and meshes.
- `core/`: Core engine components (Physics, Collision Detection, Scoring, Robot Kinematics).
- `envs/`: Gymnasium environment wrappers (`CombatGymEnv`, `RoundRunner`).
- `policy/`: Policy interface and reference implementations.
  - `BaseCombatPolicy`: Abstract base class for all combat policies
  - `RandomCombatPolicy`: Random action policy for testing
  - `StandingCombatPolicy`: Standing still policy (no movement)
- `tools/`: Utilities for running rounds (`run_round.py`).
- `baseline/`: Baseline training implementations (Stable-Baselines3, self-play).
- `docs/`: Detailed documentation on rules, robot specs, and observation spaces.

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

Run a combat round between two policies and save as video. The default policy (no arguments) is StandingCombatPolicy which keeps the robot in place.

```bash
# Run with no policies (both standing)
python tools/run_round.py --duration 10 --video test.mp4

# Run with random policy
python tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy --duration 5 --video test.mp4

# Run two different policies
python tools/run_round.py \
  --policy-a combatbench.policy.RandomCombatPolicy \
  --policy-b combatbench.policy.StandingCombatPolicy \
  --duration 15 --video match.mp4
```

## Documentation

- [Combat Rules](docs/RULE.md) / [中文规则](docs/RULE_zh.md)
- [Environment Details](docs/ENVIRONMENT.md) / [中文环境](docs/ENVIRONMENT_zh.md)
- [Robot Specifications](docs/ROBOT.md) / [中文机器人](docs/ROBOT_zh.md)
- [Observation Space](docs/OBSERVATION.md) / [中文观测](docs/OBSERVATION_zh.md)
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
