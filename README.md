# CombatBench: Humanoid Robot Combat Benchmark

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
- `envs/`: Gymnasium environment wrappers (`CombatGymEnv`).
- `utils/`: Helpful scripts for generating textures, compiling XMLs, etc.
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

You can run the environment without any policy to verify your setup. This will generate a random-action combat simulation and save it as an MP4 video.

```bash
python run_without_policy.py
```

## Documentation

- [Combat Rules](docs/RULE.md)
- [Environment Details](docs/ENVIRONMENT.md)
- [Robot Specifications](docs/ROBOT.md)
- [Observation Space](docs/OBSERVATION.md)
- [Scene Overview](docs/SCENE.md)
- [Policy Submission Guide](docs/SUBMISSION.md)

## Contributing

We welcome contributions! Please follow standard open-source pull request workflows.
