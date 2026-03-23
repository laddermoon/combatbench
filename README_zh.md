# CombatBench: 人形机器人对战基准平台

![CombatBench Hero](assets/images/hero.png)

CombatBench 是一个用于人形机器人对战的开源仿真环境。它提供了一个基于 MuJoCo 的标准化环境，其中两个 21自由度 (21-DOF) 的人形机器人可以互相进行格斗。

## 特性

- **21自由度人形机器人**：具有脚踝关节的高保真机器人，能够实现更真实的格斗动作（移动、躲闪）。
- **标准对战竞技场**：标准的 6.1m x 6.1m 封闭房间，配备合理的灯光和多角度摄像机设置。
- **Gymnasium 接口**：标准强化学习环境接口（支持 `reset`, `step` 等）。
- **无头渲染 (Headless)**：基于 EGL 的快速渲染，用于生成格斗回放视频。
- **高可扩展性**：采用面向对象设计，支持未来接入新的机器人（如 宇树G1）以及纯视觉观测(Vision-based)的强化学习。

## 项目结构

- `assets/`: 仿真所需的 XML 模型、贴图纹理和网格文件。
- `core/`: 核心引擎组件（物理引擎、碰撞检测、得分计算、机器人运动学）。
- `envs/`: Gymnasium 环境封装 (`CombatGymEnv`, `RoundRunner`)。
- `policy/`: 策略接口和参考实现。
  - `BaseCombatPolicy`: 所有对战策略的抽象基类
  - `RandomCombatPolicy`: 随机动作策略，用于测试
  - `StandingCombatPolicy`: 静止策略（无动作）
- `tools/`: 运行回合的工具 (`run_round.py`)。
- `baseline/`: 基线训练实现 (Stable-Baselines3, 自我对弈)。
- `docs/`: 关于规则、机器人规格以及观测空间的详细文档。

## 安装指南

### 依赖要求

- Python 3.8+
- MuJoCo 3.x
- Gymnasium
- NumPy
- OpenCV (cv2)

### 环境配置

```bash
# 克隆仓库
# git clone https://github.com/your-org/combatbench.git
# cd combatbench

# 安装依赖项 (请确保你已经安装了 mujoco)
pip install mujoco gymnasium numpy opencv-python imageio egl
```

## 快速开始

运行两个策略之间的对战回合并保存为视频。默认策略（无参数）是 StandingCombatPolicy，它会保持机器人原地不动。

```bash
# 无策略运行（双方都静止）
python tools/run_round.py --duration 10 --video test.mp4

# 使用随机策略运行
python tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy --duration 5 --video test.mp4

# 运行两个不同的策略
python tools/run_round.py \
  --policy-a combatbench.policy.RandomCombatPolicy \
  --policy-b combatbench.policy.StandingCombatPolicy \
  --duration 15 --video match.mp4
```

## 文档

- [对战规则](docs/RULE_zh.md)
- [环境详情](docs/ENVIRONMENT_zh.md)
- [机器人规格](docs/ROBOT_zh.md)
- [观测空间](docs/OBSERVATION_zh.md)
- [策略提交指南](docs/SUBMISSION_GUIDE_zh.md)

## 策略接口

所有对战策略必须继承自 `BaseCombatPolicy` 并实现 `act()` 方法：

```python
from combatbench.policy import BaseCombatPolicy
import numpy as np

class MyPolicy(BaseCombatPolicy):
    def __init__(self, observation_space=None, action_space=None, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        # 你的初始化代码

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """返回形状为 (21,) 的动作数组，值范围在 [-1, 1]"""
        # 你的动作计算
        return action
```

完整的接口定义请参见 [`policy/base.py`](policy/base.py)。

## 参与贡献

我们欢迎各位开发者的贡献！请遵循标准的开源 Pull Request 流程。
