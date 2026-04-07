# CombatBench: 人形机器人对战基准平台

![CombatBench Hero](assets/images/hero.png)

CombatBench 是一个基于 MuJoCo 的开源人形机器人对战仿真平台。它不仅仅是一个单一的基准场景或一个简单的 Gym 封装：它是一个可复用的**环境运行时框架**，用于构建机器人对机器人的任务，而 `humanoid21` 是基于此框架的第一个完整实现。

如果你想要一个可以在控制、观测设计、世界规则、训练适配器和评估协议上快速迭代，而不需要每次都重写整个环境的实验沙箱，CombatBench 就是为这个目的设计的。

## 为什么选择 CombatBench

- **框架优先**：`envs/framework` 层将物理后端、世界规则、运行时调度和观测输出分离。
- **机器人友好的契约**：状态、动作、观测和派生信号都是显式接口，而不是临时的数组。
- **为实验而构建**：同一运行时可支持强化学习训练、脚本基线、消融实验、评估比赛和未来的新机器人。
- **人形对战作为压力测试**：平衡、接触、扰动、非对称策略和自我博弈都在一个环境中展现。
- **实用的媒体循环**：无头 EGL 渲染使得生成回放视频和调试视图变得简单直接。

## `envs/` 系统：它的有趣之处

CombatBench 最有价值的资产是 `envs/` 系统。

它围绕一个简单的理念设计：**保持物理沙箱纯粹，让其他一切可组合**。

### `framework`：可复用核心

`envs/framework` 层是 CombatBench 不仅仅是一次性基准的主要原因。

- **`BaseSimulator`** 让后端专注于物理步进、状态读写和动作应用。
- **世界插件** 处理客观世界逻辑，如约束、裁决、事件和指标。
- **观测插件** 构建面向任务的输出，如观测、奖励、调试视图或分析特性。
- **`EnvRuntime`** 作为稳定的公共运行时入口，编排双方动作和完整回合生命周期。

对用户而言，这打开了广阔的空间：

- **在相同运行时训练不同算法** 而无需重写环境核心。
- **交换观测定义** 同时保持相同对战世界和仿真器。
- **添加新规则插件** 用于不倒地、生命值系统、扰动或课程阶段。
- **直接基于标准数据访问接口构建评估器和可视化工具。**
- **移植到未来机器人或后端** 同时保留大部分周围运行时逻辑。

这意味着 CombatBench 可用作：

- **强化学习基准**
- **机器人控制沙箱**
- **自我博弈实验平台**
- **比赛评估和回放系统**

## `humanoid21`：首个完整环境实现

`envs/humanoid21` 是当前的旗舰实现。

它围绕新框架契约打包了一个 21 自由度人形机器人对战环境，其设计对学习系统和环境工程都友好。

- **归一化位置控制** 保持动作接口稳定且有界。
- **结构化数据契约** 分离静态数据、核心物理状态和派生学习信号。
- **以自我为中心的状态设计** 使观测空间在策略和出生布局间更可复用。
- **面向插件的运行时** 保持对战逻辑、观测逻辑和仿真器机制解耦。
- **从第一天起的双智能体设置** 使支持自我博弈和评估器对策略工作流自然。

实际上，`humanoid21` 为你提供了想象和构建的空间：

- **站立/恢复/防倒地控制器**
- **接触感知的运动和击打策略**
- **从生存到激进进攻的自我博弈课程**
- **集中式评论家或分散式演员设置**
- **未来仅视觉或部分观测变体**

## 基线：起点，而非上限

`baseline/` 目录是 CombatBench 立即可用的地方。

当前的 `baseline/humanoid21` 路线为训练人形机器人策略提供了基于 GRPO 的具体起始路径，从最基础的能力开始：**站立**，然后迈向**扰动鲁棒站立**。

这作为以下两方面都有用：

- **环境栈的健全性检查**
- **训练集成的参考实现**
- **更强对战策略的发射台**

你无需照搬基线。重点是框架和基线已经以某种方式结合，使你的下一个实验更便宜地启动。

## 项目结构

- `assets/`：MuJoCo XML 模型、纹理、网格和媒体资产。
- `envs/`：环境运行时框架和具体环境。
  - `framework/`：后端契约、运行时编排、插件系统。
  - `humanoid21/`：当前 21 自由度人形机器人实现。
- `policy/`：策略接口和参考策略。
- `baseline/`：训练基线和可复现的起始点。
- `docs/`：基准规则和支持设计文档。

## 安装

### 要求

- Python 3.8+
- MuJoCo 3.x
- Gymnasium
- NumPy
- OpenCV (cv2)

### 设置

```bash
# 克隆仓库
# git clone https://github.com/laddermoon/combatbench.git
# cd combatbench

# 安装依赖（确保已安装 mujoco）
pip install mujoco gymnasium numpy opencv-python imageio egl
```

## 快速开始

在当前 `humanoid21` 环境中运行一回合并保存视频：

```bash
# 无显式策略运行（默认站立行为）
python envs/humanoid21/run_round.py --duration 10 --video test.mp4

# 使用随机策略运行
python envs/humanoid21/run_round.py --policy-a random --duration 5 --video test.mp4

# 运行两个不同策略
python envs/humanoid21/run_round.py \
  --policy-a random \
  --policy-b standing \
  --duration 15 --video match.mp4
```

## 关键文档

如果你想要设计契约而不是 README 摘要，直接查看这些文档：

- **框架架构**：[`envs/framework/DESIGN.md`](envs/framework/DESIGN.md)
- **Humanoid21 观测设计**：[`envs/humanoid21/OBSERVATION_zh.md`](envs/humanoid21/OBSERVATION_zh.md)
- **Humanoid21 数据契约**：[`envs/humanoid21/DATASPEC.md`](envs/humanoid21/DATASPEC.md)
- **Humanoid21 控制契约**：[`envs/humanoid21/CONTROLSPEC.md`](envs/humanoid21/CONTROLSPEC.md)
- **Humanoid21 基线指南**：[`baseline/humanoid21/README.md`](baseline/humanoid21/README.md)

附加项目文档：

- [对战规则](docs/RULE.md) / [中文规则](docs/RULE_zh.md)
- [环境详情](docs/ENVIRONMENT.md) / [中文环境](docs/ENVIRONMENT_zh.md)
- [策略提交指南](docs/SUBMISSION_GUIDE.md) / [中文提交指南](docs/SUBMISSION_GUIDE_zh.md)

## 策略接口

所有对战策略必须继承自 `BaseCombatPolicy` 并实现 `act()` 方法：

```python
from combatbench.policy import BaseCombatPolicy
import numpy as np

class MyPolicy(BaseCombatPolicy):
    def __init__(self, observation_space=None, action_space=None, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        # 你的初始化

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """返回形状为 (21,) 的动作数组，值范围在 [-1, 1]"""
        # 你的动作计算
        return action
```

完整接口定义请参见 [`policy/base.py`](policy/base.py)。

## 贡献

我们欢迎贡献！请遵循标准开源拉取请求工作流。
