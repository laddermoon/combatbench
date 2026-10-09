# Policy 模块

> 类型：指南

本模块存放**参考实现策略**，以及提交给 combatbench.tech 平台的策略目录规范。

> **标准接口定义位置**：`envs/framework/policy.py` 中的 `Policy` 抽象基类是
> 整个 combatbench 框架对「策略」的**唯一契约来源**。本目录下的策略（以及
> 任何自定义策略）只需实现这个 ABC，就能插进 `EpisodeRunner` /
> `RoundRunner` / `MatchRunner` / rollout 层。

## Policy 目录结构规范

### 必需文件

- **`policy_blueprint.yaml`**: 策略蓝图，框架的加载入口（见下文 schema）
- **策略代码文件**: 蓝图 `cls` 字段指向的 Python 文件（常用 `policy.py`）
  - 必须包含一个实现 `Policy` 接口的类（类名自定义）
  - 必须实现 `act()` 方法

### 可选文件

- **`model.pt`**: 模型权重等资产，在 `config` 里用 `${DIR}` 相对引用
- **`requirements.txt`**: 依赖包列表（提交平台时使用）

### 示例 Policy 目录

```
my_policy/
├── policy_blueprint.yaml   # 必需：入口配置
├── policy.py               # 策略实现
├── model.pt                # 可选：权重
└── requirements.txt        # 可选：依赖
```

`policy_blueprint.yaml` 示例（`${DIR}` 会被替换为蓝图所在目录的绝对路径）：

```yaml
version: 1
cls: "file:${DIR}/policy.py:MyCombatPolicy"   # 或 "package.module:Class"
config:
  model_path: "${DIR}/model.pt"
```

加载方式：

```python
from envs.framework.policy import PolicyBlueprint

policy = PolicyBlueprint.load("my_policy/policy_blueprint.yaml").build()
```

## Policy 接口

所有策略实现 `envs/framework/policy.py` 的 `Policy` 抽象基类。

### Policy

```python
class Policy(ABC):
    """策略抽象基类——框架对策略的唯一契约。"""

    @abstractmethod
    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[Any]]:
        """根据观测计算动作。

        Args:
            observation: 由 simulator.get_observation() 返回的该 agent 的
                观测值。类型由策略与环境约定，框架不约束。
            want_extra: runner 是否需要可选的 extra 载荷（on-policy RL
                记录 log_prob / value 等）。False 时可返回 extra=None。

        Returns:
            (action, extra) 元组：
            - action: 传给 simulator.step 的动作，不允许为 None；
              humanoid21 下是 shape=(21,) 的 [-1, 1] 数组。
            - extra: 可选辅助载荷（dict / None）。
        """
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """新回合开始时重置内部状态。

        seed 是 runner 从 base_seed 通过 SeedSequence 派生的**每策略
        子种子**，有 RNG 的策略应该用它重置随机源以保证 rollout 可复现。
        默认 no-op。
        """

    def to_blueprint(self, dest_path=None) -> "PolicyBlueprint":
        """可选钩子：把策略自身导出为可部署的 PolicyBlueprint
        （含源码/权重落盘）。不支持则抛 NotImplementedError。"""
```

> 训练用的随机策略另见 `baseline/framework/ppo/stochastic_policy.py` 的
> `StochasticPolicy` 接口（`sample()` / `evaluate_actions()`）——它与
> `Policy` 正交：rollout 时由 `SamplingPolicy` 包装成普通 `Policy`。

## 内置策略

### 1. RandomCombatPolicy

生成均匀随机动作的策略，用于基线对比和冒烟测试。

**目录结构**:
```
policy/random/
└── policy.py
```

**使用方法**（蓝图见 `policy/blueprints/random.yaml`）:
```python
from policy.random.policy import RandomCombatPolicy

policy = RandomCombatPolicy(
    scale=0.1,      # 动作范围 [-scale, scale]
    seed=42,        # 构造期种子；runner 每回合会用派生种子重播种
    action_dim=21,
)
action, extra = policy.act(obs)   # act 返回 (action, extra) 元组
```

### 2. StandingCombatPolicy

`policy/humanoid21/standing/`：固定站立姿态的脚本策略，蓝图见
`policy/blueprints/humanoid21/standing.yaml`。

### 3. 训练导出策略

历史快照库 `policy/baseline/`（81 个 fight/follow 快照）已随环境/策略
契约演进废弃删除（P-POL-1）。当前训练产出的可部署蓝图在各训练
run 的 `baseline/runs/<run>/policy_exports/*.yaml`，加载方式同样是
`PolicyBlueprint.load(<dir>/policy_blueprint.yaml).build()`。

## 实现自定义 Policy

### 步骤 1: 创建 Policy 目录

```bash
mkdir -p my_policy
cd my_policy
```

### 步骤 2: 创建 policy.py

```python
# my_policy/policy.py
import numpy as np
from envs.framework.policy import Policy

class MyCombatPolicy(Policy):
    """我的自定义策略"""

    def __init__(self, action_dim: int = 21, **_ignored):
        self.action_dim = action_dim
        self.counter = 0

    def act(self, observation, *, want_extra=False):
        """返回 (action, extra)。action shape=(21,) float32。"""
        action = np.zeros(self.action_dim, dtype=np.float32)
        # ... 计算动作 ...
        return action, None

    def reset(self, seed=None):
        """重置内部状态；如有 RNG 请使用 seed 重新播种。"""
        self.counter = 0
```

> ctor 建议吸收 `**_ignored` kwargs——蓝图 config 中多余的键会被透传，
> 宽容处理可以避免无关参数导致崩溃。

### 步骤 3: 创建 policy_blueprint.yaml

```yaml
version: 1
cls: "file:${DIR}/policy.py:MyCombatPolicy"
config:
  action_dim: 21
```

### 步骤 4: 添加依赖（可选）

如果策略需要额外的包（如 PyTorch），创建 `requirements.txt`：

```txt
torch>=2.0.0
numpy>=1.20.0
```

## Policy 规范总结

| 项目 | 要求 | 说明 |
|------|------|------|
| **目录结构** | 必须是独立目录 | 每个策略一个目录 |
| **policy_blueprint.yaml** | 必需 | 入口配置，`cls` 支持 `pkg.mod:Class` 与 `file:${DIR}/x.py:Class` |
| **策略代码** | 必需 | 蓝图 `cls` 指向的 .py 文件，含实现 `Policy` 的类 |
| **requirements.txt** | 可选 | 额外依赖包列表 |
| **act(obs, *, want_extra=False)** | 必须实现 | 返回 `(action, extra)` 元组；action 不允许为 None |
| **reset(seed=None)** | 可选实现 | 接受 per-episode 子种子；有 RNG 必须用它重播种 |
| **to_blueprint()** | 可选实现 | 自导出为可部署蓝图（平台提交用） |
| **动作值域** | 必须 | humanoid21 下为 [-1, 1] 的 (21,) 数组 |

提交平台时的打包规范详见 [`docs/SUBMISSION.md`](../docs/SUBMISSION.md)。
