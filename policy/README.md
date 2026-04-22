# Policy 模块

本模块提供策略的 **ABC 基类** 和参考实现，供 `load_policy(...)` 动态加载。

> **标准接口定义位置**：`envs/framework/policy.py` 中的 `Policy` Protocol 是
> 整个 combatbench 框架对「策略」的**唯一契约来源**。本包的
> `BaseCombatPolicy` 是该 Protocol 的一个 ABC 实现，外加 observation/action
> space、kwargs 透传、自动发现等便利功能；不需要这些便利的代码可以直接
> duck-typed 实现 Protocol，同样能插进 `EpisodeRunner` / `RoundRunner` /
> `ParallelRunner`。

## Policy 目录结构规范

### 必需文件

- **`policy.py`**: 必须存在，包含一个实现了 `BaseCombatPolicy` 的类
  - 类名可以自定义（如 `MyCombatPolicy`）
  - 必须继承 `BaseCombatPolicy`
  - 必须实现 `act()` 方法

### 可选文件

- **`requirements.txt`**: 依赖包列表（如果策略需要额外的 Python 包）

### 示例 Policy 目录

```
my_policy/
├── policy.py            # 必须
└── requirements.txt     # 可选（如需要 torch, tensorflow 等）
```

## Policy 接口

所有策略必须继承自 `BaseCombatPolicy` 并实现 `act` 方法。

### BaseCombatPolicy

```python
class BaseCombatPolicy(ABC):
    """策略的抽象基类。实现 envs.framework.policy.Policy Protocol。"""

    ACTION_DIM = 21  # 默认动作维度（humanoid21）

    def __init__(
        self,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        **kwargs,
    ):
        """初始化策略。kwargs 来自 load_policy 的 query string 参数透传。"""

    @abstractmethod
    def act(self, observation: Any) -> np.ndarray:
        """根据观测计算并返回动作。

        Args:
            observation: 由绑定的 observer plugin 的 get_output() 返回的值。
                通常是 1D float32 观测向量。

        Returns:
            action: 可被 np.asarray(dtype=float32) 转换的动作。不允许返回
                None——框架在这一层不支持「沿用上一步动作」的语义。
        """
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """在新回合开始时重置策略内部状态。

        seed 是 EpisodeRunner 从 base_seed 通过 SeedSequence 派生的**每策略
        子种子**，有 RNG 的策略应该用它重置自己的随机源以保证 rollout 可复现。
        默认为 no-op。
        """
        return None

    # 可选钩子：act_with_extras(observation) -> (action, extras_dict)
    # 开启 RolloutConfig.store_extras 时会用于记录 log_prob / value 等
    # 可选钩子：close() —— 资源释放；框架不会自动调用，caller 自行负责。
```

## 内置策略

### 1. RandomCombatPolicy

生成随机动作的策略，用于基线对比和测试。

**目录结构**:
```
policy/random/
└── policy.py
```

**使用方法**:
```python
from combatbench.policy import RandomCombatPolicy

policy = RandomCombatPolicy(
    scale=0.1,      # 动作范围 [-scale, scale]
    seed=42         # 随机种子
)
action = policy.act(obs)
```

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
from combatbench.policy import BaseCombatPolicy

class MyCombatPolicy(BaseCombatPolicy):
    """我的自定义策略"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        """根据观测计算动作。返回 shape=(21,) 的 float32 数组。"""
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        # ... 计算动作 ...
        return action

    def reset(self, seed=None) -> None:
        """重置内部状态；如有 RNG 请使用 seed 重新播种。"""
        self.counter = 0
```

### 步骤 3: 添加依赖（可选）

如果策略需要额外的包（如 PyTorch），创建 `requirements.txt`：

```txt
# my_policy/requirements.txt
torch>=2.0.0
numpy>=1.20.0
```

## Policy 规范总结

| 项目 | 要求 | 说明 |
|------|------|------|
| **目录结构** | 必须是独立目录 | 每个策略一个目录 |
| **policy.py** | 必需文件 | 必须包含实现 BaseCombatPolicy 的类 |
| **requirements.txt** | 可选文件 | 额外依赖包列表 |
| **类继承** | 必须 | 继承 BaseCombatPolicy（或直接实现 Policy Protocol） |
| **act(observation)** | 必须实现 | 返回 shape=(21,) 的 float32 数组，**不允许 None** |
| **reset(seed=None)** | 可选实现 | 接受 per-episode 子种子；有 RNG 必须用它重播种 |
| **act_with_extras** | 可选实现 | on-policy RL 记录 log_prob / value 时提供 |
| **动作值域** | 必须 | [-1, 1] 范围内的 float32 |
