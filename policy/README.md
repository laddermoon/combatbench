# Policy 模块

本模块定义了策略的抽象接口和参考实现。

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
    """策略的抽象基类"""

    ACTION_DIM = 21  # 动作空间维度

    def __init__(
        self,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        **kwargs
    ):
        """初始化策略"""

    @abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        """
        根据当前观测计算并返回动作

        Args:
            obs: 当前观测 (numpy数组)
            info: 环境信息字典 (可选)

        Returns:
            action: 动作数组，值域为 [-1, 1]，shape=(21,)
        """
        pass

    def reset(self) -> None:
        """
        在新回合开始时重置策略的内部状态

        默认实现不执行任何操作。
        有内部状态的策略（如RNN）应该重写此方法。
        """
        pass
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
        # 你的初始化代码
        self.counter = 0

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """
        根据观测计算动作

        Args:
            obs: 96维观测数组
            info: 环境信息（可选）

        Returns:
            action: 21维动作数组，值域 [-1, 1]
        """
        # 你的策略逻辑
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        # ... 计算动作 ...
        return action

    def reset(self) -> None:
        """重置内部状态"""
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
| **类继承** | 必须 | 继承 BaseCombatPolicy |
| **act() 方法** | 必须实现 | 返回 shape=(21,) 的动作数组 |
| **reset() 方法** | 可选实现 | 重置内部状态 |
| **动作值域** | 必须 | [-1, 1] 范围内的 float32 |
