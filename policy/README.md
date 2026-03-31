# Policy 模块

本模块定义了策略的抽象接口和参考实现。

## 目录结构

```
policy/
├── __init__.py           # 模块导出
├── base.py               # BaseCombatPolicy 抽象基类
├── random/               # 随机策略
│   └── policy.py         # RandomCombatPolicy
└── standing/             # 静止策略
    └── policy.py         # StandingCombatPolicy
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
            action: 动作数组，值域为 [-1, 1]
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

```python
from combatbench.policy import RandomCombatPolicy

policy = RandomCombatPolicy(
    scale=0.1,      # 动作范围 [-scale, scale]
    seed=42         # 随机种子
)
action = policy.act(obs)
```

### 2. StandingCombatPolicy

返回零动作的策略，智能体保持当前姿态。

```python
from combatbench.policy import StandingCombatPolicy

policy = StandingCombatPolicy()
action = policy.act(obs)  # 返回全零动作
```

## 实现自定义策略

### 基本示例

```python
import numpy as np
from combatbench.policy import BaseCombatPolicy

class MyPolicy(BaseCombatPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 你的初始化代码

    def act(self, obs, info=None):
        # 你的策略逻辑
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        # ... 计算动作 ...
        return action

    def reset(self):
        # 重置内部状态（可选）
        pass
```

### 带内部状态的策略

```python
import numpy as np
from combatbench.policy import BaseCombatPolicy

class MemoryPolicy(BaseCombatPolicy):
    def __init__(self, memory_size=10, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.history = []

    def act(self, obs, info=None):
        # 保存历史观测
        self.history.append(obs.copy())
        if len(self.history) > self.memory_size:
            self.history.pop(0)

        # 基于历史计算动作
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        # ... 使用 self.history 计算动作 ...
        return action

    def reset(self):
        # 清空历史
        self.history.clear()
```

### 加载训练好的模型

```python
import numpy as np
from combatbench.policy import BaseCombatPolicy

class TrainedPolicy(BaseCombatPolicy):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        # 加载你的模型
        import torch
        return torch.load(path)

    def act(self, obs, info=None):
        # 将观测转换为模型输入
        model_input = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.model(model_input)
        return action.cpu().numpy().astype(np.float32)

    def reset(self):
        # 如果模型有状态（如RNN），在这里重置
        if hasattr(self.model, 'reset'):
            self.model.reset()
```

## 策略开发建议

1. **动作裁剪**：确保返回的动作在 `[-1, 1]` 范围内
2. **类型正确**：返回 `np.float32` 类型的 numpy 数组
3. **错误处理**：在 `act` 方法中捕获异常，避免崩溃
4. **状态重置**：如果策略有内部状态，记得实现 `reset` 方法
5. **性能考虑**：`act` 方法会被频繁调用，注意计算效率

## 注意事项

- 策略的 `act` 方法应该尽可能快，因为每个控制步都会调用
- 环境会自动裁剪动作到 `[-1, 1]` 范围
- `info` 字典包含额外的环境信息，可以用于更复杂的策略
- 观测值已经是归一化的，可以直接用于神经网络输入
