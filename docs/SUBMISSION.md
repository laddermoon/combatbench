# 策略提交指南 (Policy Submission Guide)

欢迎参与 CombatBench 的机器人对战挑战！本指南将向您展示如何提交您的控制策略（Policy）到对战平台。

## 1. 策略接口规范

您提交的策略需要封装为一个符合以下接口的 Python 类。平台将实例化您的类并调用它来获取动作。

```python
import numpy as np

class CombatPolicy:
    def __init__(self, observation_space, action_space):
        """
        初始化您的策略。
        
        Args:
            observation_space: 字典，包含机器人的观测空间定义
            action_space: dict, 机器人的动作空间定义
        """
        self.action_dim = action_space.shape[0]

    def act(self, obs, info=None):
        """
        根据当前环境观测计算动作。
        
        Args:
            obs: np.ndarray, 当前帧的观测向量 (详情见 docs/OBSERVATION.md)
            info: dict, 可选的附加信息
            
        Returns:
            action: np.ndarray, 动作向量 (期望各关节的扭矩或位置)
        """
        # 示例：随机策略
        return np.random.uniform(-0.1, 0.1, self.action_dim)
        
    def reset(self):
        """
        回合结束或环境重置时调用，用于清理内部状态（如 RNN 隐藏状态）。
        """
        pass
```

## 2. 依赖限制

为了保证比赛的公平性和系统的安全性，您的代码需要遵循以下限制：

- **只允许使用平台预装的库**：如 `numpy`, `torch`, `scipy`。如果您的策略需要特殊的包，请在提交时附带 `requirements.txt`。
- **文件大小限制**：模型权重文件 (.pt, .pth, .onnx) 总大小不应超过 **1GB**。
- **推理时间限制**：您的 `act` 函数必须在 **10 毫秒** (100Hz 频率) 内返回动作，超时将被判负或强制使用零动作。

## 3. 提交流程

1. 将您的策略类保存为 `policy.py`。
2. 将您的模型权重（如果有）放在同一目录下，例如 `model.pt`。
3. 如果有其他依赖辅助文件（如配置），也放在该目录下。
4. 使用提供的提交工具将整个目录打包并提交。

目录结构示例：
```
my_submission/
├── policy.py       # 必须包含 CombatPolicy 类
├── model.pt        # 您的预训练权重
└── config.json     # 自定义配置文件
```

## 4. 提交工具使用 (CLI)

我们提供了一个命令行工具帮助您验证和打包提交：

```bash
# 验证策略接口是否符合规范
python utils/submit_tool.py verify my_submission/

# 打包提交
python utils/submit_tool.py pack my_submission/
```

这会生成一个 `submission.zip` 文件。

## 5. 前往 Web 平台

获得 `submission.zip` 后，请前往我们的官方对战平台：

**🔗 [https://arena.combatbench.com](https://arena.combatbench.com) (示例链接)**

在平台上登录您的账号，进入 "Submit" 页面，上传生成的 ZIP 文件即可参与天梯匹配。
