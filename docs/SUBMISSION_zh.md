# 策略提交指南 (Policy Submission Guide)

欢迎参与 CombatBench 的机器人对战挑战！本指南将向您展示如何将您的控制策略（Policy）提交到对战平台。

## 1. 策略接口规范

您提交的策略需要被封装为一个符合以下接口规范的 Python 类。对战平台将会实例化您的类，并在每一步调用它以获取动作。

```python
import numpy as np

class CombatPolicy:
    def __init__(self, observation_space, action_space):
        """
        初始化您的策略。
        
        参数:
            observation_space: 观测空间定义（包含维度和限制等）
            action_space: 动作空间定义
        """
        self.action_dim = action_space.shape[0]

    def act(self, obs, info=None):
        """
        根据当前的环境观测向量计算动作。
        
        参数:
            obs: np.ndarray, 当前帧的观测向量 (详情参见 docs/OBSERVATION_zh.md)
            info: dict, 可选的附加环境信息
            
        返回:
            action: np.ndarray, 动作向量 (期望输出的关节目标扭矩或位置)
        """
        # 示例：随机策略
        return np.random.uniform(-0.1, 0.1, self.action_dim)
        
    def reset(self):
        """
        当一个回合 (episode) 结束或环境重置时调用。
        可以在此方法中清理策略的内部状态（例如清空 RNN 的隐藏状态）。
        """
        pass
```

## 2. 依赖与限制

为了保证比赛的绝对公平性和系统运行的安全性，您的代码必须遵循以下限制：

- **允许的第三方库**：默认情况下只允许使用平台预装的库（如 `numpy`, `torch`, `scipy`）。如果您的策略需要依赖特定的包，请在提交的目录中附带 `requirements.txt` 文件。
- **文件大小限制**：您提交的模型权重文件（如 `.pt`, `.pth`, `.onnx`）总大小不得超过 **1GB**。
- **推理时间限制**：您的 `act` 函数必须在 **10 毫秒** (100Hz 的频率) 内返回动作结果。如果发生超时，平台将自动采用“零动作”或者直接判决当前回合负。

## 3. 提交流程

1. 将您的策略类代码保存命名为 `policy.py`。
2. 将您的预训练模型权重文件（如果有）放置在同一个目录下，例如命名为 `model.pt`。
3. 如果还有其他的辅助文件（如自定义的配置参数文件），也请一并放置在该目录下。
4. 使用官方提供的提交工具脚本，将整个目录验证并打包成 ZIP 压缩包。

目录结构示例：
```
my_submission/
├── policy.py       # 必须包含定义的 CombatPolicy 类
├── model.pt        # 您的预训练权重
└── config.json     # 自定义的配置文件
```

## 4. 提交工具使用 (CLI)

官方提供了一个命令行工具，帮助您在提交之前验证代码规范并进行打包压缩：

```bash
# 验证您的策略接口是否符合官方规范
python utils/submit_tool.py verify my_submission/

# 验证无误后，进行打包提交
python utils/submit_tool.py pack my_submission/
```

这会在当前目录生成一个名为 `submission.zip` 的压缩文件。

## 5. 前往 Web 平台

获得 `submission.zip` 文件后，请前往官方对战平台：

**🔗 [https://arena.combatbench.com](https://arena.combatbench.com) (示例链接)**

在平台上登录您的账号，进入 "Submit" 页面，上传生成的 ZIP 压缩包，即可参与天梯榜单匹配。
