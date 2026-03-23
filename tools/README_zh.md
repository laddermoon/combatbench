# CombatBench 工具集

此目录包含用于辅助运行、评估和测试 CombatBench 环境的实用脚本。

## 1. 回合运行器 (`run_round.py`)

统一的脚本，用于在两个策略之间运行格斗回合。所有策略都使用一致的规格格式加载，支持构造函数参数。

### 策略规格格式

#### 1. Python 模块路径
```bash
# 简单形式（无参数）
--policy-a combatbench.policy.RandomCombatPolicy

# 带参数（查询字符串格式）
--policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2&seed=42"

# SB3 模型
--policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=model.zip&device=cuda"
```

#### 2. Python 文件路径
```bash
# 带类名
--policy-a "path/to/policy.py:MyPolicy"

# 带类名和参数
--policy-a "path/to/policy.py:MyPolicy?scale=0.5"
```

#### 3. 配置文件（JSON）
```bash
--policy-a "@policy_config.json"
```

配置文件格式 (`policy_config.json`)：
```json
{
  "type": "combatbench.policy.RandomCombatPolicy",
  "params": {
    "scale": 0.2,
    "seed": 42
  }
}
```

或用于 SB3 模型：
```json
{
  "type": "combatbench.baseline.sb3.policies.SB3CombatPolicy",
  "params": {
    "model_path": "runs/stand_v1/model_final.zip",
    "device": "cuda"
  }
}
```

#### 4. 默认（不动策略）
```bash
# 如果省略 --policy-a，则使用 StandingCombatPolicy
python tools/run_round.py --duration 10 --video test.mp4
```

### 参数类型支持

查询字符串格式支持自动类型转换：
- **数字**: `?scale=0.5`, `?count=10`
- **布尔值**: `?enabled=true`, `?debug=false`
- **字符串**: `?name=my_policy`, `?model_path=model.zip`
- **JSON 值**: 对于复杂类型，使用 JSON 编码的值：
  - 列表: `?list=[1,2,3]`
  - 对象: `?config={"key":"value"}`
  - 空值: `?optional=null`

### 使用 RoundRunner 类

您可以从 `combatbench.envs` 导入 `RoundRunner` 类到您自己的评估或训练循环中：

```python
from combatbench.envs import RoundRunner
from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy

# 使用两个策略创建运行器
runner = RoundRunner(
    policy_a=RandomCombatPolicy(scale=0.1),
    policy_b=StandingCombatPolicy(),
    match_duration=30.0,  # 秒
    render_mode="rgb_array",
)

# 运行回合，并可选择保存输出视频
result = runner.run(save_video_path="round_output.mp4")

# 通过 RoundResult 数据类访问结果
print(f"获胜者: {result.winner}")
print(f"步数: {result.steps}")
print(f"最终血量 - 红方: {result.scores['robot_a']}, 蓝方: {result.scores['robot_b']}")
print(f"造成伤害 - A: {result.damage_dealt['robot_a']}, B: {result.damage_dealt['robot_b']}")
```

### 命令行使用示例

不使用策略运行（双方都使用 StandingCombatPolicy）：
```bash
python tools/run_round.py --duration 10 --video test.mp4
```

使用 Python 模块策略运行：
```bash
python tools/run_round.py --policy-a combatbench.policy.RandomCombatPolicy \
                         --policy-b combatbench.policy.StandingCombatPolicy
```

使用 SB3 模型运行：
```bash
python tools/run_round.py \
  --policy-a "combatbench.baseline.sb3.policies.SB3CombatPolicy?model_path=runs/stand_v1/model_final.zip" \
  --video match.mp4
```

带参数运行：
```bash
# 自定义尺度的随机策略
python tools/run_round.py --policy-a "combatbench.policy.RandomCombatPolicy?scale=0.2"

# 带多个参数的自定义策略
python tools/run_round.py --policy-a "mypolicy.MyPolicy?model_path=model.zip&noise=true"
```

使用配置文件运行：
```bash
python tools/run_round.py --policy-a "@configs/policy_a.json" --policy-b "@configs/policy_b.json"
```

**命令行参数：**

| 参数 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--policy-a` | `--model-a` | 机器人 A（红方）的策略 | `StandingCombatPolicy` |
| `--policy-b` | `--model-b` | 机器人 B（蓝方）的策略 | `StandingCombatPolicy` |
| `--duration` | `--match-duration` | 回合持续时间（秒） | `30.0` |
| `--control-frequency` | `--fps` | 控制频率（Hz） | `20` |
| `--initial-distance` | | 机器人之间的初始距离（米） | `2.0` |
| `--phase` | | 控制器配置的训练阶段 | `None` |
| `--non-fall-mode` | | 启用姿态限制（防跌倒模式） | `False` |
| `--non-fall-pitch-limit-deg` | | 防跌倒模式的俯仰角限制 | `15.0` |
| `--non-fall-roll-limit-deg` | | 防跌倒模式的横滚角限制 | `10.0` |
| `--damage-scale` | | 伤害缩放因子 | `100.0` |
| `--video` | `--output` | 保存视频的路径 | `None`（不保存视频） |
| `--device` | | 策略推理的设备 | `auto` |
| `--quiet` | `-q` | 抑制进度输出 | `False` |

### 策略接口

策略必须实现：
```python
def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
    """返回形状为 (21,) 的动作数组，值在 [-1, 1] 范围内"""
    pass

def reset(self) -> None:
    """在回合开始时重置内部状态（可选）"""
    pass
```

### RoundResult 字段

- `steps`: 采取的总步数
- `end_reason`: 回合结束的原因
- `winner`: 哪个机器人获胜（'robot_a'、'robot_b' 或 'draw'）
- `scores`: 双方的最终血量得分
- `initial_scores`: 初始血量得分（通常为 100）
- `damage_dealt`: 每个机器人造成的总伤害
- `total_reward`: 累计的总奖励
- `video_frames`: 捕获的视频帧数
