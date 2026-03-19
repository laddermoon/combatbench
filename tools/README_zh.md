# CombatBench 工具集

此目录包含用于辅助运行、评估和测试 CombatBench 环境的实用脚本。

## 1. 对战运行器 (`run_match.py`)

一个用于模拟两个策略 (Policy) 之间单回合对战的实用工具类。

### 在代码中使用

您可以将 `MatchRunner` 类导入到您自己的评估或训练循环中，以轻松策划一场比赛：

```python
from tools.run_match import MatchRunner

# 假设 policy_a 和 policy_b 是您训练好的策略对象
# 策略必须实现：`act(obs, info)` 和 `reset()` 方法
runner = MatchRunner(
    policy_a=my_red_policy,
    policy_b=my_blue_policy,
    match_duration=30.0 # 秒
)

# 运行比赛，并可选择保存输出视频
result = runner.run(save_video_path="match_output.mp4")

print(f"比赛结果: {result['end_reason']}")
print(f"最终血量 - 红方: {result['scores']['robot_a']}, 蓝方: {result['scores']['robot_b']}")
```

### 命令行使用

您也可以直接通过命令行运行它，使用随机动作（虚拟策略）来测试仿真环境：

```bash
python tools/run_match.py --video output_video.mp4 --duration 10.0
```

**参数说明:**
- `--video`: 保存 MP4 回放视频的路径（例如 `match.mp4`）。如果省略，则不保存视频。
- `--duration`: 比赛的最大时间限制，单位为秒（默认为 30.0）。
