# 机器人站立训练指南

本文档说明如何在当前 `combatbench` baseline 上训练机器人站立、验证 `10 秒不倒`、以及导出成功视频。

## 1. 目标

当前站立基线的最小成功标准是：

- 使用 `stand` 阶段训练得到策略
- 在 `10s @ 20Hz` 的评估配置下稳定站满 `200` control steps
- 能导出一段可复核的 mp4 视频

## 2. 关键控制设定

当前站立不是直接学习原始 torque，而是使用：

- `reference standing pose`
- `residual joint position action`
- 环境内部固定 PD 控制
- 每个 physics substep 重算一次 PD torque

当前默认站立控制器参数在 `baseline/sb3/selfplay_env.py` 中：

- `Kp = 12.0`
- `Kd = 0.2`
- `STAND_ACTION_SCALE_MULTIPLIER = 0.8`

这套配置已经验证可让 `stand_env_residual_smoke_v3/best_model.zip` 稳定站满 `10s`。

## 3. 训练前检查

在 `things/combatbench` 目录下先运行：

```bash
python3 -m combatbench.baseline.sb3.validate_env
```

期望结果：

- stand / fight 两个 phase 都能 reset / step
- SB3 wrapper 可以正常返回观测和奖励
- 没有维度错误、NaN、初始化错误

## 4. 启动站立训练

推荐先从一个 smoke run 开始：

```bash
python3 -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 100000 \
  --run-name stand_smoke_v1 \
  --device cpu
```

说明：

- `stand` 阶段默认 `match_duration=10.0`
- `stand` 阶段默认 `initial_distance=2.0`
- 输出目录默认写到：
  - `combatbench/baseline/sb3/runs/<run-name>/`

如果要认真训练，可把 `timesteps` 提高到 `300k - 1M+`。

## 5. 训练产物

训练目录下重点关注：

- `run_config.json`
- `train.log`
- `best_model/best_model.zip`
- `checkpoints/`
- `model_final.zip`

如果你是后台跑训练，建议把 stdout/stderr 重定向到 `train.log`，并周期性查看评估输出。

## 6. 评估是否达成 10 秒站立

训练后用 deterministic self-play 评估：

```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode selfplay \
  --model combatbench/baseline/sb3/runs/stand_smoke_v1/best_model/best_model.zip \
  --phase stand \
  --episodes 5 \
  --duration 10 \
  --control-frequency 20 \
  --initial-distance 2.0
```

成功标准：

- 平均步数接近或达到 `200`
- 最好能看到 `5/5` 回合跑满 `200` steps
- 结束原因应为：
  - `Time limit reached (10.0s), draw`

## 7. 导出站立成功视频

推荐使用独立脚本入口：

```bash
python3 combatbench/run_policy_video.py \
  --mode shared_env \
  --model combatbench/baseline/sb3/runs/stand_smoke_v1/best_model/best_model.zip \
  --phase stand \
  --duration 10 \
  --control-frequency 20 \
  --initial-distance 2.0 \
  --video combatbench/baseline/sb3/runs/stand_smoke_v1/stand_success_10s.mp4 \
  --device cpu
```

这个脚本复用了已经验证成功的直连渲染路径，适合以后重复导出视频。

## 8. 当前已验证的成功样例

当前仓库内已经验证成功的站立模型：

- `combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/best_model/best_model.zip`

对应成功视频：

- `combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/stand_success_10s.mp4`
- `combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/stand_success_10s_reexport.mp4`

## 9. 常见问题

### 9.1 奖励变好了，但还是站不满 10 秒

不要只看 reward，优先看 deterministic 评估的 episode length。

### 9.2 视频导出失败

优先使用：

- `python3 combatbench/run_policy_video.py ...`

而不是依赖 `python -m ...` 的模块路径去调渲染。

### 9.3 站立回归

如果出现站立退化，优先检查：

- 是否仍然是 residual position action
- 是否仍然是 fixed PD
- 是否仍然在每个 physics substep 重算 PD torque
- `Kp/Kd` 是否被改回更弱的配置

## 10. 站立之后的下一步

推荐顺序：

1. 先把 `stand` 训练稳定到 `10 秒不倒`
2. 用 `best_model.zip` 作为 `fight` 阶段初始化
3. 再验证伤害逻辑、敌方观测和对战视频导出
