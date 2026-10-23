# policy_capture — 策略 rollout 动捕管线

用训练好的策略在**标准环境 + 推力扰动**下跑 rollout，把逐帧物理状态录成
mocap 库可用的轨迹数据。产出的是"机器人原生动作"：所有速度/接触都是真实
动力学量，帧即合法初始状态（可直接做 RSI 锚点），天然 L2（执行即验证）。

## 数据定位

这是第三类数据源（`source=policy_rollout`），与 CMU/HDM05 人捕**分开管理**：

- 适合：跟踪参考、扰动恢复/受击类数据、RSI 初始状态、动力学可行性先验
- **不适合**直接当 AMP 人类风格正样本（会把判别器的风格分布拉回策略自身）

## 组成

| 文件 | 作用 |
|---|---|
| `push_probe_base_env.yaml` | 基础环境：标准 `Humanoid21Simulator`（站立起始）+ `RandomPushPlugin`（随机方向、30–250N 力度区间）。无倒地重置、无计分——摔倒原样记录，下游筛。 |
| `recorder.py` | `TrajectoryRecorder`（PostActionRecorder）：逐 action 步录全量状态，episode 末写 `ep_<seed>_<robot>.npz` + `ep_<seed>_meta.json` |
| `collect.py` | 批量驱动：跑 N 个 episode × 不同 seed，可选前几个渲 mp4 |
| `slice_clip.py` | 切片：把 ep npz 按 action-step 窗口切成 clip（npz + meta），可选同步切 mp4 |

## 用法

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 -m mocap.humanoid21.policy_capture.collect \
    --env-blueprint mocap/humanoid21/policy_capture/push_probe_base_env.yaml \
    --policy-a <policy_blueprint.yaml> \
    --episodes 8 --seed-base 1000 \
    --out-dir /data1/dev/policy_capture/<run_name> --render 2
```

### 策略接入

- **普通策略**（96 维 obs 训练）：直接把导出 blueprint yaml 传给 `--policy-a`。
- **gait-clock 系策略**（step/balance/follow 链，obs_dim=99）：用
  `policy/humanoid21/gait_clock/policy.py:GaitClockAdapterPolicy` 包一层——
  时钟是 action-step 的确定函数，适配器在策略内部补齐 3 维，环境保持标准 96 维：

  ```yaml
  version: 1
  cls: "policy.humanoid21.gait_clock.policy:GaitClockAdapterPolicy"
  config:
    gait_period: 40
    counter_step: 2
    inner:
      version: 1
      cls: "file:/abs/path/runs/<run>/policy/policy.py:ExportedStateMixtureBoundedStdTruncNormPolicy"
      config: {}
  ```

- A/B 可以挂不同策略（`--policy-b`），比如一边是受测策略、一边是脚本策略。
- 非标准观测的策略（如 CommandSimulator 66-dim）：不走 collect.py，直接
  `envs.framework.round_runner` + `--recorder
  mocap.humanoid21.policy_capture.recorder:TrajectoryRecorder?output_dir=...`，
  env/options 用该实验 videos/ 目录下的复现产物。

### 扰动扫描

改 yaml 里的 `force_magnitude_min/max`、`min/max_interval`、
`push_duration_steps`、`target_body` 生成变体环境；seed 由 EpisodeRunner
统一分发给插件，每个 episode 的推力时序都可复现。

## 数据契约

npz 字段与库 clip 对齐：`joint_angles`(rad) / `joint_velocities`(rad/s 真值) /
`root_pos` / `root_quat`(wxyz) / `root_linvel`+`root_angvel`(world) /
`feet_forces` / `body_xpos`+`body_xquat`（全部 body）/ `action` /
`push_active`+`push_force_vec`。meta.json 记 seed、终止原因、
**逐次推力事件**（起始步、时长、力向量、大小）——切"推→恢复"窗口直接用。

采样率 = 20 Hz（500 Hz 物理 ÷ 25 phy/action）。入库导出（15 点 keypoint
映射、push 窗口切片、index 条目）是下一步。

### 切片

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.policy_capture.slice_clip \
    --npz ep_1010_robot_a.npz --meta ep_1010_meta.json --robot robot_a \
    --start 10 --end 60 --out clips/<name>.npz \
    --video-in ep_1010.mp4 --video-out clips/<name>.mp4
```
