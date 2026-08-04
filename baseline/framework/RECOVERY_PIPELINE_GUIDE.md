# 迭代平衡恢复训练 — 脚本使用说明

本文档说明 Step 1-3 中实现的所有脚本和组件的使用方法。

## 目录

- [架构概览](#架构概览)
- [Step 1: ImpulsePerturbationPlugin — 扰动状态生成](#step-1-impulseperturbationplugin--扰动状态生成)
  - [test_impulse_plugin.py — 插件检验脚本](#test_impulse_pluginpy--插件检验脚本)
- [Step 2: 边界测绘](#step-2-边界测绘)
  - [probe_impulse_boundary.py — 网格扫描存活率](#probe_impulse_boundarypy--网格扫描存活率)
  - [plot_impulse_boundary.py — 可视化热力图](#plot_impulse_boundarypy--可视化热力图)
- [Step 3: 状态池生成](#step-3-状态池生成)
  - [generate_state_bank.py — 批量生成标注数据](#generate_state_bankpy--批量生成标注数据)
- [蓝图文件](#蓝图文件)
- [完整工作流示例](#完整工作流示例)

---

## 架构概览

```
                         ImpulsePerturbationPlugin
                                  │
                   ┌──────────────┼──────────────┐
                   ▼              ▼              ▼
            test_impulse     probe_impulse   generate_state_bank
            _plugin.py       _boundary.py    .py
            (验证)           (边界测绘)       (状态池生成)
                                  │                │
                                  ▼                ▼
                          plot_impulse         .npz 文件
                          _boundary.py        (states+obs+labels)
                              │                      │
                              ▼                      ▼
                          .png 热力图          Step 4: StateBankInitPlugin
                                               (从状态池加载注入)
```

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `ImpulsePerturbationPlugin` | `envs/humanoid21/disturbance_plugins.py` | 内部 sim + 策略生成物理合理的扰动状态 |
| `StateCapturePlugin` | `envs/humanoid21/disturbance_plugins.py` | 在第一个 action step 前捕获 core_state + observation |
| `StateCaptureObserver` | `envs/humanoid21/disturbance_plugins.py` | 通过 observer_outputs 暴露捕获的数据 |
| `ImbalanceTerminationPlugin` | `baseline/humanoid21/plugins/imbalance_termination.py` | 检测机器人失衡并请求终止 |
| `impulse_boundary_env.yaml` | `baseline/humanoid21/blueprints/` | 参数化环境蓝图，集成以上插件 |

### 数据维度

| 数据 | 维度 | 说明 |
|------|------|------|
| `core_state` | 55 | root_pos(3) + root_rot(4) + root_vel_local(3) + root_angular_vel_local(3) + joint_pos_norm(21) + joint_vel_norm(21) |
| `observation` | 96 | 本体感知(42) + 全局状态(13) + 足底力(2) + 对手观测(39) |
| `impulse_direction` | 3 | 单位向量 [x, y, z] |

---

## Step 1: ImpulsePerturbationPlugin — 扰动状态生成

### 原理

`ImpulsePerturbationPlugin` 在 `on_pre_episode` 钩子中执行：

1. 读取真实环境的当前 core_state
2. 创建内部 `Humanoid21Simulator`，复制该状态
3. 加载策略（从 `policy_blueprint_path`）
4. 在 `duration_action_steps` 个 action step 内，每个物理子步施加外力 + 策略控制
5. 将扰动后的状态写回主环境

关键设计：**策略在扰动期间运行**，使扰动状态物理合理（策略会尝试抵抗推力）。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_robot` | str | `"robot_a"` | 被扰动的机器人 |
| `policy_blueprint_path` | str | `None` | 策略蓝图路径（用于内部 sim） |
| `impulse_body` | str | `"torso"` | 施力部位 |
| `force_magnitude` | float 或 (min, max) | `(100, 500)` | 力大小（N），标量=固定值，元组=随机范围 |
| `duration_action_steps` | int 或 (min, max) | `(1, 8)` | 持续 action step 数 |
| `direction_mode` | str | `"random_horizontal"` | 方向模式：`random_horizontal` 或 `fixed` |
| `fixed_direction` | list | `None` | 固定方向 [x, y, z]（`direction_mode='fixed'` 时使用） |
| `phy_steps_per_action` | int | `25` | 每 action step 的物理子步数 |
| `random_seed` | int | `None` | 随机种子 |

### test_impulse_plugin.py — 插件检验脚本

**文件**: `baseline/framework/test_impulse_plugin.py`

验证 `ImpulsePerturbationPlugin` 生成的扰动状态是否物理合理。

**检验项**:
1. 扰动后 `root_vel_local` 在推力方向上有非零分量
2. 扰动后 `root_pos[2]` 在合理范围内（不穿透地面、不飞天）
3. `joint_pos_norm` 和 `joint_vel_norm` 无 NaN/Inf
4. 不同 force 值产生的状态有可测量差异
5. 不同 seed 产生的状态不同（随机性生效）

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 baseline/framework/test_impulse_plugin.py \
    --policy-export baseline/runs/train_basic_balance_v2_standup_ppo_20260801_003425/policy \
    --force 200 --duration 4 --direction 1,0,0 --body torso
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy-export` | str | `None` | 策略导出目录（含 `policy_blueprint.yaml`） |
| `--force` | float | `200` | 力大小（N） |
| `--duration` | int | `4` | 持续 action step 数 |
| `--direction` | str | `"1,0,0"` | 力方向，逗号分隔的 x,y,z |
| `--body` | str | `"torso"` | 施力部位 |
| `--seed` | int | `42` | 随机种子 |

**输出示例**:

```
=== ImpulsePerturbationPlugin 检验 ===
policy: .../policy_blueprint.yaml
force=200N  duration=4 action steps  direction=[1.0, 0.0, 0.0]  body=torso  seed=42

--- 扰动前后状态对比 ---
root_pos before: [0.  0.  1.282]
root_pos after:  [0.02  0.01  1.275]
root_vel_local before: [0. 0. 0.]
root_vel_local after:  [0.15 -0.03 -0.01]

[检验1] 推力方向速度分量: 0.1500 m/s
  PASS: 推力方向上有非零速度分量
[检验2] root_pos[2] = 1.2750 m
  PASS: 高度在合理范围内
[检验3] PASS: 所有状态字段无 NaN/Inf
[检验4] force=200 vs force=20 速度差: 0.1200
  PASS: 不同 force 产生可测量差异
[检验5] seed=42 vs seed=1042 速度差: 0.0500
  PASS: 不同 seed 产生不同状态
```

---

## Step 2: 边界测绘

### probe_impulse_boundary.py — 网格扫描存活率

**文件**: `baseline/framework/probe_impulse_boundary.py`

在 (force, duration) 网格上扫描，每个格子跑 N 个 episode，统计存活率，找到平衡恢复的边界区域。

**工作流**:
1. 加载策略蓝图和参数化环境蓝图
2. 对每个 (force, duration) 组合，materialize 一个 `EnvBlueprint`
3. 用 `ParallelRollouter` 并行跑所有 episode
4. 从 `Episode.termination_proposals` 判断是否存活（含 `"imbalance"` = 摔倒）
5. 输出 CSV + 汇总表

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 baseline/framework/probe_impulse_boundary.py \
    --policy-export baseline/runs/train_basic_balance_v2_standup_ppo_20260801_003425/policy \
    --force-grid 10,20,30,50,70,100,150,200 \
    --duration-grid 1,2,3,4,6,8,12 \
    --episodes-per-cell 20 \
    --workers 8 \
    --output baseline/runs/recovery_iter/gen0_boundary.csv
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy-export` | str | (必填) | 策略导出目录 |
| `--blueprint` | str | `impulse_boundary_env.yaml` | 环境蓝图路径 |
| `--force-grid` | str | `"50,100,...,700"` | 逗号分隔的力值列表（N） |
| `--duration-grid` | str | `"1,2,4,8,12,20"` | 逗号分隔的持续时间列表（action steps） |
| `--episodes-per-cell` | int | `20` | 每个网格格子的 episode 数 |
| `--workers` | int | `8` | 并行 worker 数 |
| `--seed` | int | `42` | 基础随机种子 |
| `--max-steps` | int | `600` | 每 episode 最大 action step 数 |
| `--output` | str | `None` | 输出 CSV 路径（不指定则只打印） |
| `--agent-id` | str | `"robot_a"` | 目标机器人 |

**输出**:

- **CSV 文件**: 列 `force, duration, survived, fell, total, surv_rate, mean_len`
- **终端汇总**: 每格存活率表 + 单调性检查 + 边界区域识别

**输出示例**:

```
  force  dur  survived  fell  total  surv_rate  mean_len
------------------------------------------------------------
     10    1        18     2     20      0.900     580.5
     10    4        15     5     20      0.750     520.3
     ...
    200    8         0    20     20      0.000      18.2

=== Summary Checks ===
  Boundary cells (surv_rate in [0.2, 0.8]): 12
    force=50N  duration=4  surv=0.500
    ...
```

### plot_impulse_boundary.py — 可视化热力图

**文件**: `baseline/framework/plot_impulse_boundary.py`

将 `probe_impulse_boundary.py` 的 CSV 输出可视化为热力图。

**用法**:

```bash
python3 baseline/framework/plot_impulse_boundary.py \
    --input baseline/runs/recovery_iter/gen0_boundary.csv \
    --output baseline/runs/recovery_iter/gen0_boundary_heatmap.png
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | (必填) | CSV 文件路径 |
| `--output` | str | `None` | 输出 PNG 路径（不指定则存到 CSV 同目录） |

**输出**: 两张热力图 — 存活率 + 平均 episode 长度。

---

## Step 3: 状态池生成

### generate_state_bank.py — 批量生成标注数据

**文件**: `baseline/framework/generate_state_bank.py`

生成带标签的状态池，用于后续训练平衡恢复分类器。

**工作流**:
1. 加载策略蓝图和环境蓝图（含 `ImpulsePerturbationPlugin` + `StateCapturePlugin` + `StateCaptureObserver`）
2. 对每个 (force, duration) 组合，materialize `EnvBlueprint`
3. 用 `ParallelRollouter` 并行跑所有 episode
4. 从 `Episode.observer_outputs["state_capture"]` 提取扰动后的 core_state + observation
5. 从 `Episode.termination_proposals` 判断存活/摔倒标签
6. 保存为 `.npz` 文件

**数据捕获机制**:

```
on_pre_episode:
  1. ImpulsePerturbationPlugin → 生成扰动状态，写回 sim
  2. StateCapturePlugin.on_pre_episode → 重置捕获标志

on_pre_action_step (第一帧):
  3. StateCapturePlugin → 读取 core_state + observation，写入 ctx.metrics
     (此时物理步尚未执行，状态 = 扰动后初始状态)

on_post_action_step (第一帧):
  4. StateCaptureObserver → 从 ctx.metrics 读取数据，存入 get_output()
  5. EpisodeRecorder → 记录到 Episode.observer_outputs["state_capture"]
```

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 baseline/framework/generate_state_bank.py \
    --policy-export baseline/runs/train_basic_balance_v2_standup_ppo_20260801_003425/policy \
    --force-grid 10,20,30,50,70,100,150 \
    --duration-grid 1,2,3,4,6,8 \
    --episodes-per-cell 20 \
    --workers 8 \
    --output baseline/runs/recovery_iter/gen0_state_bank.npz
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy-export` | str | (必填) | 策略导出目录 |
| `--blueprint` | str | `impulse_boundary_env.yaml` | 环境蓝图路径 |
| `--force-grid` | str | `"10,20,30,50,70,100,150"` | 逗号分隔的力值列表（N） |
| `--duration-grid` | str | `"1,2,3,4,6,8"` | 逗号分隔的持续时间列表 |
| `--episodes-per-cell` | int | `20` | 每个网格格子的 episode 数 |
| `--workers` | int | `8` | 并行 worker 数 |
| `--seed` | int | `42` | 基础随机种子 |
| `--max-steps` | int | `600` | 每 episode 最大 action step 数 |
| `--tolerance` | int | `6` | 失衡容忍步数（连续 N 步非脚部触地 = 摔倒） |
| `--output` | str | (必填) | 输出 `.npz` 文件路径 |
| `--agent-id` | str | `"robot_a"` | 目标机器人 |

**输出 (.npz 文件内容)**:

| 数组名 | 形状 | 类型 | 说明 |
|--------|------|------|------|
| `states` | (N, 55) | float32 | 扰动后 core_state |
| `observations` | (N, 96) | float32 | 扰动后 observation |
| `forces` | (N,) | float32 | 冲击力大小（N） |
| `durations` | (N,) | int32 | 冲击持续时间（action steps） |
| `directions` | (N, 3) | float32 | 冲击方向单位向量 |
| `labels` | (N,) | float32 | 1.0=存活, 0.0=摔倒 |
| `ep_lengths` | (N,) | int32 | episode 长度（action steps） |
| `core_state_fields` | (6,) | str | core_state 字段名 |
| `core_state_dims` | (6,) | int | core_state 各字段维度 |

**输出示例**:

```
=== State Bank Generation ===
Total episodes: 420
Rollout time: 77.6s (0.18s/episode)

Total states: 420
Survived: 147  Fell: 273  Rate: 0.350
State dim: 55  Obs dim: 96

  force  dur  surv  fell  total   rate  mean_len
--------------------------------------------------
     10    1     7     3     10  0.700     539.2
     ...
    150    8     0    10     10  0.000      14.6

State bank saved to gen0_state_bank.npz
File size: 0.2 MB
Verification: loaded shapes OK (states=(420, 55), obs=(420, 96))
```

---

## 蓝图文件

### impulse_boundary_env.yaml

**文件**: `baseline/humanoid21/blueprints/impulse_boundary_env.yaml`

参数化环境蓝图，集成所有需要的插件。

**参数 (parameters)**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_distance` | `2.0` | 机器人初始间距（m） |
| `max_steps` | `600` | 最大 action step 数 |
| `agent_id` | `"robot_a"` | 目标机器人 |
| `tolerance` | `6` | 失衡容忍步数 |
| `policy_blueprint_path` | `null` | 策略蓝图路径 |
| `impulse_body` | `"torso"` | 施力部位 |
| `force_magnitude` | `[100, 500]` | 力大小范围 |
| `duration_action_steps` | `[1, 8]` | 持续时间范围 |
| `direction_mode` | `"random_horizontal"` | 方向模式 |
| `fixed_direction` | `null` | 固定方向 |

**插件 (plugins)**:

1. `ImbalanceTerminationPlugin` — 失衡检测终止
2. `ImpulsePerturbationPlugin` — 冲击扰动生成
3. `StateCapturePlugin` — 状态捕获（写入 ctx.metrics）

**观察器 (observer_plugins)**:

1. `state_capture` — `StateCaptureObserver`，暴露捕获的 core_state + observation
2. `cross_support` — `CrossSupportBalanceRewarder`
3. `posture` — `PostureRewarder`
4. `wall_contact` — `WallContactObserver`

---

## 完整工作流示例

```bash
# 设置环境变量
export CB_ROOT=/data1/mono/things/combatbench
export PYTHONPATH=$CB_ROOT
export POLICY=$CB_ROOT/baseline/runs/train_basic_balance_v2_standup_ppo_20260801_003425/policy

# Step 1: 验证插件
python3 $CB_ROOT/baseline/framework/test_impulse_plugin.py \
    --policy-export $POLICY \
    --force 200 --duration 4 --direction 1,0,0

# Step 2: 边界测绘
python3 $CB_ROOT/baseline/framework/probe_impulse_boundary.py \
    --policy-export $POLICY \
    --force-grid 10,20,30,50,70,100,150,200 \
    --duration-grid 1,2,3,4,6,8,12 \
    --episodes-per-cell 20 \
    --workers 8 \
    --output $CB_ROOT/baseline/runs/recovery_iter/gen0_boundary.csv

# Step 2b: 可视化
python3 $CB_ROOT/baseline/framework/plot_impulse_boundary.py \
    --input $CB_ROOT/baseline/runs/recovery_iter/gen0_boundary.csv \
    --output $CB_ROOT/baseline/runs/recovery_iter/gen0_boundary_heatmap.png

# Step 3: 状态池生成
python3 $CB_ROOT/baseline/framework/generate_state_bank.py \
    --policy-export $POLICY \
    --force-grid 10,20,30,50,70,100,150 \
    --duration-grid 1,2,3,4,6,8 \
    --episodes-per-cell 20 \
    --workers 8 \
    --output $CB_ROOT/baseline/runs/recovery_iter/gen0_state_bank.npz
```

### 输出文件

```
baseline/runs/recovery_iter/
├── gen0_boundary.csv              # 边界测绘数据
├── gen0_boundary_heatmap.png      # 存活率热力图
└── gen0_state_bank.npz            # 状态池（states + observations + labels）
```

### 后续步骤

- **Step 4**: `StateBankInitPlugin` — 从 `.npz` 状态池加载随机状态注入环境
- **Step 5**: 新蓝图 `balance_recover_v3_env.yaml` + 新实验 `exp_balance_recover_v3.py`
- **Step 6**: 迭代循环脚本 `recovery_iter_loop.py`
