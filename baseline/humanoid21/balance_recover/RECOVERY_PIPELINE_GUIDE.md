# 平衡恢复训练 — 资源与脚本说明

本文档说明 `balance_recover/` 目录下所有脚本、蓝图、实验文件和插件的功能与用法。

> **状态标记说明**
> - ✅ **当前流程在用**：在"探测边界 → 实时扰动训练"流程中直接使用
> - ⚠️ **当前流程不用**：属于旧状态池方案，当前流程不再使用，但保留作为参考
> - 🔧 **框架组件**：被其他脚本/蓝图依赖的底层组件

---

## 数据维度

| 数据 | 维度 | 说明 |
|------|------|------|
| `core_state` | 55 | root_pos(3) + root_rot(4) + root_vel_local(3) + root_angular_vel_local(3) + joint_pos_norm(21) + joint_vel_norm(21) |
| `observation` | 96 | 本体感知(42) + 全局状态(13) + 足底力(2) + 对手观测(39) |
| `impulse_direction` | 3 | 单位向量 [x, y, z]（由相对角度 + 机器人朝向计算得到） |
| `impulse_direction_angle` | 1 | 相对机器人朝向的角度（度），0°=正面, 90°=右侧, 180°=背面, 270°=左侧 |

---

## 插件

### ImpulsePerturbationPlugin 🔧

**文件**: `envs/humanoid21/disturbance_plugins.py`

通过内部 sim + 参考策略生成物理合理的扰动初始状态。在 `on_pre_episode` 钩子中执行：

1. 读取真实环境的当前 core_state
2. 创建内部 `Humanoid21Simulator`，复制该状态
3. 加载策略（从 `policy_blueprint_path`）
4. 在 `duration_action_steps` 个 action step 内，每个物理子步施加外力 + 策略控制
5. 将扰动后的状态写回主环境

关键设计：**策略在扰动期间运行**，使扰动状态物理合理（策略会尝试抵抗推力）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_robot` | str | `"robot_a"` | 被扰动的机器人 |
| `policy_blueprint_path` | str | `None` | 策略蓝图路径（用于内部 sim） |
| `impulse_body` | str | `"torso"` | 施力部位（固定为 torso） |
| `force_magnitude` | float | `200` | 力大小（N），固定值或档位：轻=50, 中=100, 大=300 |
| `duration_action_steps` | int 或 (min, max) | `(1, 8)` | 持续 action step 数（可变量） |
| `direction_mode` | str | `"relative_angle"` | 方向模式：`relative_angle`（相对机器人朝向的角度）或 `fixed` |
| `direction_angle` | float 或 (min, max) | `(0, 360)` | 相对机器人朝向的角度（度）：0°=正面, 90°=右侧, 180°=背面, 270°=左侧。对两个机器人使用相同定义 |
| `phy_steps_per_action` | int | `25` | 每 action step 的物理子步数 |
| `random_seed` | int | `None` | 随机种子 |

### ImbalanceTerminationPlugin 🔧

**文件**: `baseline/humanoid21/plugins/imbalance_termination.py`

检测机器人失衡并请求终止。连续 N 步非脚部触地判定为摔倒。

### StateCapturePlugin ⚠️

**文件**: `envs/humanoid21/disturbance_plugins.py`

在第一个 action step 前捕获 core_state + observation，写入 `ctx.metrics`。配合 `StateCaptureObserver` 使用，用于状态池生成流程。当前流程不使用。

### StateCaptureObserver ⚠️

**文件**: `envs/humanoid21/disturbance_plugins.py`

通过 `observer_outputs` 暴露 `StateCapturePlugin` 捕获的数据。当前流程不使用。

### StateBankInitPlugin ⚠️

**文件**: `envs/humanoid21/disturbance_plugins.py`

在 `on_pre_episode` 中从 `.npz` 状态池采样一个扰动状态，通过 `set_core_state` 注入到 sim，替代 `ImpulsePerturbationPlugin` 的实时扰动生成。当前流程不使用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `state_bank_path` | str | (必填) | `.npz` 状态池文件路径 |
| `target_robot` | str | `"robot_a"` | 目标机器人 |
| `seed` | int | `42` | 随机采样种子 |

---

## 脚本

### test_impulse_plugin.py ✅

**文件**: `balance_recover/test_impulse_plugin.py`

验证 `ImpulsePerturbationPlugin` 生成的扰动状态是否物理合理。

**检验项**:
1. 扰动后 `root_vel_local` 在推力方向上有非零分量
2. 扰动后 `root_pos[2]` 在合理范围内（不穿透地面、不飞天）
3. `joint_pos_norm` 和 `joint_vel_norm` 无 NaN/Inf
4. 不同 force 值产生的状态有可测量差异
5. 不同 seed 产生的状态不同（随机性生效）

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/test_impulse_plugin.py \
    --policy-export baseline/runs/.../policy \
    --force 200 --duration 4 --direction-angle 90
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy-export` | str | `None` | 策略导出目录（含 `policy_blueprint.yaml`） |
| `--force` | float | `200` | 力大小（N），可选档位：50/100/300 |
| `--duration` | int | `4` | 持续 action step 数 |
| `--direction-angle` | float | `0` | 力方向角度（度），相对机器人朝向：0°=正面, 90°=右侧, 180°=背面, 270°=左侧 |
| `--seed` | int | `42` | 随机种子 |

### probe_impulse_boundary.py ✅

**文件**: `balance_recover/probe_impulse_boundary.py`

固定力档位，在 (direction, duration) 网格上做粗粒度全扫描，每个格子跑 N 个 episode，统计存活率。扫描结果用于拟合存活率分布，训练时从分布中采样扰动参数（侧重边界区域）。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/probe_impulse_boundary.py \
    --policy-export baseline/runs/.../policy \
    --force 100 \
    --direction-grid 0,45,90,135,180,225,270,315 \
    --duration-grid 1,2,3,4,6,8,12 \
    --episodes-per-cell 20 \
    --workers 8 \
    --output baseline/runs/recovery_iter/gen0_boundary.csv
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy-export` | str | (必填) | 策略导出目录 |
| `--blueprint` | str | `impulse_boundary_env.yaml` | 环境蓝图路径 |
| `--force` | float | `100` | 固定力大小（N），可选档位：50/100/300 |
| `--direction-grid` | str | `"0,45,90,135,180,225,270,315"` | 逗号分隔的方向角度列表（度），相对机器人朝向 |
| `--duration-grid` | str | `"1,2,4,8,12,20"` | 逗号分隔的持续时间列表（action steps） |
| `--episodes-per-cell` | int | `20` | 每个网格格子的 episode 数 |
| `--workers` | int | `8` | 并行 worker 数 |
| `--seed` | int | `42` | 基础随机种子 |
| `--max-steps` | int | `600` | 每 episode 最大 action step 数 |
| `--output` | str | `None` | 输出 CSV 路径（不指定则只打印） |
| `--agent-id` | str | `"robot_a"` | 目标机器人 |

**输出**: CSV 文件（列 `direction, duration, survived, fell, total, surv_rate, mean_len`）+ 终端汇总（每格存活率表 + 边界区域识别）

### plot_impulse_boundary.py ✅

**文件**: `balance_recover/plot_impulse_boundary.py`

将 `probe_impulse_boundary.py` 的 CSV 输出可视化为热力图。

**用法**:

```bash
python3 balance_recover/plot_impulse_boundary.py \
    --input baseline/runs/recovery_iter/gen0_boundary.csv \
    --output baseline/runs/recovery_iter/gen0_boundary_heatmap.png
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | (必填) | CSV 文件路径 |
| `--output` | str | `None` | 输出 PNG 路径（不指定则存到 CSV 同目录） |

**输出**: 两张热力图 — 存活率 + 平均 episode 长度。

### generate_state_bank.py ⚠️

**文件**: `balance_recover/generate_state_bank.py`

生成带标签的状态池（`.npz`），用于旧状态池训练方案。当前流程不使用，保留作为参考。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/generate_state_bank.py \
    --policy-export baseline/runs/.../policy \
    --force-grid 10,20,30,50,70,100,150 \
    --duration-grid 1,2,3,4,6,8 \
    --episodes-per-cell 20 \
    --workers 8 \
    --output baseline/runs/recovery_iter/gen0_state_bank.npz
```

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
| `--tolerance` | int | `6` | 失衡容忍步数 |
| `--output` | str | (必填) | 输出 `.npz` 文件路径 |
| `--agent-id` | str | `"robot_a"` | 目标机器人 |

**输出 (.npz 内容)**: `states`(N,55), `observations`(N,96), `forces`(N,), `durations`(N,), `directions`(N,3), `labels`(N,), `ep_lengths`(N,), `core_state_fields`(6,), `core_state_dims`(6,)

### verify_state_bank.py ⚠️

**文件**: `balance_recover/verify_state_bank.py`

验证 `StateBankInitPlugin` 注入的状态能产生与状态池 label 一致的结果。当前流程不使用，保留作为参考。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/verify_state_bank.py \
    --policy-export baseline/runs/.../policy \
    --state-bank /tmp/state_bank_verify.npz \
    --workers 8
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy-export` | str | (必填) | 策略导出目录 |
| `--state-bank` | str | (必填) | `.npz` 状态池路径 |
| `--blueprint` | str | `impulse_boundary_env.yaml` | 基础蓝图路径 |
| `--workers` | int | `8` | 并行 worker 数 |
| `--max-steps` | int | `600` | 最大 action step 数 |
| `--tolerance` | int | `6` | 失衡容忍步数 |
| `--agent-id` | str | `"robot_a"` | 目标机器人 |
| `--seed` | int | `42` | 基础随机种子 |

### recovery_iter_loop.py ⚠️

**文件**: `balance_recover/recovery_iter_loop.py`

自动化迭代训练循环。**当前实现基于状态池方案**（generate → filter → train with state bank），与当前"探测边界 → 实时扰动训练"流程不兼容，需要重写后才能使用。

现有实现的核心机制：
1. 生成状态池（调用 `generate_state_bank.py`）
2. 分析边界（计算 per-cell 存活率，找到 `boundary_force`）
3. 过滤边界状态（只保留存活率在 [20%, 80%] 的 cell）
4. 训练 PPO（用边界状态池，warm-start 从当前策略）
5. 评估前移（`boundary_force` 是否右移）
6. 自适应 grid（下一轮根据边界位置调整 force grid）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--base-policy` | str | (必填) | 初始策略目录 |
| `--output-dir` | str | `baseline/runs/recovery_iter` | 输出根目录 |
| `--max-iters` | int | `5` | 最大迭代轮数 |
| `--train-updates` | int | `5000` | 每轮 PPO 训练 update 数 |
| `--force-grid` | str | `10,20,30,50,70,100,150,200` | 初始力值网格 (N) |
| `--duration-grid` | str | `1,2,3,4,6,8` | 持续时间网格 |
| `--episodes-per-cell` | int | `20` | 每格 episode 数 |
| `--gen-workers` | int | `8` | 状态池生成并行数 |
| `--rollout-workers` | int | `8` | 训练 rollout 并行数 |
| `--max-steps` | int | `600` | 每 episode 最大步数 |
| `--tolerance` | int | `6` | 失衡容忍步数 |
| `--boundary-range` | str | `0.2,0.8` | 边界 cell 的存活率范围 |
| `--no-improve-patience` | int | `2` | 连续无改善轮数停止 |
| `--target-boundary-force` | float | `300` | 目标 boundary force 停止 |
| `--seed` | int | `42` | 基础随机种子 |
| `--smoke` | flag | off | Smoke test 模式 |
| `--no-adapt-grid` | flag | off | 禁用 grid 自适应 |

---

## 蓝图文件

### impulse_boundary_env.yaml ✅

**文件**: `balance_recover/impulse_boundary_env.yaml`

参数化环境蓝图，集成 `ImpulsePerturbationPlugin` + `StateCapturePlugin`，用于边界探测和插件验证。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_distance` | `2.0` | 机器人初始间距（m） |
| `max_steps` | `600` | 最大 action step 数 |
| `agent_id` | `"robot_a"` | 目标机器人 |
| `tolerance` | `6` | 失衡容忍步数 |
| `policy_blueprint_path` | `null` | 策略蓝图路径 |
| `impulse_body` | `"torso"` | 施力部位（固定） |
| `force_magnitude` | `100` | 力大小（N），固定档位：50/100/300 |
| `duration_action_steps` | `[1, 8]` | 持续时间范围（可变量） |
| `direction_mode` | `"relative_angle"` | 方向模式：相对机器人朝向的角度 |
| `direction_angle` | `[0, 360]` | 方向角度范围（度），0°=正面, 90°=右侧, 180°=背面, 270°=左侧 |

**插件**: `ImbalanceTerminationPlugin`, `ImpulsePerturbationPlugin`, `StateCapturePlugin`

**观察器**: `state_capture`, `cross_support`, `posture`, `wall_contact`

### balance_recover_v3_env.yaml ⚠️

**文件**: `balance_recover/balance_recover_v3_env.yaml`

训练用环境蓝图，使用 `StateBankInitPlugin` 从状态池注入扰动状态。当前流程不使用（当前流程用 `ImpulsePerturbationPlugin` 实时扰动）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_bank_path` | `null` | `.npz` 状态池路径（必填） |
| `state_bank_seed` | `42` | 状态采样种子 |
| `max_steps` | `600` | 最大 action step 数 |
| `tolerance` | `6` | 失衡容忍步数 |
| `agent_id` | `"robot_a"` | 目标机器人 |

**插件**: `ImbalanceTerminationPlugin`, `StateBankInitPlugin`

**观察器**: 无（纯生存奖励，最大化 rollout 速度）

### balance_recover_v4_env.yaml ✅

**文件**: `balance_recover/balance_recover_v4_env.yaml`

训练用环境蓝图，使用 `ImpulsePerturbationPlugin` 实时生成扰动。适用于当前"探测边界 → 实时扰动训练"流程。

### basic_balance_v2_phi_dual_impulse_env.yaml ✅

**文件**: `balance_recover/basic_balance_v2_phi_dual_impulse_env.yaml`

双代理环境蓝图，包含两个 `ImpulsePerturbationPlugin` 实例（每个机器人一个），用于双代理冲量扰动实验。

---

## 实验文件

### exp_balance_recover_v3.py ⚠️

**文件**: `balance_recover/exp_balance_recover_v3.py`

基于状态池的 PPO 训练实验，继承 `CombatExperimentBase`。每个 episode 从 `.npz` 采样扰动状态注入。当前流程不使用。

**环境变量**:

| 环境变量 | 必填 | 说明 |
|----------|------|------|
| `STATE_BANK_PATH` | 是 | `.npz` 状态池路径 |
| `BASE_POLICY_PATH` | 否 | base policy 目录（含 `model.pt`） |

### exp_balance_recover_v4.py ✅

**文件**: `balance_recover/exp_balance_recover_v4.py`

基于 `ImpulsePerturbationPlugin` 实时扰动的 PPO 训练实验。通过环境变量配置冲量参数和 warm-start 策略路径。适用于当前流程。

**环境变量**:

| 环境变量 | 说明 |
|----------|------|
| `IMPULSE_FORCE` | 冲量力大小（N），固定档位：50/100/300 |
| `IMPULSE_DURATION` | 冲量持续时间或范围（action steps） |
| `IMPULSE_DIRECTION_ANGLE` | 冲量方向角度或范围（度），相对机器人朝向 |
| `POLICY_BLUEPRINT_PATH` | 参考策略蓝图路径（用于内部 sim） |
| `BASE_POLICY_PATH` | Warm-start 策略路径 |

---

## MuJoCo 跨进程非确定性 ⚠️

> 以下内容来自旧状态池验证流程的观察，当前流程不涉及状态池注入，但 MuJoCo 跨进程非确定性仍然存在。

**关键事实**:
- 同进程跑两次：100% 一致
- 跨进程（workers=1 vs workers=8）：~62% 一致
- Mismatch 集中在不稳定边界 case（episode length 200-500 步）

详细分析见 `MUJOCO_CROSS_PROCESS_NONDETERMINISM.md`。

**对训练的影响**：可忽略。单个样本的随机翻转不影响整体训练统计。
