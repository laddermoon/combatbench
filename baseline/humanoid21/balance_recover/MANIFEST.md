# balance_recover/ — 文件清单 (Manifest)

本目录包含平衡恢复训练相关的插件、脚本、蓝图、实验和数据文件。
每个条目记录文件用途、创建原因、用法（如果是脚本）和当前状态。

> **状态标记**
> - ✅ **在用**：当前流程正在使用
> - ⚠️ **废弃**：旧方案遗留，当前流程不再使用，保留作参考
> - 🔧 **组件**：被其他文件依赖的底层组件

---

## 插件

### relative_impulse_plugin.py ✅

`RelativeImpulsePlugin` — 纯执行层插件，从 `episode_options["impulse_params"]` 接收预采样扰动参数，在 `on_pre_episode` 中用内部 `EnvRuntime` + `ConstantForcePlugin` 施力，将扰动后状态写回真实环境。非目标机器人在内部 sim 中冻结。

**创建原因**：将采样逻辑与执行逻辑解耦，采样由 `ImpulseSampler`（在 `sample_distribution.py` 中）完成，实验类 `exp_weighted_impulse.py` 负责调用采样器并通过 `episode_options` 传参。

### freeze_robot_plugin.py ✅

`FreezeRobotPlugin` — 在前 N 个 action step 内冻结指定机器人（每个物理子步 reset 到初始 state）。

**创建原因**：验证脚本 `verify_consistency.py` 中 Path A 需要冻结 robot_b 以匹配 `RelativeImpulsePlugin` 内部 sim 的行为。

### ConstantForcePlugin 🔧

**文件**: `envs/humanoid21/disturbance_plugins.py`

在 `on_pre_phy_step` 中对指定机器人施加恒定外力，持续 N 个 action step。方向在 `on_pre_action_step` 首次调用时根据机器人朝向计算。被 `RelativeImpulsePlugin` 内部 sim 使用。

### ImpulsePerturbationPlugin ⚠️

**文件**: `envs/humanoid21/disturbance_plugins.py`

旧版扰动插件，内置采样逻辑（force/direction/duration 均在插件内采样）。已被 `RelativeImpulsePlugin` + `ImpulseSampler` 取代。

### DualImbalanceTerminationPlugin 🔧

**文件**: `baseline/humanoid21/plugins/imbalance_termination.py`

检测机器人失衡（连续 N 步非脚部触地）并请求终止。被多个环境蓝图使用。

### StateCapturePlugin / StateCaptureObserver ⚠️

**文件**: `envs/humanoid21/disturbance_plugins.py`

捕获 core_state + observation 到 `ctx.metrics`，用于旧状态池方案。

### StateBankInitPlugin ⚠️

**文件**: `envs/humanoid21/disturbance_plugins.py`

从 `.npz` 状态池采样扰动状态注入 sim，替代实时扰动生成。旧方案。

---

## 脚本

### verify_consistency.py ✅

验证 `ConstantForcePlugin` 直接施力 vs `RelativeImpulsePlugin` 内部 sim 施力的一致性。用 `RoundRunner` 跑两条路径，生成对比视频和 per-step 记录。

**创建原因**：确保 `RelativeImpulsePlugin` 的内部 sim 准确复现了 `ConstantForcePlugin` 的施力效果。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 \
    baseline/humanoid21/balance_recover/verify_consistency.py \
    --policy-blueprint-path baseline/runs/.../policy_blueprint.yaml \
    --seed 42 --force 200 --direction 90 --duration 4 \
    --output-dir /tmp/verify_consistency
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy-blueprint-path` | `None` | 策略蓝图路径，省略则用 zero action |
| `--seed` | `42` | 随机种子 |
| `--force` | `200` | 力大小 (N) |
| `--direction` | `90` | 方向角度 (度) |
| `--duration` | `4` | 持续 action step 数 |
| `--output-dir` | `/tmp/verify_consistency` | 输出目录（视频 + 记录） |

**输出**: `path_a.mp4`, `path_b.mp4` + `path_a_rec/`, `path_b_rec/` (per-step PNG + JSON)

### test_impulse_plugin.py ✅

验证 `ImpulsePerturbationPlugin` 生成的扰动状态是否物理合理（速度方向、高度范围、无 NaN）。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/test_impulse_plugin.py \
    --policy-export baseline/runs/.../policy \
    --force 200 --duration 4 --direction-angle 90
```

### probe_impulse_boundary.py ✅

在 (direction, duration) 网格上扫描存活率，用于拟合存活率分布。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/probe_impulse_boundary.py \
    --policy-export baseline/runs/.../policy \
    --force 100 \
    --direction-grid 0,45,90,135,180,225,270,315 \
    --duration-grid 1,2,3,4,6,8,12 \
    --episodes-per-cell 20 --workers 8 \
    --output baseline/runs/recovery_iter/gen0_boundary.csv
```

### probe_boundary.py ✅

`probe_impulse_boundary.py` 的变体/早期版本，功能类似。

### plot_impulse_boundary.py ✅

将 `probe_impulse_boundary.py` 的 CSV 输出可视化为热力图（存活率 + 平均 episode 长度）。

**用法**:

```bash
python3 balance_recover/plot_impulse_boundary.py \
    --input baseline/runs/recovery_iter/gen0_boundary.csv \
    --output baseline/runs/recovery_iter/gen0_boundary_heatmap.png
```

### sample_distribution.py ✅

权重分布加载与采样逻辑。包含 `ImpulseSampler` 类：从 `sample_weights.npz` 加载方向/力/持续时间权重分布，按权重采样扰动参数。

**创建原因**：将采样逻辑从 `RelativeImpulsePlugin` 和 `exp_weighted_impulse.py` 中抽出，封装为可复用的采样器。

**用法（作为模块）**:

```python
from baseline.humanoid21.balance_recover.sample_distribution import ImpulseSampler
sampler = ImpulseSampler("path/to/sample_weights.npz", direction_jitter=5.0)
params = sampler.sample(rng)  # {"direction_angle", "force", "duration_action_steps", "body"}
```

也可作为独立脚本运行，生成采样分布可视化和 `sample_weights.npz`。

### verify_direction_video.py ✅

对 0°（正面）、90°（右侧）、180°（背面）、270°（左侧）各生成一个视频，用大力 + 长 duration 让用户目视确认冲量方向是否正确。支持 `--agent-id` 指定目标机器人。

**创建原因**：验证 `RelativeImpulsePlugin` 重构后方向计算是否正确。

**用法**:

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 \
    baseline/humanoid21/balance_recover/verify_direction_video.py \
    --policy-blueprint-path baseline/runs/.../policy_blueprint.yaml \
    --force 300 --duration 8 \
    --agent-id robot_a \
    --output-dir /data1/dev/verify_direction
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy-blueprint-path` | (必填) | 策略蓝图路径 |
| `--blueprint` | `weighted_impulse_env.yaml` | 环境蓝图路径 |
| `--force` | `300` | 力大小 (N) |
| `--duration` | `8` | 持续 action step 数 |
| `--output-dir` | `/data1/dev/verify_direction` | 输出目录 |
| `--max-steps` | `400` | 每 episode 最大步数 |
| `--agent-id` | `robot_a` | 目标机器人（`robot_a` 或 `robot_b`） |

**输出**: 4 个 MP4 文件（`impulse_<agent>_<label>_<angle>deg.mp4`）

**已生成视频**: `verify_videos/` 下有 robot_a 和 robot_b 各 4 个方向共 8 个视频。

### verify_monotonicity.py ✅

验证存活率随力大小单调递减（力越大越容易摔倒）。

### generate_state_bank.py ⚠️

生成带标签的状态池 `.npz`，用于旧状态池训练方案。

### verify_state_bank.py ⚠️

验证 `StateBankInitPlugin` 注入的状态能产生与状态池 label 一致的结果。

### recovery_iter_loop.py ⚠️

自动化迭代训练循环（状态池方案）。与当前流程不兼容，需要重写。

---

## 蓝图文件 (YAML)

### weighted_impulse_env.yaml ✅

参数化环境蓝图，使用 `RelativeImpulsePlugin`（双机器人），扰动参数通过 `episode_options` 注入。被 `exp_weighted_impulse.py` 使用。

### impulse_boundary_env.yaml ✅

参数化环境蓝图，使用 `ImpulsePerturbationPlugin` + `StateCapturePlugin`，用于边界探测。

### balance_recover_v4_env.yaml ✅

训练用环境蓝图，使用 `ImpulsePerturbationPlugin` 实时生成扰动。

### basic_balance_v2_phi_dual_impulse_env.yaml ✅

双代理环境蓝图，包含两个 `ImpulsePerturbationPlugin` 实例。

### relative_impulse_env.yaml ⚠️

`RelativeImpulsePlugin` 的早期环境蓝图，已被 `weighted_impulse_env.yaml` 取代。

### balance_recover_v3_env.yaml ⚠️

训练用环境蓝图，使用 `StateBankInitPlugin` 从状态池注入扰动状态。旧方案。

---

## 实验文件

### exp_balance_recover_v4.py ✅

基于 `ImpulsePerturbationPlugin` 实时扰动的 PPO 训练实验。通过环境变量配置冲量参数。

### exp_balance_recover_v3.py ⚠️

基于状态池的 PPO 训练实验。旧方案。

> **注**: `exp_weighted_impulse.py` 位于 `baseline/experiments_v2/`，不在本目录下。它使用 `ImpulseSampler` + `weighted_impulse_env.yaml` + `RelativeImpulsePlugin` 构建训练 job。

---

## 数据文件

| 文件 | 说明 |
|------|------|
| `sample_weights.npz` | 采样权重分布（方向×力×持续时间的权重矩阵） |
| `sample_distribution.json` | 采样分布的完整 JSON 快照 |
| `samples.csv` | 采样参数列表 |
| `boundary_fixaw_s42.csv` / `.json` | 固定策略的边界探测结果 |
| `boundary_gen0.csv` / `.json` | 第 0 代边界探测结果 |
| `boundary_test.csv` / `.json` | 边界探测测试数据 |
| `monotonicity_50N.csv` / `_single.csv` / `monotonicity_check.csv` | 单调性验证数据 |
| `heatmap_*.png` | 各种热力图（存活率、权重分布） |
| `gen0/` | 第 0 代探测数据子目录 |
| `verify_videos/` | 方向验证视频（robot_a + robot_b 各 4 个方向，共 8 个 MP4） |

---

## 其他文档

| 文件 | 说明 |
|------|------|
| `RECOVERY_ITERATION_PLAN.md` | 迭代训练计划文档 |
| `MUJOCO_CROSS_PROCESS_NONDETERMINISM.md` | MuJoCo 跨进程非确定性分析（如存在） |

---

## 数据维度参考

| 数据 | 维度 | 说明 |
|------|------|------|
| `core_state` | 55 | root_pos(3) + root_rot(4) + root_vel_local(3) + root_angular_vel_local(3) + joint_pos_norm(21) + joint_vel_norm(21) |
| `observation` | 96 | 本体感知(42) + 全局状态(13) + 足底力(2) + 对手观测(39) |
| `impulse_direction_angle` | 1 | 相对机器人朝向的角度（度），0°=正面, 90°=右侧, 180°=背面, 270°=左侧 |
