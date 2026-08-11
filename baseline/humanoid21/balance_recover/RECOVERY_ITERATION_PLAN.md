# 平衡恢复迭代训练方案

## 目标

训练一个能力强的平衡恢复策略：机器人在尽可能大的扰动下恢复平衡，且恢复后保持双足交替支撑的稳定步态。

> **注意：** 本方案仅针对恢复策略的训练。分类器所需的状态边界数据可能更多样（不同受击前状态、不同扰动类型），需要用其它方法生成，暂时搁置。

---

## 背景：当前问题诊断

### 1. 初始态扰动不物理

`InitialStatePerturbationPlugin` 独立采样 21 个关节角 ±29°、躯干倾斜、速度，产生的是任意扭曲构型，不是真实可达状态。实测 scale 0.90：66.5% 的 episode 开局双脚离地 >5cm，最高 48.4cm，而 `root_pos[2]` 不变、垂直速度独立采样——动力学自相矛盾。

### 2. 恢复策略从初代的良好交替步态退化成永久扑腾

| 量 | 初代 (basic_balance_v2) 稳态 | 恢复策略 (recover_v2 lv11) 稳态 |
|---|---|---|
| `foot_height` 峰峰值 | 0.096 m | 0.377 m (4×) |
| `joint_vel` 均值 | 1.23 | 2.41 (2×) |
| 瞬态 vs 稳态 | 明显收敛 | **几乎相同，从未稳定** |

初代策略确实在双足交替（83.5% 的步数有脚抬起），恢复策略则整个 5 秒都在大幅扑腾。

### 3. 奖励阈值与实际工作点严重失配

`extract_rewards` 中 `joint_vel` 死区 = 0.1，而初代实测 = 1.23（偏低 12×）；`joint_deviation` 死区 = 0.1，实测 = 0.23（偏低 2.3×）。死区结构从未生效，这两项退化成恒定的"最小化运动"压力，持续对抗恢复动作和交替步态。

### 4. 难度是 5 维交互，每维单独都是惰性的

scale 0.90 下各维单独存活率 ≥ 0.99，五维合在一起 = 0.758。`f ∝ s^{4.4}`，6 个 level（scale ≤ 0.40）难度恒为 0。

---

## 方案：迭代式力扰动 + 实时训练

### 核心思路

没有主策略（fighting policy），用平衡保持策略 `basic_balance_v2` 作为初代 π₀。每代：

1. **边界探测**：固定 (direction, force) 组合，在 duration 轴上二分查找存活→摔倒的临界点，得到 48 条边界曲线
2. **拟合分布**：用边界数据拟合 (direction, force) → critical_duration 的映射曲面
3. **采样训练**：从拟合的分布中采样扰动参数（侧重边界区域），配置 `ImpulsePerturbationPlugin` 实时生成扰动初始状态进行 PPO 训练
4. 训练直到 eval 成功率超过阈值，或连续 N 轮不再上升
5. 用新策略 π_{N+1} 重复

### 扰动参数空间

扰动由**外部冲量**实现，施力部位固定为 **torso**。可变量只有两个：

| 维度 | 说明 | 取值 |
|---|---|---|
| **受力方向** | 相对机器人朝向的角度，0°=正面，90°=右侧，180°=背面，270°=左侧。对两个机器人使用相同定义（即都是"从正面来的力"而非绝对方位） | 0°~360° 连续或离散采样 |
| **受力持续时间** | 冲量施加的 action step 数 | 1~N 步 |

**力大小**不做连续变化，设为固定档位：

| 档位 | 力大小（N） | 用途 |
|---|---|---|
| 轻力 | 50 | 基线探测 |
| 中等 | 100 | 常规训练 |
| 大力 | 300 | 极限测试 |

> 力和持续时间只要有一个变化即可，它们最终效果类似（力×时间≈冲量）。固定力大小、变化持续时间可以简化参数空间。

### 能力边界探测与采样

#### 离散组合

- **方向**：16 个离散值（0°~360°，每 22.5° 一个）
- **力档位**：3 个（轻 50N / 中 100N / 大 300N）
- **组合总数**：16 × 3 = 48 个 (direction, force) 组合

#### duration 单调性假设

核心假设：对固定的 (direction, force)，`surv_rate(duration)` 单调递减——短时扰动一定能站住，超过某个临界 duration 后站不住，且不会再翻回来。

**需要先验证**：选 3-4 个代表性组合（如正面+中力、侧面+大力），跑全 duration 网格（1~20 步每步都跑），画 surv_rate vs duration 曲线，确认单调性。

可能的非单调情况（需排除）：
- 短时强推的惯性效应反而比中等时长更易倒
- 中等时长与策略响应周期共振
- 长时间推力反而让策略"靠上去"稳住

#### 单调性验证结果（2026-08-11）

**验证方法**：使用 `RelativeImpulsePlugin`（方向相对机器人朝向），固定 force=50N，扫描 duration=1~20，每个 cell 跑 1 个 episode（seed 固定），判断目标机器人（robot_a）的终止原因是否为 imbalance。策略使用 `fixaw_survonly_crossphi2_s42` 导出策略。

**验证脚本**：`verify_monotonicity.py`
**方向验证脚本**：`verify_direction_video.py`（生成 4 方向视频，已目视确认 robot_a 和 robot_b 方向均正确）

**结果**：4 个方向均为**完美阶跃函数**，无非单调点：

| 方向 | 存活区间 | 摔倒区间 | 临界 duration |
|---|---|---|---|
| 0°（向前） | dur 1-6 | dur 7+ | **7** |
| 90°（向右） | dur 1-8 | dur 9+ | **9** |
| 180°（向后） | dur 1-5 | dur 6+ | **6** |
| 270°（向左） | dur 1-4 | dur 5+ | **5** |

**结论**：
- surv_rate(duration) 单调不增，**单调性假设成立**
- 函数形态为阶跃式（0→1 跳变），非平滑递减
- Spearman rho 偏低（-0.02~-0.70）是因为阶跃函数有大量 tied ranks，不代表非单调
- **二分查找完全可行**：阶跃边界正是二分查找最擅长定位的场景
- 每个点只需 1 个 episode（确定性策略 + 固定 seed），无需统计存活率

**注意事项**：
- `direction_angle` 表示**力指向的方向**（机器人倒下的方向），不是力来源的方向
- MuJoCo 右手坐标系（z-up）中 +y 指向机器人左侧，因此用 `heading - angle`（顺时针）使 90° 对应右方
- 插件在 `on_pre_episode` 施力，此时机器人为 standing 姿态（pitch/roll≈0），heading 提取准确

#### 边界探测：全量并行扫描

**方法选择**：虽然二分查找在串行场景下 episode 数更少，但全量扫描可以一次性提交所有 episode 并行执行，实际墙钟时间更短，且边界精度更高（每个 duration 都有数据点）。因此采用全量并行扫描而非二分查找。

**扫描脚本**：`probe_boundary.py`

**力档位设置**（固定，不随策略迭代变化）：

| 档位 | 力 (N) | 物理意义 | 用途 |
|---|---|---|---|
| 轻 | 40 | ~0.2 倍体重，轻微推搡 | 弱策略边界低，强策略边界高，始终有区分度 |
| 中 | 100 | ~0.5 倍体重，明显冲击 | 常规训练难度 |
| 重 | 200 | ~1.0 倍体重，猛烈撞击 | 极限测试，只有很强策略才能恢复 |

**扫描参数**：
- 方向：16 个离散值（0°~337.5°，每 22.5° 一个）
- 力档位：3 个（40N / 100N / 200N）
- duration：1~40（全扫描）
- 每 cell 1 个 episode（确定性策略 + 固定 seed）
- 总 episodes：16 × 3 × 40 = 1920

**使用方法**：

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/probe_boundary.py \
    --policy-export baseline/runs/fixaw_survonly_crossphi2_s42/policy \
    --output baseline/humanoid21/balance_recover/boundary_fixaw_s42.csv \
    --json-output baseline/humanoid21/balance_recover/boundary_fixaw_s42.json
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--policy-export` | （必填） | 策略导出目录（含 policy_blueprint.yaml） |
| `--directions` | 16 方向（22.5° 间隔） | 逗号分隔的方向角度 |
| `--forces` | `40,100,200` | 逗号分隔的力大小 |
| `--duration-min` | 1 | 最小 duration |
| `--duration-max` | 40 | 最大 duration |
| `--workers` | 96 | 并行 worker 数 |
| `--seed` | 42 | 基础种子 |
| `--agent-id` | robot_a | 目标机器人 |
| `--output` | boundary.csv | 完整扫描数据 CSV |
| `--json-output` | boundary.json | 边界汇总 JSON |

**输出格式**：
- **CSV**：每行 (direction_angle, force, duration, survived, mean_len)，完整扫描数据
- **JSON**：每行 (direction_angle, force, critical_duration)，边界汇总 + 元数据
- **终端**：汇总表（每方向 × 每力的 critical_duration）+ 统计信息

**critical_duration 定义**：最后一个存活的 duration，即 dur ≤ critical 存活、dur > critical 摔倒。

**初代策略（fixaw_survonly_crossphi2_s42）扫描结果（2026-08-12）**：

| 方向 | F=40N | F=100N | F=200N |
|---|---|---|---|
| 0°（前） | 7 | 2 | 0 |
| 22.5° | 14 | 6 | 2 |
| 45° | 12 | 4 | 1 |
| 67.5° | 11 | 3 | 1 |
| 90°（右） | 12 | 3 | 1 |
| 112.5° | 12 | 3 | 1 |
| 135° | 13 | 3 | 1 |
| 157.5° | 10 | 2 | 1 |
| 180°（后） | 7 | 2 | 1 |
| 202.5° | 6 | 1 | 0 |
| 225° | 5 | 1 | 0 |
| 247.5° | 7 | 1 | 0 |
| 270°（左） | 3 | 1 | 0 |
| 292.5° | 4 | 0 | 0 |
| 315° | 5 | 0 | 0 |
| 337.5° | 6 | 2 | 0 |

**统计**：
- F=40N：mean=8.4，min=3，max=14 — 有很好的区分度
- F=100N：mean=2.1，min=0，max=6 — 中等区分度
- F=200N：mean=0.6，min=0，max=2 — 当前策略太弱，大部分直接摔倒

**方向模式**：
- 270°（左侧）最弱（crit=3），22.5° 最强（crit=14）
- 左右不对称：右 90°=12 vs 左 270°=3
- 前向（0°）和后向（180°）均为 7

**性能**：1920 episodes，96 workers 并行，21 秒完成

#### 权重分布生成

从全量扫描结果生成训练用的采样分布，核心是**以跳变点为中心分配权重**，让训练扰动集中在策略的生存边界附近。

**处理脚本**：`sample_distribution.py`

**权重设计逻辑**：

1. **找跳变点**：对每个 (direction, force) cell，在 duration 轴上找 1→0 的跳变位置。如 dur=7 存活、dur=8 摔倒，则 7 和 8 都是边界点
2. **duration 权重衰减**：以跳变点为中心高斯衰减（sigma=3），远离跳变点的 duration 权重递减。如跳变在 7-8，则 dur=6 和 dur=9 权重稍低，dur=1 和 dur=40 权重最低
3. **方向插值**：原始扫描只有 16 个离散方向（22.5° 间隔），在方向轴上做周期性线性插值，得到 360 个方向的权重分布
4. **方向抖动**：最终采样时在插值后的方向附近加 ±5° 抖动，避免每次精确采到同一角度

**使用方法**：

```bash
PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/sample_distribution.py \
    --input baseline/humanoid21/balance_recover/boundary_fixaw_s42.csv \
    --output-dir baseline/humanoid21/balance_recover/
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | （必填） | 全量扫描 CSV（probe_boundary.py 输出） |
| `--output-dir` | `.` | 输出目录 |
| `--sigma` | 3.0 | duration 权重高斯衰减的 sigma |
| `--n-interp` | 360 | 方向插值数 |
| `--n-samples` | 1000 | 采样数 |
| `--direction-jitter` | 5.0 | 方向抖动（度，±） |
| `--seed` | 42 | 随机种子 |

**输出文件**：

| 文件 | 说明 |
|---|---|
| `sample_weights.npz` | 权重矩阵（interp_angles, interp_weights, forces, durations, transitions），供训练时加载 |
| `samples.csv` | 1000 个采样参数（angle, force, duration） |
| `sample_distribution.json` | 完整分布数据 |
| `heatmap_survived_F{40,100,200}.png` | 存活/摔倒分布热力图 |
| `heatmap_critical_duration_polar.png` | 临界 duration 极坐标图 |
| `heatmap_weight_F{40,100,200}.png` | 各力档位采样权重热力图 |
| `heatmap_weight_total.png` | 三力叠加总权重分布 |

**初代策略分布统计**：
- F=40N: 41.1%，F=100N: 32.5%，F=200N: 26.4%（轻力占比高因为边界更宽）
- Duration mean=5.9，集中在边界附近
- 方向覆盖均匀（mean=167.6°，std=105°）

### 训练实验：加权冲量扰动训练

#### 实验构成

**实验文件**：`baseline/experiments_v2/exp_weighted_impulse.py`
**环境蓝图**：`baseline/humanoid21/balance_recover/weighted_impulse_env.yaml`
**插件**：`RelativeImpulsePlugin`（修改版，增加 `weight_npz_path` 参数）

**参照实验**：`baseline/experiments_v2/exp_basic_balance_v2_phi_dual_fixaw_survonly_crossphi2_impulse.py`

奖励设计和轨迹处理**完全照搬**参照实验：

| 奖励通道 | 计算 | Actor Weight |
|---|---|---|
| `r_fall` | 0.01 × φ(t) per step，无终止信号 | 固定 3.0 |
| `r_cross` | 交替支撑奖励/惩罚 | 1.0 × φ² |

- **双代理**：robot_a 和 robot_b 同时被扰动，各自独立计算奖励和轨迹
- **轨迹截断**：在代理终止步截断（imbalance 则截断到摔倒步，timeout 则用全长）
- **φ 加权**：r_fall 按 φ(t) 缩放，r_cross 的 actor weight 按 φ² 缩放
- **Warm-start**：从 BASE_POLICY_PATH 加载 checkpoint 权重

#### 与参照实验的区别

| 方面 | 参照实验 (crossphi2_impulse) | 本实验 (weighted_impulse) |
|---|---|---|
| 扰动插件 | `ImpulsePerturbationPlugin`（绝对方向） | `RelativeImpulsePlugin`（相对机器人朝向） |
| 方向定义 | `random_horizontal`（均匀随机） | 从权重分布采样（边界加权） |
| 力/时长 | 固定范围 [50,150]N / [2,4] steps | 从权重分布采样（40/100/200N × 1~40 steps） |
| 环境变量 | `POLICY_BLUEPRINT_PATH`, `BASE_POLICY_PATH` | 增加 `WEIGHT_NPZ_PATH` |

#### 插件修改

`RelativeImpulsePlugin` 新增两个参数：

| 参数 | 说明 |
|---|---|
| `weight_npz_path` | 权重分布文件路径。提供后按权重采样 (angle, force, duration)，忽略 force/direction/duration 固定参数 |
| `direction_jitter` | 方向抖动（度，±），默认 5.0 |

插件**不读任何环境变量**，所有参数通过构造函数传入。环境变量在实验类中捕获后通过 `env_bp.materialize()` 注入。

#### 启动命令

```bash
POLICY_BLUEPRINT_PATH=baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/policy_blueprint.yaml \
BASE_POLICY_PATH=baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/model.pt \
WEIGHT_NPZ_PATH=baseline/humanoid21/balance_recover/sample_weights.npz \
PYTHONPATH=/data1/mono/things/combatbench python3 baseline/framework/train.py \
    --experiment v2_weighted_impulse --algo ppo \
    --run-name recover_weighted_gen0 --no-snapshot
```

**环境变量说明**：

| 环境变量 | 必填 | 说明 |
|---|---|---|
| `POLICY_BLUEPRINT_PATH` | 是 | 内部 sim 的参考策略蓝图（`RelativeImpulsePlugin` 用此策略在内部 sim 中控制机器人，生成物理合理的扰动状态） |
| `BASE_POLICY_PATH` | 是 | Warm-start checkpoint（`.pt` 文件，加载 actor 权重） |
| `WEIGHT_NPZ_PATH` | 是 | 权重分布文件（`sample_distribution.py` 生成的 `sample_weights.npz`） |

**可选**：加 `--background` 后台运行，日志在 `run_dir/train.log`。

#### Smoke 测试

```bash
POLICY_BLUEPRINT_PATH=... BASE_POLICY_PATH=... WEIGHT_NPZ_PATH=... \
python3 baseline/framework/train.py --experiment v2_weighted_impulse --algo ppo --smoke --no-snapshot --run-dir /tmp/test
```

已验证：2 updates 完成，survival_rate 0.125 → 0.500，权重采样正常工作。

### 为什么这样做

- **状态在流形上**：`ImpulsePerturbationPlugin` 通过内部 sim + 策略生成物理合理的扰动状态，动力学一致
- **难度自校准**：先测当前策略能承受的扰动程度，再用边界参数训练
- **无状态池问题**：实时生成扰动，不存在代内分布偏移（DAgger 问题）
- **结构性保留基础平衡**：同一个策略既要探测边界、又要被训练，丢掉基础平衡则迭代断掉
- **边界即学习信号**：边界参数下的扰动让策略学到"差一点就救不回来"的临界控制

### 关键设计

#### 边界参数化

用 **(受力方向, 受力持续时间)** 作为主参数空间，力大小设为固定档位，施力部位固定为 torso。方向 16 个离散值 × 力 3 档 = 48 个组合，每个组合在 duration 轴上二分查找边界。拟合 (direction, force) → critical_duration 曲面后从中采样训练参数。方向和持续时间物理可解释——不同方向扰动不同方向的恢复能力，持续时间越长越难恢复。

#### 两阶段奖励

| 阶段 | 目标 | 生效项 |
|---|---|---|
| 恢复瞬态 | 别摔，怎么野都行 | `r_fall` + `r_cross`（弱），关掉姿态惩罚 |
| 稳态（末尾 N 步） | 回到平静交替平衡 | 全部姿态项 + 交替项，阈值按初代实测校准 |

阶段划分用事件触发：从"首次满足直立 + 双支撑交替"起进入稳态。

#### 阈值重标

按初代 `basic_balance_v2` 实测工作点校准：

| 量 | 当前阈值 | 初代实测 | 建议死区 |
|---|---|---|---|
| `joint_vel` | 0.1 | 1.23 | ~1.5 |
| `joint_deviation` | 0.1 | 0.23 | ~0.30 |
| `foot_height` | 0.10 | 0.063 | 0.10（合理） |
| `torso_tilt` | 0.26 | 0.165 | 0.26（合理） |

#### `recovered` 正向定义

末尾连续 N 步满足：躯干直立度 > 阈值 ∧ 出现有效左右交替支撑 ∧ 无墙接触 ∧ 关节速度在死区内。

#### 晋级/停止判据

- **主判据**：eval 成功率超过目标阈值（如 80%）
- **停止条件**：连续 M 轮 eval 成功率不再上升
- **守卫判据**：基础平衡（无扰动）成功率不退化 + 稳态躁动度不退化
- 增大 eval_episodes 或用统计检验，避免噪声晋级

---

## 落地顺序

| 步骤 | 产出 | 状态 |
|---|---|---|
| **0** | 单调性验证：`verify_monotonicity.py` 跑全 duration 扫描，确认 surv_rate 对 duration 单调递减 | ✅ 完成 |
| **1** | 全量并行扫描边界：`probe_boundary.py` 16方向×3力×40duration=1920 episodes | ✅ 完成 |
| **2** | 权重分布生成：`sample_distribution.py` 从扫描结果生成 `sample_weights.npz` + 热力图 | ✅ 完成 |
| **3** | 训练实验：`exp_weighted_impulse.py` + `weighted_impulse_env.yaml`，从权重分布采样扰动参数 | ✅ 完成 |
| **4** | 正式训练：从 u00460 checkpoint 启动 `recover_weighted_gen0` | 🔄 进行中 |
| **5** | 迭代循环：训练完成 → 重新探测边界 → 更新分布 → 再训练 | 待定 |

---

## 已知局限

- **π_nom 依赖**：扰动状态分布取决于 `ImpulsePerturbationPlugin` 内部 sim 使用的参考策略。未来换成格斗策略后受击状态会很不一样，需更新参考策略。
- **墙壁污染**：大冲量下 79% 的"存活"靠墙撑住。当前配置下不严重（lv11 = 12.8%），但需监控。
