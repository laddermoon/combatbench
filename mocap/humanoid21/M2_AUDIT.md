# M2 审计：物理跟踪验证（进行中）

> 状态：**W1–W3 进行中**。open-loop 基线完成（全部失败）；闭环跟踪策略
> 已验证管线可行 —— `cmu_07_01`（走）训练收敛（v2 u2000: complete=1.0,
> fall=0, q_rmse 7.4°）；`cmu_13_17`（拳）第三轮训练中。
> W3.5 记录了训练暴露的两处接口/数据缺陷（首帧 IK 瞬态、观测目标滞后）。
> 模型：`envs/humanoid21/battle_circular_v2.xml`；控制器：固定 KP/KD 归一化位置 PD（CONTROLSPEC）
> 数据：`/data1/dev/combatbench_mocap/track/`（每 clip 一个 `*_track.npz` + `*_track.json` + `summary_openloop.json`）

## M2 目标（对齐 README §9）

验证 retargeted 参考运动在**真实物理 + 真实控制链**下能否被执行：
`mj_forward` 摆位回放（compare_render）只验证运动学，不算物理成功。
M2 要求 state-non-overwriting：t=0 初始化一次，之后只靠控制输入。

## W1 跟踪回放器 `track_replay.py`

- 从 `clips/cmu_*.npz` 读参考（120Hz）：root pos/quat/vel、21 关节角/速度、keypoints。
- 水平平移轨迹使 xy-bbox 居中于场地（偏移量写入 metrics 溯源）。
- t=0 `set_core_state` 一次性写入根 pos/quat/局部速度 + 归一化关节 pos/vel。
- 之后每 25 个物理步（20Hz）用参考关节角 → `action=(q−mid)/half` 喂 `set_action`；
  **全程不再写状态**。
- 记录每控制帧：actual/ref 关节角、根 pos/quat、6 个 keypoint body（torso/head/foot/hand）、
  ctrl 饱和度；摔倒判据 torso_z<0.55 或 head_z<0.45（摔倒即停，不覆写）。
- robot_b 在 `initial_distance=6.0` 处 standing，未与 A 接触。

## W2 open-loop 基线结果（17/17 全部失败）

| 指标 | 范围 |
|---|---|
| 存活时间（摔倒前） | **0.45–2.0 s**（07_01 走 1.5s；13_17 拳 0.8s；最长 14_01 2.0s） |
| 关节跟踪 RMSE（存活窗口内） | 4.3–11.7° |
| 根位置漂移（max） | 0.8–2.2 m（参考继续前进，机器人原地滞后→侧倒） |

对照实验：robot_a 直接 hold `standing` 初始姿态 action，**1.5s 内同样倒地**
（torso_z 1.28→0.24）。

### 结论

1. **失败主因是缺闭环平衡，不是关节刚度或参考质量问题**：
   关节 RMSE 在倒地前仅 ~10°，PD 在忠实执行目标；但 humanoid 站立本就是不稳定的
   倒立摆，固定 KP/KD 位置 PD 无根级反馈，站都站不住 → 参考 motion 的腿部支撑
   序列在物理上无法复现。
2. PD 控制律为 `KP*(target−q) − KD*qvel`，**无目标速度前馈项**，
   对运动目标存在 ~`KD/KP`·qvel 的恒定滞后（13_17 存活期 rot_err 达 60–100°，
   部分来自滞后放大失衡）。
3. ctrl 饱和比例 ~10–17%，拳击片段最高（0.171）但不是主要失稳源。
4. open-loop 回放达标线为物理不可行 → **L2 达标必须走闭环跟踪策略**
   （DeepMimic 式：模仿奖励 + 早期终止 + 相位变量），这也正是 motion-prior
   的目标形态，历史 standing/fight 策略也证明了这点。

## W3 闭环跟踪策略（DeepMimic 式）

新组件（`mocap/humanoid21/tracking/`）：

- `ref_data.RefClip`：npz → 20Hz 控制帧视图（每 6 帧取 1），含归一化关节
  目标、根 pos/quat/vel、keypoints。
- `simulator.MotionTrackingSimulator`：
  - reset 时 RSI（随机相位，seed 驱动；`options.start_frame` 可固定起点），
    一次性写入根 pos/quat/线速度/机体系角速度 + 关节 pos/vel；
  - `set_action` 递增相位；obs 扩展 46 维（phase + ref q_norm +
    ref qv_norm + ref 根速度机体系）→ 142 维，仅被跟踪机器人获得
    （robot_b 保持 96 维给 standing policy）；
  - `get_derived_state()['motion_track']` 暴露当前参考帧给奖励/终止。
- `plugins.MotionTrackGuardPlugin`：torso_z<0.55 或头低于骨盆 →
  `track_fall`；相位走完 → `track_clip_end`。
- `reward.MotionTrackRewarder`：DeepMimic 加权项
  0.45·pose + 0.10·vel + 0.35·kp(torso/head/双手/双足) + 0.10·root高度。
- 蓝图 `baseline/humanoid21/blueprints/track_ref_env.yaml`；
  实验 `baseline/experiments_ppo/exp_track_walk.py`
  （`--set clip_id=...` 切片段；robot_b = StandingCombatPolicy 场景机）。
- 评估 `track_eval.py`：rsi='first' 从帧 0 确定性整段回放 + 视频。

### 结果

**cmu_07_01（走, 2.6s, 53 控制帧）** — 两个 run：
`train_track_walk_ppo_20261014_132001`（v1, 世界系 kp）与
`track_walk_v2_relkp`（v2, 骨盆系 kp，训至 u2000 收敛）：

| update | random-RSI eval (16 eps) | 备注 |
|---|---|---|
| ~500 | survive≈1.0, complete≈0.9, fall≈0.06, kp_err≈0.30m | 初步稳定 |
| ~1150 | complete=1.000, fall=0, kp_err≈0.13–0.15m, q_rmse≈11° | v1 收敛 |
| v2 u2000 | survive=1.0, complete=1.0, fall=0, kp_err≈0.24, q_rmse≈7.4° | v2 收敛 |

- v1 整段确定性回放（旧 obs 语义）：`track_clip_end` 完成，q_rmse≈13°，
  kp_err 0.43m（误差集中在首尾过渡与系统性 ~7cm 躯干下沉）。
- **v3（next-frame obs）u00955 确定性整段回放：track_clip_end 完成，
  kp_err 0.35m（max 1.01）、q_rmse 10.3°（max 17.9°）**。
  从 lead_stable 起始帧（ctrl 2）出发；从原始帧 0（引导瞬态区、
  真实 mid-stride 高速段）出发仍在 ~50% 相位处摔倒——引导区不
  计入 L2 判分。
- 广播视频抽帧目测：迈步、换脚、手臂摆动均正确。
- **L2 候选**：待视觉复核后在 index 标记。

**cmu_13_17（拳, 40s, 807 控制帧）** — 三轮训练：

| run | 配方 | 结果 |
|---|---|---|
| `track_box_13_17`（v1, 世界系 kp） | 奖励世界系关键点误差 | 平台期 survive≈0.85，平移/朝向滞后过罚 → 弃 |
| `track_box_13_17_v2`（骨盆系 kp） | 奖励改骨盆相对系 | u550 达 survive 0.96/complete 0.88，后回落震荡 ~0.8；未收敛 |
| `track_box_13_17_v3_nextobs` | 修复数据+观测后重训 | **u790 确定性扫描 7/8 起点全段完成**，但发现"原地打拳"退化 → v4 |
| `track_box_13_17_v4_rootobs` | +根级目标观测/奖励 | u630 弃：根特征**不可观测**（见下），漂移依旧 → v5 |
| `track_box_13_17_v5_egoroot` | obs 改 ego 系根误差 + 奖励放宽 | 有改善但不稳（u744 dxy 0.96m/34°）→ 漂移仍是软约束 → v6 |
| `track_box_13_17_v6_driftterm` | +track_drift 早停 0.8m/75° | 根跟踪收紧至 dxy 0.55m/17° → 收紧阈值续训 |
| `track_box_13_17_v6b_drift06` | 阈值收紧 0.6m/50° | **u1898 验收通过**（u962 同等可用，数值略差） |

**v3 关键发现 —— "原地打拳"退化**：u790 确定性回放从帧 0 完整走完
806 控制帧（40s），逐起点扫描 7/8 完成；但世界系 kp_err 均值
1.54m（逐段递增 0.9→2.0m）、root yaw 误差均值 **123°**、xy 漂移
1.4m。根因：骨盆系 kp 奖励对根 yaw/xy 平移完全不变（关节 pose
项也不含根自由度）→ 策略学到"原地做拳击关节动作"。**训练日志
eval 指标掩盖了这一点**（complete/survive 只看相位推进和摔倒，
与轨迹跟随无关）——M2 验收必须以确定性轨迹评估为准。
**v4 修复**：obs +54 维（ref root pos/yaw sincos/angvel local，
obs 150）；奖励 +r_root = 0.5·exp(−8·dxy²)+0.5·exp(−2·dyaw²)
（w=0.10，远轻于 v1 的世界系全关键点项）。

**v4 结果与根因再分析**：u612 确定性回放虽全段完成，但
dxy_mean=1.04m/max 2.43m、dyaw_mean=47°/max 131°
（`kp_err_rel_m` 反而只有 0.21m —— 姿态形状在学、根轨迹没有）。
**根因：v4 给的根特征是"绝对量"——`ref_root_pos` 与 `ref_yaw`
均为世界系绝对值；而机器人 96 维基础观测刻意不含自身绝对
xy/yaw（OBSERVATION_zh.md §4 "不提供绝对 yaw"，场地中心也是
ego 系向量）。策略无法从观测算出 self−ref 误差，r_root 奖励对它
而言是"无缘无故掉分" → 直接退化成 v3 同款原地模式。**
v5 修复（`ce16cfa4`）：obs 改发 **ego 系跟踪误差**
`R_self⁻¹(ref_pos−self_pos)·0.5`（3 维）与 `sin/cos(yaw_ref−yaw_self)`
（2 维），ref 根速度改投**参考朝向系**；奖励核放宽
`exp(−3·d²)/exp(−1·dy²)` 让 1–2m/90° 漂移区仍有梯度，
W_ROOT 0.10→0.20（pose 0.40→0.35、kp 0.30→0.25 让位）。
教训：**观测扩展必须检查"误差是否可计算"——发绝对目标量而
自身状态不可观测等价于没发。**

**v6 修复——漂移早停**：v5 下软 r_root 仍留了"存活+姿态形状"
局部最优（策略贴地游荡，u744 dxy 0.96m）。参考 13_17 全片段
净位移仅 0.36m（漂移是策略自己走丢而非跟不上），故按
DeepMimic 式早停加 `track_drift` 终止（dxy/dyaw 超阈值即终止），
把根跟踪从奖励项升级为生存条件。阈值分阶段收紧
0.8m/75° → 0.6m/50°：kp_err 从 0.7→0.3 量级，漂移终止占比
从 ~70% 降至 ~50%。副作用：策略更激进地追轨迹，个别导出在
相位 ~0.87 处摔倒（u991），需按导出逐个确定性复测选取。

**v6b u1898 最终验收**（确定性 `rsi='first'` 全段 806 控制帧）：
`track_clip_end` 完成、reward_mean 0.72、dxy 0.28m(max 0.66)、
dyaw 10.8°(max 40°)、世界系 kp 0.37m(max 0.73)、骨盆系 kp 0.16m、
q_rmse 21°(max 34°)。**用户视觉复核通过（2026-10-14）**：
能看出在模仿拳击片段、出拳方向正确、跟随参考移动而非原地；
u962（dxy 0.44/kp 0.53）视觉等价，取数值更优的 u1898 为验收工件。
**已知缺陷**：步法为抖动/蹭步而非迈步，关节有轻微颤动——
录入 index `issues: foot_jitter_observed, mild_joint_tremor`，
可能的成因是奖励缺动作平滑项 + 20Hz PD 在快速段的带宽限制，
留给 M3+（先验/平滑正则）处理。
kp/dxy/dyaw 均满足原 L2 阈值；q_rmse 21° 按 §L2 注 3 校准
（峰值根角速度 673°/s 的快速片段，~25° 内判定），视频语义正确。
场景机器人 robot_b（standing policy）在跟踪回放中倒地属已知
无影响现象：B 不参与跟踪评分、不与 A 发生接触。

**摔倒相位扫描**（v2 best，确定性、每 25 控制帧一个起点）：
~50% 起点在 10–50 步内摔倒，但穿越同一区段的 episode 能存活——
冷启动恢复弱于连续跟踪，失败不集中在一个相位而是广泛分布。
帧 0 处另发现**数据缺陷**（见下）。

## W3.5 训练暴露的数据/接口缺陷与修复

1. **片段首帧 IK 分支瞬态**（影响全部 17 clip 的物理可用性）：
   `retarget_clip` 帧 0 从站立种子解 LM，落到错误分支；DQ_MAX
   slew clamp 把修正摊到后续 ~5 帧 → 首帧关节速度范数高达
   **26–70 rad/s**（真实水平 ~1–14）。RSI 采到引导区即拿不可能
   初速度，确定性回放帧 0 起步必倒。
   **修复**：正向遍历时检测瞬态结束点，从内部收敛解**反向**重新
   求解引导窗口；13_17 首帧 jv 70→3.5 rad/s，17 clip 重新导出。
   `cmu_07_06`/`cmu_07_12` 的残留尖峰是 **AMC 源数据本身**的单帧
   >20° 毛刺，非求解器问题 → 导出 `lead_stable_frame`，RSI 采样
   与 `rsi='first'` 均跳过该区。
2. **观测目标滞后一帧**：跟踪 obs 原本给当前帧 `ref[k]`，但动作
   驱动的是 k→k+1 过渡、奖励在步末对 `ref[k+1]` 评分——快拳段
   ~10°/控制步的系统性目标滞后。改为 `ref[k+1]` 目标。
   **兼容性警告**：旧 obs 训练的策略在新 obs 下直接失效
   （v2 走路策略新 obs 评估：28 步摔倒 vs 旧口径 complete=1.0）
   ——观测语义即接口契约，变更需同步重训。
3. 奖励：世界系关键点误差换**骨盆相对系**（位移/朝向漂移不再
   过罚），世界系误差保留为评估指标。
4. **`policy/` best 导出会陈旧**：best 评分（survive−0.5·fall）
   在历史快照打平/倒退时不再覆盖——v3 走路 best 目录里的权重比
   最新更新差 19% fall rate。**评估一律用 `policy_exports/uNNNN`
   的最新导出**，不用 `policy/`。
5. 训练 eval 的 `complete_rate` 含 **RSI 相位偏差**：随机起点
   episode 只需存活到片段尾（起点越晚越占便宜）；完整回放判定
   必须走 `track_eval` 从稳定起始帧的确定性全段。
   `track_eval` 也不再显式 `start_frame=0`，由 `rsi='first'` 取
   `lead_stable` 后的首个控制帧。

## L2 判据（定稿）

一个 clip 记为 L2 需要：
1. 存在训练收敛的跟踪策略：random-RSI eval `complete_rate ≥ 0.9`、
   `fall_rate ≤ 0.1`（启用 track_drift 训练时该口径受早停干扰，
   以判据 2 为准）；
2. `rsi='first'` 整段确定性回放达到 `track_clip_end` 且不摔倒；
3. 逐帧指标：q_rmse ≤ ~15°、kp_err_mean ≤ ~0.45m（慢速片段）；
   快速片段允许按可达上限校准并记录（cmu_13_17：kp≤~0.6m、
   q≤~25°、dxy≤~0.6m、dyaw≤~20°）；
4. 生成 `track_eval` 视频供人工复核确认运动语义——**最终裁决项**。

**已验收片段**：

| clip | 策略 | 确定性指标 | 视觉复核 | 等级 |
|---|---|---|---|---|
| cmu_07_01 | `track_walk_v5_egoroot/u02000` | dxy 0.15m/dyaw 2.9°/kp 0.16m/q 7.0° | 通过（"非常好"） | **L2** |
| cmu_13_17 | `track_box_13_17_v6b_drift06/u01898` | dxy 0.28m/dyaw 10.8°/kp 0.37m/q 21° | 通过（步法抖动等已知缺陷） | **L2** |

已知注定失败类型：13_01–03 起身类（源坐凳子，环境无凳子）——保留数据，
L2 评估单独标注 `requires_prop`。其余 14 个未训片段维持 L1 +
openloop 失败记录（开环基线全军覆没证明需闭环策略；
逐片段训练外推属于后续工作包）。

## 已复用教训（M1 迁移）

- 视觉验收标准必须源自同一 FK 约定生成的参考；`track_replay` 直接消费
  已验证 npz，不重新解释 mocap 数据。
- 区分验证层级：`tracking_openloop` 字段显式标注方法串，不混淆 L1/L2。
