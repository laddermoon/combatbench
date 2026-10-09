# M4 计划：standup 任务转换与环境级验收

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

对应 [ROADMAP.md](ROADMAP.md) §7。输入：`standup_4stage_dense_v2_env.yaml`（`exp_standup_floor04.py` 的目标环境）。M3 已交付设备数据平面、BatchRuntime、观测构建器（96 维，host 对照 ~5e-7）、`DeviceTimeoutPlugin`。

## 1. 转换对象审计

### `RandomFallenStatePlugin`（plugins: target both, max 1000 substeps, h<0.3, interval 5）

CPU 语义（disturbance_plugins.py:736）：
1. 读真实环境 core state → 写入**内部独立** `Humanoid21Simulator`；
2. 目标机器人设 uniform(-1,1) 随机 action（**每 episode 一个常量**）；
3. 内部 sim 逐物理步推进；非目标机器人每 `reset_interval` 步重置回初始；
4. 目标 root 高度 < 0.3 提前终止，否则跑满 `max_phy_steps`；
5. 摔倒后 core state 经 `ctx.mutator.set_core_state` 写回真实环境。

**批量原生等价物**：内部 sim 可完全省去——pre_episode 时真实 batch 状态本就是刚 reset 的站姿，直接在 warp sim 上并行跑摔倒 rollout：

- 每 env 一个随机 action（torch，seed_offsets 派生——**分布等价、样本不逐位复现**，discuss.md §83 已豁免）；
- 逐子步查 `qpos[:, root_adr+2] < threshold` → per-env settle 掩码；首次达标时**捕获该 env 的 qpos/qvel 行快照**，其余 env 继续；
- 全部 settle 或 max_steps 后，`dev_set_integration_rows` 恢复各 env 的首次达标行；
- 非目标机器人 interval 重置 = 每 k 步写回非目标关节行切片（本 blueprint 双目标，路径仍实现以备复用）；
- metrics：per-env fallen_init_steps / fallen_init_height / threshold_met 入插件池。

已知差异（诚实声明）：CPU 内部 sim 是 fp64 CPU 物理，warp 是 fp32 CUDA——**摔倒姿态分布只能统计等价，不能逐位一致**。验收走两步（ROADMAP §7）：先注入 CPU 生成的相同倒地状态验下游逻辑，再做原生 reset 分布统计验收。

### `StandingBalance4StageRewarder`（observer × 2，per-agent）

纯计算插件（standing_balance_4stage.py），per-agent 输出 {stage, potential, f_score, contact_score, d_score, d_hf, w_foot, h_score, h_torso}。全部信号可向量化：

- `f_score`：torso quat → `-2(xz-wy)`，clip 归一；
- `contact_score`：hand/foot 高度 proximity × `1/(1+0.5·extra_count)`；
- `d_score`：hand-mid 与 foot-mid 的 **XY** 距离；
- `w_foot`：`f_foot/(f_hand+f_foot)`，`F_LOAD_MIN=10N` 门限；
- `h_score`：`(h_torso-0.15)/(1.28-0.15)` clip；
- stage 自上而下判定 + 分段 potential（0.1/0.1/0.1/0.7 带宽）。

接触解析依赖 per-contact 分类：ground-side geom == 'ground'、robot-side body 归属（foot_l/r、hand_l/r、其它）、force≥1N、**extra_count 是 distinct body 去重计数**。flat contacts 上向量化：robot-side body → 类别码 scatter，extra_count 用 (B, nbody) bool 累加后求和。

**常量单一来源**：`H_*`/`D_*`/`F_*`/`OTHER_PENALTY_K`/带宽全部 `from baseline...standing_balance_4stage import`——不复制字面量，保证双侧联动。

## 2. 工作包与执行顺序

### T1：接触力提取共享化（重构 device_obs.py）

把 `_feet_forces` 里的 flat-contact force 分解（efc 行 gather → frameᵀ@f_local → force_mag）提成 `contact_forces_flat(state) -> (B?) (C,) 张量` 共享 helper；观测构建器与 rewarder 复用同一份实现，避免接触力计算两处漂移。附带提供 robot-side body 分类表（body_id → {foot_l,foot_r,hand_l,hand_r,other,env}）。

### T2：`DeviceStandup4StageRewarder`（native observer）

- `BaseDeviceObserver` 实现，`agent_idx` 参数化（0/1）；输出 dict of (B,) 张量；内部 stage/potential 等状态走 `declare_state` pool（partial reset 行清零 + 本插件 on_envs_reset 重置 stage=1——对齐 CPU `on_pre_episode` 清零语义）。
- **V1 验收（同态逻辑对照）**：CPU 轨迹/姿态快照经 `dev_set_integration_rows` 注入 → 双侧 rewarder 输出逐字段比对。覆盖四阶段代表姿态 + 门槛附近样例（d_score≈D_GATE、f_score≈F_ENTER、total_load≈F_LOAD_MIN、h≈0.3），接触 body 去重、力阈值单列断言。容差：fp32 级 {atol 5e-3, rtol 1e-2}（含物理注入差异时标注为物理侧）。
- 注册表更新为 NATIVE + factory。

### T3：`DeviceFallenResetPlugin`（native resetter）

- `BaseDevicePlugin`，`on_pre_episode` 内驱动 `sim.physical_step` 循环（不进 runtime 的 episode 簿记）。
- 逐 env 捕获/恢复 + 非目标 interval 重置 + per-env RNG action。
- **验收 A（下游逻辑）**：用 CPU 生成的倒地 core-state fixture 经 `dev_set_integration_rows` 注入，验证 rewarder 首步输出与 CPU 一致（排除 reset 差异混淆）。
- **验收 B（分布等价）**：双侧各生成 N≥64 个倒地初态（CPU 内部 sim 法 vs 设备原生），比对统计量：root 高度分布、着地 body 集合分布、关节位置主成分、steps-to-settle 分布。门限在实测后冻结并如实记录；**禁止为通过而放宽**。
- RNG：`hash(seed_offset, salt)` 派生 per-env action；与 CPU 样本不等但同分布。

### T4：环境级固定策略交叉评估（V3 门槛）

- 设备端 collector 最小闭环：`rt.step(actions)` + obs/reward 导出（仅本阶段验收用，非 M5 rollout）。
- 取 CPU 侧**未训练/中期/成熟**三档固定策略 checkpoint（M0 已冻结）+ **相同倒地初态注入**，双侧各跑若干 episode，比对：reward 轨迹逐字段统计、potential 终值/峰值分布、episode 长度分布。
- batch 敏感性：B=4 vs B=64 统计一致（部分 reset 频率差异不引入系统性偏移）。
- 门限沿用 M0 冻结：success 非劣效 ≤2pp、final potential 非劣效 ≤0.02。

### T5：收尾

- `M4_RESULTS.md`：差异表、验收证据、失败记录；blueprint→device 转换档案（输入清单/转换规则/未覆盖项）作为 M8 AI 迁移的样例资产。
- 注册表把本实验全部蓝图类标 NATIVE。

## 3. 放行标准（ROADMAP §7 对齐）

- V1 同态逻辑对照通过（reward/observer 在相同状态特征上逐字段一致，fp32 容差）。
- reset 分布验收通过或有定量差异报告；初态分布无系统性偏移。
- V3 固定策略交叉评估达到冻结门限；B 敏感性无系统性差异。
- 接触 body 去重/力阈值/承重比在门槛附近样例单独统计。
- 所有字段显式存在，不以零值/空输出掩盖缺失。

## 4. 明确不做

- 不接 rollout/PPO/Debug 数据链路（M5）。
- 不转换蓝图外的插件（RandomMove、ImpulsePerturbation 等保持注册表 PENDING/UNSUPPORTED）。
- 不做渲染/录像（永远 HOST_SLOW）。
- 不追求摔倒姿态逐位一致（分布等价即可，语义见 §1）。

## 5. 暂停条件

- 固定旧策略在 warp 上系统性退化（先修环境，不让 PPO 适应错误实现）；
- 原生 reset 分布出现无法解释的偏移；
- rewarder 门槛附近样例出现逻辑性分歧（非 fp32 噪声）。
