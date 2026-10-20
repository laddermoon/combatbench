# Strike 两阶段训练方案（strike_2phase）

> 类型：实验方案  ·  实验：`exp_strike_2phase.py`（`strike_2phase`）
> 环境：`blueprints/strike_2phase_env.yaml`  ·  数据：`assets/clips/full/`（SSOT）

## 1. 定位

打击技能训练的**干净重写版**。前置版本 `skill_2phase_mbs` 挂在
`SkillTrain→Trans→Gate→Step→Full` 继承链上，层叠机制过多、无法
确认奖励语义被正确实现，已废弃。本版**只继承
`CombatExperimentPPOBase`**，所有逻辑在一个文件内显式写出。

范围收敛为**只训四个打击技能**（jab_l / jab_r / heavy_l /
heavy_r），不掺移动/转身/踏步任务，目标是快速把打击链路跑通。

## 2. Episode 结构

每个 episode = **热身起手 → 单条打击指令**：

1. `WarmupSteppingPlugin`：把真实状态拷进内部 sim，用**当前训练
   策略**（`episode_options["warmup_policy_cls"]` 每 job 注入最新
   导出 policy，`sample()` 随机采样）跑 10 步零指令原地踏步，
   终点姿态写回作为初始状态 → 阶段一从"步态中间态"出发，姿态
   分布跟随策略自身步态。
2. `CommandSimulator`（v2，`strike_dyn_target=true`）：执行一条
   strike 指令行 `[n,0,0,0,0, +1,side,land_x,land_y,land_z]`，
   obs 中的手靶回放参考 clip 出拳手轨迹（随时间移动）。

指令长度 n 按落地帧 + 余量定：jab = 11+13=24 步，heavy =
6+6+13=25 步。episode 无摔倒终止，固定跑满（倒地数据由 reward
结构处理，见 §4 护栏）。

## 3. 两阶段语义与边界

**阶段一**（对齐）：从当前姿态进入出拳准备姿态。准备姿态 = 取
机器人当前 XY 位置与水平朝向，应用于 clip 第 0 帧定义的关节与
身体位姿（ego 锚定）。

**边界（launch gate）**：所有受控关节（去踝 17 个）与 clip 帧 0
的归一化误差全部 < 各自量程的 **5%**（|q_err|<0.10）。**固定
判据**——每个 job 注入 `launch_gate={"frames":0,"frac":0.05}`，
不走自适应课程。

**阶段二**（出拳）：从准备姿态到出完拳。要求符合 clip 的
motion，且**后脚（发力脚）不能动**。

## 4. 奖励通道

全部为**负势场**形式（每步计算，越小越好，reward = −Φ）。
目标关节为去踝 17 关节的归一化姿态误差 e_j。

| 通道 | 定义 | 阶段 | AW |
|---|---|---|---|
| `r_strike_p1_mse` | −mean(e_j²) vs clip 帧0 | 一 | 4 |
| `r_strike_p1_var` | −var(e_j²)：防单关节大偏被均值摊薄 | 一 | 4 |
| `r_strike_p2_mse` | −mean(e_j²) vs **落地帧 k_max** | 二 | 4 |
| `r_strike_p2_var` | −var(e_j²) vs 落地帧 | 二 | 4 |
| `r_strike_p2_foot` | −min(‖Δp_foot‖, 0.5m)：后脚世界系 3D 位移罚 | 二 | 8 |
| `r_potential` | +0.01·min(phi, 0.9)：稠密直立护栏 | 全程 | 1 |

护栏说明：phi≥0.9（正常打击姿态都在其上）时 `r_potential` 为
常数 bonus，不塑形动作；只在失衡时提供回拉梯度。显式 `r_fall`
已移除——所有打击通道 ≤0，摔倒若再被一次性重罚反而增加一处
需要调平的激励，护栏只留稠密项。

**后脚判定**：clip 帧 0 骨盆 ego 系下前向坐标较小的脚；前后差
<5cm 的近似平行站姿（jab）退化为全段总位移较小的脚。已验证：
原始 clip → 右脚，镜像 clip → 左脚。位移锚点记录在**过门那一
帧**的后脚位置（非过门后首步），3D 位移（抬脚/滑步都算，绕脚
尖旋转的枢轴近似不动、允许蹬转）。

## 5. 阶段隔离：按 Trajectory 拆分

observer 每步发射 `gated`；**过门后的第一步成为轨迹边界**——
rollout 后 `build_trajectories` 把 episode 切成两条 Trajectory：

- **轨迹一 `[0,g)`**：挂 `r_strike_p1_*` + `r_potential`；p1 通道
  `is_terminated=True`——回报在边界截断，不向阶段二状态
  bootstrap（那些状态 p1 critic 从未见过）；`r_potential` 末端
  正常 bootstrap `V(obs[g])`。过门当步仍计阶段一。
- **轨迹二 `[g,T)`**：挂 `r_strike_p2_*` + `r_potential`。
- **未过门**：单轨迹，只挂 p1 通道 + `r_potential`（不挂死通道）。

GAE 按轨迹计算，两阶段 critic/优势天然互不污染——不需要靠
AW=0 掩码近似隔离（AW 掩码只挡 actor 梯度、挡不住 critic 的
跨阶段 bootstrap）。

## 6. Job 构造与采样

- `build_jobs` 独立实现（不继承旧链）：`skills` + `move_schedule`
  + `launch_gate` + `warmup_policy_cls`，`initial_distance=2.0`。
- 双机器人独立抽技能；`SKILL_SAMPLE_W` 给镜像侧加权 ×2
  （jab_r/heavy_l=2.0）——历史数据显示镜像侧为弱侧。
- rollout：1536 ep/update，96 workers；eval 每 5 update、64 ep。

## 7. Eval 指标

| 指标 | 含义 |
|---|---|
| `{skill}_gate_qmax` | episode 内最小过门关节误差（判据 0.10） |
| `{skill}_ungated` / `gate_t` | 未过门率 / 过门耗时 |
| `{skill}_landed` / `land_t` | 落地帧命中 / 推进帧位 |
| `{skill}_hand` | 末帧手误差（m） |
| `{skill}_foot` | 阶段二后脚最大位移（m） |
| `falls` / `quality` | 摔倒率 / best 跟踪综合分 |

早停：500 update 无 best 提升且 ≥300 update。

## 8. 已知取舍

- **"符合 clip motion"只被端点近似**：阶段二评落地帧姿态 + 后脚，
  不直接评中间路径形状（指针/进度项不在本方案内）。若实测轨迹
  直插终点不走发力路径，候选补丁：进度/路径通道或 AMP 风格通道。
- **落地帧 = 伸展终点**（ext_only 指针 `k_max`），收招/恢复不评。
- 后脚约束是世界系点位移，不直接保证触地承重（可看后续接触
  诊断）。

## 9. 运行记录

- 首次启动：`train_strike_2phase_ppo_20261020_145535`，从
  `skill_2phase_mbs` run 的 u07200 checkpoint 恢复（actor 权重
  继承，打击 critic 冷启动），GPU1。
