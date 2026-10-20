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

1. `WarmupSteppingPlugin`：每台机器人独立均匀抽一个热身技能
   （四向移动、双向转身、原地踏步、四种打击；**排除随后要训练
   的同一个打击技能**），热身步数独立均匀抽 `[0,20]`（包含两端，
   0 表示保持出生状态）。`episode_options["warmup"]` 注入各自
   技能 schedule 与步数；`warmup_policy_cls` 注入当前导出策略，
   用 `sample()` 在内部 `CommandSimulator` 执行热身并写回状态。
   热身帧**不进入打击训练轨迹**，仅改变阶段一起手分布。
2. `CommandSimulator`（v2，`strike_dyn_target=true`）：执行一条
   strike 指令行 `[n,0,0,0,0, +1,side,land_x,land_y,land_z]`，
   obs 中的手靶回放参考 clip 出拳手轨迹（随时间移动）。

所有打击指令和 episode 均固定 **30 步**。无摔倒终止；成功后
episode 可以继续执行，但成功之后的帧不进入训练轨迹。

## 3. 两阶段语义与边界

**阶段一**（对齐）：从当前姿态进入出拳准备姿态。准备姿态 = 取
机器人当前 XY 位置与水平朝向，应用于 clip 第 0 帧定义的关节与
身体位姿（ego 锚定）。

**边界（launch gate）**：首次所有受控关节（去踝 17 个）与
clip 帧 0 的归一化误差全部 ≤ 各自量程的 **5%**
（max|q_err|≤0.10）。**固定判据**——每个 job 注入
`launch_gate={"frames":0,"frac":0.05}`，不走自适应课程。

**阶段二**（出拳）：从准备姿态到出完拳。要求符合 clip 的
motion，且**后脚（发力脚）不能动**。终点同样以去踝 17 个关节
对 clip 落地帧的 max|q_err|≤0.10 判定，而非旧的指针/手误差
复合判据。

## 4. 奖励通道

全部为**负势场**形式（每步计算，越小越好，reward = −Φ）。
目标关节为去踝 17 关节的归一化姿态误差 e_j。

| 通道 | 定义 | 阶段 | AW |
|---|---|---|---|
| `r_strike_p1_mse` | −mean(e_j²) vs clip 帧0；首次达标帧额外 +1 | 一 | 4 |
| `r_strike_p1_var` | −var(e_j²)：防单关节大偏被均值摊薄 | 一 | 4 |
| `r_strike_p2_mse` | −mean(e_j²) vs **落地帧 k_max**；首次达标帧额外 +1 | 二 | 4 |
| `r_strike_p2_var` | −var(e_j²) vs 落地帧 | 二 | 4 |
| `r_strike_p2_foot` | −min(‖Δp_foot‖, 0.5m)：后脚世界系 3D 位移罚 | 二 | 8 |
| `r_potential` | +0.01·min(phi, 0.9)：稠密直立护栏 | 全程 | 1 |

护栏说明：phi≥0.9（正常打击姿态都在其上）时 `r_potential` 为
常数 bonus，不塑形动作；只在失衡时提供回拉梯度。显式 `r_fall`
已移除；打击通道平时为负势场，首次达标时 MSE 通道会额外
获得 +1。护栏只留稠密项。

**后脚规则**（不从 clip 几何推断）：heavy 是后手拳，拳侧脚在
后；jab 是前手拳，对侧脚在后。即 heavy_l → 左脚、heavy_r →
右脚、jab_l → 右脚、jab_r → 左脚。此规则仅在本实验的 observer
配置中启用，旧实验保持原逻辑。位移锚点记录在**过门那一帧**
的后脚位置（非过门后首步），3D 位移（抬脚/滑步都算）。

## 5. 阶段隔离：按 Trajectory 拆分

rollout 后，按每步去踝姿态最大归一化误差寻找首次达标帧，
以该帧**包含在当前轨迹尾部**的方式切成最多两条 Trajectory：

- **轨迹一 `[0,g+1)`**：首帧到首次贴近 clip 0 的第 g 帧，
  挂 p1 + `r_potential`；`r_strike_p1_mse[g] += 1`，p1 通道
  终止（不 bootstrap）。`r_potential` 在边界正常 bootstrap
  `V(obs[g+1])`，因为护栏继续跨阶段作用。
- **轨迹二 `[g+1,l+1)`**：过门下一帧到首次贴近 clip landing
  的第 l 帧；挂 p2 + `r_potential`；`r_strike_p2_mse[l] += 1`。
  到达 landing 时通道终止（不 bootstrap），`l` 后的帧丢弃。
- **未过门**：只保留整段 p1（无成功奖励）；已过门但未到达
  landing：p2 截至 episode 末尾，无成功奖励。
  两种失败都耗尽固定 30 步任务预算，相关通道 `is_terminated=True`，
  尾值为 0。不能把失败 episode 当无限继续的任务做 bootstrap。

GAE 分别按两条轨迹与各自 critic 计算；每段只挂所属阶段
的打击通道，**没有“另一个阶段 AW=0”的通道**。

实现审计与失败终止修正的证据见 `STRIKE_2PHASE_AUDIT.md`。

## 6. Job 构造与采样

- `build_jobs` 独立实现（不继承旧链）：`skills` + `move_schedule`
  + `warmup` + `launch_gate` + `warmup_policy_cls`，`initial_distance=2.0`。
- 双机器人独立抽技能；`SKILL_SAMPLE_W` 给镜像侧加权 ×2
  （jab_r/heavy_l=2.0）——历史数据显示镜像侧为弱侧。
- rollout：1536 ep/update，96 workers；eval 每 5 update、64 ep。

## 7. Eval 指标

| 指标 | 含义 |
|---|---|
| `{skill}_gate_qmax` | episode 内最小过门关节误差（判据 0.10） |
| `{skill}_ungated` / `gate_t` | 未过门率 / 过门耗时 |
| `{skill}_landed` / `land_qmax` | 5% 落地姿态达标率 / 最小关节误差 |
| `{skill}_land_t` | 旧指针推进帧位，仅作诊断，非终点判据 |
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

- 早期尝试 `train_strike_2phase_ppo_20261020_145535` 从
  `skill_2phase_mbs` u07200 恢复，**不作为本方案起点**。
- 本版从 `train_skill_full_mbs_ppo_20261020_104315` 的最后
  checkpoint `u06685` 起手；新打击 critic 冷启动。
- `train_strike_2phase_ppo_20261020_154336` 首次以默认 confidence
  启动：首个 update 的 p1 critic EV 为负、confidence=0，导致
  p1 虽有数据且 AW=4，却不参与 actor 更新；已停。
- 有效 run：`train_strike_2phase_ppo_20261020_154404` 从相同
  u06685 起手，附加 `--no-confidence`，避免冷启动 critic
  关闭打击通道的 actor 梯度；仍需监测严格 5% 过门率和 p2 数据。
