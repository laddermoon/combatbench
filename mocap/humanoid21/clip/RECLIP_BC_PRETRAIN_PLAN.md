# Reclip 批量专家数据 + BC 预训练实验计划

> 状态：计划草案，待执行  
> 日期：2026-10-22  
> 范围：`mocap/humanoid21/clip` 与其依赖的 Humanoid21 / CommandSimulator / PPO 组件

## 1. 实验目标

用冻结的技能策略生成大量真实动态初始状态，再用 Reclip 从每个初始状态生成到达指定打击末端状态的可执行 expert trajectory。离线训练一个只看 57 维本体观测的策略，使它近似这些 expert trajectories。最后用 held-out 初始状态做端到端闭环评测，并把可用策略作为后续 PPO fine-tune 的初始化。

整体流程：

```text
冻结 warmup policy + command + steps + seed
    -> Init CoreState dataset
    -> batch Reclip(init_state -> strike target pose)
    -> expert CoreClip dataset
    -> replay expert, collect (obs57, action21)
    -> offline BC
    -> held-out end-to-end eval
    -> export actor as PPO init
    -> RL fine-tune
```

该实验的动机不是让策略复刻某一条固定轨迹，而是学习一个“从真实 locomotion 初始状态接入一次打击”的近似策略。Reclip 提供密集 expert，BC 提供先验，RL 负责后续精调。

## 2. 已选定的目标动作

用户选定的目标动作来自：

- 视频：`mocap/humanoid21/clip/data/videos/cmu_144_20_m_seg01_pass_ep.mp4`
- 对应 CoreClip：`/tmp/reclip_work/path_wp7/cmu_144_20_m_seg01_pass_ep.npz`

该 CoreClip 的关键元数据：

- `clip_id`: `cmu_144_20_m_seg01_core_reclip`
- `objective`: `pass`
- `path.mode`: `endpoints`
- `status`: `completed`
- `T = 7` states
- `event_frame = 6`
- `event_time = 0.30s`
- `s_final = 1.0490`
- `terminal_pose_error = 0.3351`
- `terminal_vel_error = 0.1862`
- `fall_count = 0`
- `ref_action_semantics = executed_pd_target`

**实验目标状态默认取该 Reclip 输出的末帧 `state(6)`。**

语义约定：

- Reclip 求解目标使用 `pass`；
- 目标是“快速穿过目标状态”，不是停在目标状态；
- 末端速度不作为必须匹配的约束；
- `terminal_pose_error` 和 `terminal_vel_error` 只作为质量报告/筛选指标；
- 如果后续要训练“停在该姿态”，需要另开 `reach-stop` 变体。

**注意：** `/tmp/reclip_work/...` 是临时产物。正式实施前必须把该 NPZ 复制/冻结到项目数据目录，或用记录的参数重新生成。

## 3. 数据与策略契约

### 3.1 初始状态

- 由冻结的 `u06689` warmup policy 生成；
- 通过不同 command、执行步数和 seed 采样；
- 保存完整 `CoreState`，包括根位置、根姿态、根速度、关节位置和关节速度；
- 初始状态只用于生成 Reclip expert，不直接进入 policy 输入。

### 3.2 训练 policy 输入

- 严格使用 `obs[:57]` 本体观测；
- 不给 opponent observation；
- 不给 target / path / phase / time；
- 不给 warmup command；
- 目标技能隐式固定，由策略参数表示。

### 3.3 Policy 契约

- 训练和 rollout 输入采用 `obs57 + zeros(9)` 的 66 维形式；
- 实际信息只来自 `obs[:57]`，后 9 维恒为零；
- 选择该形式是为了后续直接接入现有 66 维 PPO actor / checkpoint / resume 管线；
- `obs[:57]` 之外不加入 target、path、phase、time 或 opponent 信息。

### 3.4 根位置语义

- 本实验的目标是固定“身体姿态”，不是固定世界坐标；
- `pass` 目标评估应先支持忽略绝对 `root_pos` 的模式；
- 根位置仍在输出 CoreState 中记录和报告，但不参与该实验的路径进度/姿态目标；
- 后续如果需要约束水平位移，可另加“只忽略 x/y、保留 z”或“init-frame relative root position”模式。

### 3.5 训练 policy 输出

- 21 维 normalized PD target action；
- 与 Reclip `ref_action` 语义一致；
- rollout 中由 simulator 执行 25 个 0.002s 物理子步，即 20Hz 控制。

### 3.6 双机器人处理

- 默认训练 `robot_a`；
- expert 生成与 policy 评估时，`robot_b` 固定为 canonical idle state；
- `robot_b` 每帧给零 action；
- 不把 `robot_b` 状态作为 policy 输入。

## 4. 任务清单

### T0. 冻结实验契约与目标 artifact

**关键点**

- 冻结目标 NPZ 到项目数据目录；
- 记录目标视频、目标 NPZ、target frame、目标 transform；
- 确定实验目录、命名、数据格式；
- 记录本计划中的默认决策。

**产出**

- `data/reclip_bc/<exp_name>/target_clip.npz`
- `data/reclip_bc/<exp_name>/experiment_spec.json`
- 更新后的实验 manifest。

---

### T0.5. Position-free 姿态目标

**关键点**

- 让 `pass` / `reach-stop` 的路径投影、路径误差和 terminal pose error 支持忽略绝对 `root_pos`；
- 第一版目标语义是“body pose only”，即只匹配 `root_rot + joint_pos_norm`；
- `root_pos` 仍保留在 TargetPath / CoreClip / metrics 中，便于诊断，但不进入该实验的 pose score；
- `track` 暂不使用 position-free，避免破坏“同步复刻”的语义；
- 优先复用现有 `TargetPath.pose_weights`：将 `rp` 权重置 0；
- 若需要保留根高度约束，再扩展为 `rp_xy` / `rp_z` 分项。

**产出**

- position-free pose weight / objective 配置；
- 单元测试覆盖 `pass` 与 `reach-stop`；
- CLI / metadata 中能看出该 run 是否忽略 root position；
- 一份小样本 Reclip 验证：即使根位置不同，投影和 event 也只由身体姿态推进。

---

### T1. 初始状态采样器

**关键点**

- 使用 `CommandSimulator` + 冻结 `u06689`；
- 随机采样 warmup skill、command、steps、seed；
- 参考 `WarmupSteppingPlugin`：内部 sim 跑 warmup，最后取 `get_core_state()`；
- 保存完整 CoreState 与 warmup metadata；
- 支持 shard、resume、固定 seed 复现；
- 默认只用 `robot_a` 作为 expert 起点；可选同时保存 `robot_b` 作为备用样本。

**产出**

- `init_states/shard_*.npz`
- `init_manifest.jsonl`
- `init_summary.json`：数量、root height、yaw、速度、接触分布等。

---

### T2. 目标姿态构造器

**关键点**

- 目标 body pose 固定取 `cmu_144_20_m_seg01_pass_ep` 的末帧；
- 使用两点路径 `init_i -> target`；
- 根位置不参与路径进度和姿态目标，由 T0.5 的 position-free 模式处理；
- 目标速度不进入路径目标；
- absolute / init-relative / opponent-relative root position 作为后续可选模式，不进入第一版。

**产出**

- `build_target_state(init_i, target_clip)` 实现；
- `target_spec.json`；
- target 构造单元测试。

---

### T3. Batch Reclip 求解器

**关键点**

- 对每个 `init_i` 构造 `TargetPath.from_pose_pair(init_i, target_i)`；
- 使用 `ReclipObjectiveConfig(kind="pass")`；
- `budget_frames` 是最大动作步数，不是固定时长；
- 使用 multiprocessing，每个 worker 持有独立 simulator；
- 每次 solve 前 reset simulator，避免 `other_state`/动作缓存污染；
- 保存完整输出 CoreClip、求解 metadata、failure；
- 每个输出执行独立 `ref_action` replay 验证。

**产出**

- `expert_clips/*.npz`
- `reclip_results.jsonl`
- `reclip_failures.jsonl`
- `solver_summary.json`

---

### T4. Expert 质量筛选

**关键点**

`status == completed` 只是最低条件。还需要按实验目标筛选：

- `event_frame` 存在；
- `fall_count == 0`；
- `ref_action` replay 误差接近 0；
- `terminal_pose_error` 在实验阈值内；
- `path_error_max` 在实验阈值内；
- 轨迹长度合理；
- 没有明显异常 action saturation。

阈值先不设死，由 pilot 分布决定。

**产出**

- `accepted_results.jsonl`
- `rejected_results.jsonl`
- `quality_summary.json`

---

### T5. BC 数据集构建

**关键点**

- 对每条 accepted expert clip，在 `CommandSimulator` 中完整 replay；
- 在 action 前取 `obs[:57]`，与 `ref_action[k]` 配对；
- 不逐帧 teleport 后取 obs，避免足底接触特征与真实 rollout 不一致；
- 只使用 `k=0..T-2` 的真实 transition；
- `robot_b` 固定 canonical idle；
- train/val 按 `init_id` / `traj_id` 划分，不按单帧随机切分。

**产出**

- `dataset/train.npz`
- `dataset/val.npz`
- `dataset_manifest.json`

数据字段至少包括：

```text
obs[57]
action[21]
traj_id
init_id
frame_idx
progress_s
time_to_go
solver_score
```

---

### T6. BC policy 训练

**关键点**

- 第一版使用 deterministic supervised learning；
- 优先训练 66 维 PPO-compatible actor，输入为 `obs57 + zeros(9)`：

```text
StateMixtureBoundedStdTruncatedNormalPolicy
obs_dim=66
action_dim=21
hidden_dim=256
num_components=3
```

- `deterministic_action(obs)` 对 `ref_action` 做 MSE/Huber；
- 不训练 critic；
- 不让 policy 看到任何非本体输入；
- deterministic eval；
- 保存网络结构与训练 lineage。

**产出**

- `models/bc_actor.pt`
- `models/train_config.json`
- `reports/train_metrics.json`

---

### T7. Teacher forcing 与 one-step 评估

**关键点**

分层报告：

1. `π(obs57[k])` 与 `ref_action[k]` 的误差；
2. 从 expert `state[k]` 执行 policy action，比较一步后 `state[k+1]`；
3. 分开统计 joint/root/velocity error；
4. 不把 action MSE 直接当作闭环成功指标。

**产出**

- `reports/teacher_forcing.json`
- `reports/one_step_eval.json`

---

### T8. Held-out 端到端闭环评测

**关键点**

- 用没参与训练的 init states；
- 从 `init_i` 开始执行 policy，直到 `s>=1`、fall 或预算耗尽；
- 使用同一 `TargetPath` 和 evaluator；
- deterministic policy；
- 保存 rollout CoreClip 和视频；
- 与同 init 的 expert 结果对比。

**指标**

- success rate；
- event frame；
- terminal pose error；
- terminal velocity error；
- path error mean/max；
- fall rate；
- action saturation；
- per warmup skill/steps 分层成功率。

**产出**

- `reports/closed_loop_eval.json`
- `eval_rollouts/*.npz`
- `videos/*.mp4`

---

### T9. Pilot 批次

**关键点**

- 第一版不直接跑 10000；
- 先生成 100–300 个 init states；
- 用轻量 solver 参数验证完整 pipeline；
- 分析 solver 成功率、失败原因、BC 闭环成功率；
- 再决定是否扩大规模或修改 target/path 语义。

**产出**

- `pilot_report.json`
- `pilot_summary.md`

---

### T10. 大规模 expert 生成

**关键点**

- pilot 合格后扩展到数千/10000 init states；
- shard + multiprocessing + resume；
- 失败 init 可重试更强 solver 配置；
- 记录每个 expert 的 lineage；
- 不静默丢弃 failure。

**产出**

- `expert_clips_full/`
- `reclip_results_full.jsonl`
- `dataset_full/`
- `scale_report.json`

---

### T11. BC actor 导出为 PPO 初始化

**关键点**

- 导出 66 维 actor state dict；
- policy 输入为 `obs57 + zeros(9)`，不额外使用 command 信息；
- critic 从零初始化；
- optimizer state 新建；
- 验证 `--resume-from` 能加载。

**产出**

- `exports/bc_actor_export/model.pt`
- `exports/bc_actor_export/policy.py`
- `exports/bc_actor_export/MANIFEST.json`
- `ppo_init/checkpoint_u00000.pt`

---

### T12. RL fine-tune 实验

**关键点**

- 新建 Reclip/BC fine-tune PPO experiment；
- reset distribution 使用 warmup init states 或在线 warmup；
- reward 基于 path progress、path error、event、terminal pose 和 fall；
- actor 从 BC checkpoint resume；
- critic 从零；
- 与 scratch PPO 做同配置对照。

**产出**

- `baseline/experiments_ppo/exp_reclip_bc_finetune.py`
- 对应 env blueprint
- `reports/bc_vs_rl_finetune.json`

## 5. 当前仍需决策的点

### 5.1 必须在实现前定下的决策

| ID | 决策 | 推荐默认 |
|---|---|---|
| D1 | 目标 CoreClip 放在哪里 | 把 `/tmp/.../cmu_144_20_m_seg01_pass_ep.npz` 冻结到 `mocap/humanoid21/clip/data/reclip_bc/target/` |
| D2 | target frame | 使用 `pass_ep` 输出末帧 `state(6)` |
| D3 | 目标根位置语义 | `position-free`：只匹配身体姿态，不匹配绝对 `root_pos` |
| D4 | 每个 init 的路径 | `init -> target` 的 `endpoints` 路径 |
| D5 | Reclip objective | `pass`，快速穿过目标，不要求停止 |
| D6 | 控制机器人 | `robot_a`；`robot_b` 固定 canonical idle |
| D7 | policy 输入 | `obs57 + zeros(9)`，只使用本体信息但保持 66 维契约 |
| D8 | policy 结构 | 66 维 PPO-compatible MBS actor |
| D9 | init 采样 policy | 冻结 `u06689`，只生成初始状态 |
| D10 | expert 数据标签 | Reclip `ref_action[k]`，只用 `k=0..T-2` |
| D11 | 评估成功定义 | `s>=1` 且不 fall；terminal pose error 是否进入成功判据待 D14 |

### 5.2 需要 pilot 后决定的决策

| ID | 决策 | 说明 |
|---|---|---|
| D12 | solver budget / sweeps / samples | 由 init 难度和吞吐决定；先用轻量配置 |
| D13 | expert 筛选阈值 | `terminal_pose_error`、`path_error_max` 等按 pilot 分布设阈值 |
| D14 | 闭环成功阈值 | `s>=1` 是否足够，还是加 terminal pose error 上限 |
| D15 | 是否保留失败/边缘 init | 决定 policy 是否要学恢复，还是只学可发起区域 |
| D16 | BC loss | MSE 或 Huber；是否需要 per-joint 权重 |
| D17 | 训练轮次和模型规模 | 由 pilot overfit/underfit 决定 |
| D18 | 数据加权 | 是否按 trajectory 等权，避免长轨迹主导 |
| D19 | RL reward 权重 | BC 可用后再定 progress/path/event/fall 权重 |
| D20 | RL reset 分布 | 用已保存 init dataset 还是在线 warmup 生成 |

### 5.3 后续才可能需要的决策

- 是否把 `endpoints` 换成 `transformed clip all path`，以保留原始出拳中间形态；
- 是否加入 phase/time/progress 输入；
- 是否使用 recurrent policy；
- 是否加入 DAgger / on-policy correction；
- 是否扩展到多个 strike targets；
- 是否加入 opponent-relative target（需要 opponent observation，当前 body-only 契约不支持）；
- 是否需要 `reach-stop` 或 `reach-state` 变体。

## 6. 主要风险

1. **批量 Reclip 吞吐**：10000 个 init 会产生大量后缀 rollout，必须 shard/parallel/resume。
2. **endpoint path 可能失去 punch 形态**：只要求到目标姿态时，学到的是“到点”，不一定是语义出拳。
3. **多模态 expert**：不同 init/seed 可能给出不同合理动作，deterministic MSE 可能学平均动作。
4. **观测因果一致性**：BC obs 必须通过 replay action 获得，不能只 teleport 到 CoreState。
5. **`pass` 不等于精确停在目标点**：它允许穿过，terminal pose error 需要单独筛选和评估。
6. **根位置语义**：position-free 解决绝对坐标问题，但也可能放过不合理的水平漂移；必要时再拆 `rp_xy` / `rp_z` 或加 init-relative 模式。
7. **失败初始状态分布**：warmup 采到的状态可能太远或太失衡，solver 成功率可能很低。
8. **66 维零命令契约**：`obs57 + zeros(9)` 便于接入 PPO，但必须保证后 9 维恒为零，不能泄漏 command/target 信息。

## 7. 最小验收标准

Pilot 完成后至少应有：

- 一批可复现 init states；
- 一批通过 replay 验证的 expert CoreClip；
- `train/val` BC dataset；
- 一个 `obs57 + zeros9 -> action21` actor；
- teacher forcing 和 one-step error 报告；
- held-out init states 的 closed-loop success/fall/path 指标；
- 明确是否值得扩到大规模以及是否适合进入 PPO fine-tune。
