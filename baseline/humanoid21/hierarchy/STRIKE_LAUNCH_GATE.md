# 打击技能：发力准备姿态门槛（Launch-Pose Gate）

> 状态：已实现并开训（2026-10-19）—— θ 经实测分布定稿 1.65

## 0. θ 实测依据（trans u10135 导出，jab_l 对 launch 窗 [0,4]）

| 起手 | score vs launch 峰值 | 说明 |
|---|---|---|
| 站姿（无 warmup） | **1.80** | 基准"已就位"水位 |
| heavy_r 热身 20 步 | 1.72 | 重拳伸右臂，左刺拳手仍在架 |
| move_fwd 热身 35 步 | 1.69 | 需 1-2 步站稳 |
| turn 热身 30 步 | **1.63** | 转身中起手最不稳——**拒** |

θ=1.65：站姿/移动起手 1-3 步过门；转身中未稳姿态被拒。
诊断方法：launch_gate θ=99（永不过门）→ reward 流 = 对齐分轨迹。
> 背景：泛化起手（warmup）下，单调进度指针"每步追 2 帧迁就机器人"，
> 允许从 clip 中后段直接计分——机器人从未学过"回到发力准备姿态"，
> 打击技能不可重入（连招、被反击后再出手都缺这块）。

## 1. 设计

把每个 strike segment 显式分两段：

```
段 A 对齐段（gate 未过）：
  每步对 launch 窗口 [0, k_launch] 的 clip 帧打分（取最高分帧），
  reward = 该分（稠密 shaping，鼓励快速回架）
  过门条件：最高分 ≥ θ_gate（严格——"基本到了初始状态"）
  过门帧记 gate_t；k_ptr 从对齐帧起步

段 B 执行段（gate 已过）：
  现有单调指针跟踪逻辑完全不变（[k, k+lookahead] 最高分前进、
  到 k_max 后每步 hold bonus）
```

- **in_strike=1 贯穿段 A+B**——r_strike 的 actor weight 在对齐段也
  有梯度，否则"回架"动作没有学习信号
- **对齐步数计入 episode 预算**：慢回架的代价自然压到 land_t 上，
  不加第二本账
- **只作用于 ext_only 路径**；非 ext 的旧路径不变
- **纯站立起步几乎零成本**：guard ≈ frame 0，1-2 步过门——不破坏
  阶段一已验收行为

## 2. 门槛判据

复用 `_frame_score` 复合分（满分 ~2.0）。θ 从严：
先离线测"正常站姿 vs frame 0"的分（预期 ~1.6-1.8）和"泛化起手"
分布，θ 定在两者之间偏严（如正常分布的 p25）。
启动窗口 `k_launch≈4`（~0.2s clip 时间——允许对"准备姿态带"内
任一帧对齐，不死钉第 0 帧）。

## 3. 接口

`episode_options["launch_gate"] = {"score": θ, "frames": k}`
——observer 在 `on_pre_episode` 读（同 `strike_clip` 走线）；
无此 option 时行为完全不变（零侵入：train/trans/step/precise
各实验照旧）。

诊断字段：`gated`（0/1）、`gate_t`（过门步，未过=段长）。

## 4. 实验

`skill_gate_mbs`（`SkillTransMBS` 子类——要 warmup 泛化起手才
有对齐问题可学）：
- build_jobs 后处理：给所有 episode_options 注入 launch_gate
  规格（train + eval 都注）
- 起点：`skill_trans_s2b_u9860b` 最新 checkpoint
- eval 增报 `gate_t`、`ungated_frac`（没过门的 strike ep 占比）
- 首版 θ 由诊断 rollout 实测分布确定

## 5. 风险

- **预算挤压**：泛化起手回架花 5-10 步 → 出拳余量变薄，
  land_t 可能退化；先观察，必要时 n+8~10 或对齐段免计
- **θ 校准**：过紧 → ungated_frac 高、strike 奖励断流；
  过松 → 形同虚设。首版 θ=1.65（见 §0）
- **过门语义**：段 A 没有"回架时限"硬截止——episode 结束仍未
  过门则全程只有对齐分（本身就是一个干净的失败信号）

## 6. 实施记录（2026-10-19）

- `strike_track_observer.py`：`episode_options["launch_gate"]`
  `{"score": θ, "frames": k}`；seg 级 `gated`/`gate_t` 状态；
  段 A 输出对齐分（in_strike=1 保持梯度），过门后走原指针逻辑；
  无 option 行为不变
- `exp_skill_gate_mbs.py`：`SkillTransMBS` 子类，build_jobs 给
  train+eval 全注入 gate spec；eval 增报 `{skill}_gate_t` /
  `{skill}_ungated`
- **run**：`train_skill_gate_mbs_ppo_20261019_164055`（GPU3，
  PID 765981），从 `hier_s2_balance_s42i_b/checkpoint_u01635.pt`
  （S2 起点，与 repro/step_mbs 同血统）——首个 run
  （163335，trans ckpt 起点）为错误起点，已停
- 观察重点：`ungated_frac`（健康水位应 <20%）、`gate_t`（应 <5）、
  land_t 是否被预算挤压、以及泛化起手打击的"回架"动作出现与否
