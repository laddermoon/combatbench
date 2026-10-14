# strike_sparse_step 实验思路大纲

> 本文件记录该实验的**设计约定**。任何对本大纲的修改都必须先向用户
> 提出具体改动提议，经确认后方可实施。

## 目标

不使用 attack_prep 的手臂塑形脚手架（交替出拳状态机），依靠稀疏的
终局事件奖励 + 大规模探索，让策略自己发现"接近 → 有效打击 → 收手"
的完整路径。

## 起点

`follow_s42b` 最终 checkpoint（`checkpoint_u08565.pt`），
`--reset-update` 重计数。follow 策略不会出拳——出拳能力完全由本
阶段的探索自行建立。

## 距离分区奖励（核心规则）

以 `D_ENGAGE`（默认 0.9m，与 `FOLLOW_DIST_MAX` 一致）为界：

### 界外 dist > D_ENGAGE —— Follow 原生奖励

`r_potential`、`r_left_foot`、`r_right_foot`、`r_torso`、`r_fall`、
`r_radial`、`r_tangential`、`r_face` 全部按 follow 原配方生效
（各自原有的 aw 与门控不变）。

### 界内 dist ≤ D_ENGAGE —— 任务奖励 + 防摔 + 对正

只保留：

| 通道 | 作用 |
|---|---|
| `r_strike` | 任务本身：SETTLE 首付 0.5Q + 收手尾款 0.5Q |
| `r_retract` | RETRACT 阶段势差塑形（拉开距离 + 手回身侧） |
| `r_fall` | 防摔惩罚（aw=2.0） |
| `r_potential` | 站立势位（保持起身/站直能力，视为防摔组） |
| `r_face` | 面向目标（区内满权重段） |

关闭：`r_left_foot`、`r_right_foot`、`r_torso`
（`r_radial`/`r_tangential` 的 out_zone 门控在 0.9m 内本就为 0，
与界限自然衔接——界内不存在任何"继续贴近"的塑形信号）。

### RETRACT 阶段（沿用 strike_step 规则）

命中收手阶段，界内外规则不再适用：除 `r_fall`、`r_strike`、
`r_retract` 外所有通道 reward=0 且 aw=0。

## 通道集合

`FollowStep` 8 通道 + `r_strike` + `r_retract` = 10 通道。
attack_prep 的 4 个手臂塑形通道完全不存在（无对应 critic）。

## 探索参数

| 参数 | 值 | 说明 |
|---|---|---|
| `episodes_per_update` | 1024 | ×2 规模 |
| rollout `explore_factor` | 函数化 | 见下 |

### 函数化 explore_factor（σ 倍率 = 3^ef）

| 状态 | ef | σ 倍率 | 意图 |
|---|---|---|---|
| 倒地（h<1.1m） | −0.631 | ×0.5 | 保护起身技能 |
| 站立且 dist ≤ D_ENGAGE | +0.631 | ×2.0 | 在稀疏奖励区加强探索 |
| 站立且 dist > D_ENGAGE | 0 | ×1.0 | 界外不加强探索 |

`uncertainty_floor` 回到继承默认 0.4（函数化 ef 上岗后，全局地板会
让界外也被迫高探索，与本设计意图冲突）。

## 已做的解释性决策（可否决）

1. `r_potential` 归入"防摔奖励"在界内保留——它同时驱动起身，倒地
   发生在界内时仍能拉起（罕见情况）。
2. 分区为硬边界（aw 直接清零）；若观察到边界抖动/徘徊问题，可提
   议改为软过渡带（如 0.9→0.7m 线性衰减）。
3. `D_ENGAGE = 0.9m` 先取 FOLLOW_DIST_MAX 对齐值，具体数值待训练
   观察后再调。
4. 函数化 ef 替代全局 `uncertainty_floor` 提升（floor 回到 0.4）—
   探索强度改由距离门控的 σ 倍率承担，界外不加强探索。

## 实验对照

| | s42e | sparse_s42b（本实验） |
|---|---|---|
| 起点 | follow u08565 | follow u08565 |
| 手臂塑形 | attack_prep 状态机全套 | 无 |
| 界内奖励 | 全通道生效 | 仅任务+防摔+face |
| 探索 | 512 ep, floor 0.4 | 1024 ep, floor 0.6 |

## 观察指标

- eval `hit`（主动出拳是否学出来——核心验证）
- `strike_done` / `retract_timeout` / `timeout` 分布
- eval `Q`（打击质量）与 `t2hit`（接近效率）
- `r_strike` critic EV/conf（稀疏信号的可学习性）
- 界内行为视频：接近后是否主动发起攻击而非贴住目标
