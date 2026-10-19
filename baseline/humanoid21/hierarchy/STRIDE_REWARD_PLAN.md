# 按步结算的移动奖励方案（Stride-Settled Movement Reward）

> 状态：已实现并开训（2026-10-19）—— 用户审阅通过：
> 滑动 stride ✓ / ep 尾补结算（补）✓ / τ=0.1 ✓ / w_y=0.5 ✓
> 目标：替换阶段一技能训练中的移动奖励（`r_move`/`r_drift`），
> 其余通道不动；从 S2 checkpoint 重新训练。

## 1. 动机

现行移动奖励的问题（u1635_repro 已实证：方向不准、动作不像走）：

- `r_move = 2·v_along` 逐帧结算，v 由 **17 步滑窗**平滑后差分得到——
  评估窗口 ≈0.85s，一步迈歪了被摊平在滑窗里，奖励无法归因到
  "是哪一步出了问题"
- 滑窗内任何瞬时行为（晃动、先冲后飘、斜着蹭）只要均速够就照拿
  奖励，方向精度松
- 逐帧密集信号 + 长视野 → PPO 信用分配噪声大

新方案的核心：**以迈步周期为奖励原子**。一次完整迈步
（左脚一步 + 右脚一步 = 一个 stride）结束时，用该周期的
**净位移/净偏航**一次性结算奖励。迈步周期天然消掉步态摆动，
不需要滑窗；奖励视野从 0.85s 缩到一个 stride（~0.3-0.5s）。

## 2. 步界检测（离线，不动环境/observer）

数据已经现成：obs[52:54] = `feet_forces`（足底触地力，体重倍数，
逐帧录在轨迹 obs 里）。`build_trajectories` 离线切分：

```
foot_in_contact[f] = feet_forces[f] > τ_contact      (τ ≈ 0.1)
foot-strike 事件    = 接触态的上升沿（空中→触地）
stride 序列         = strikes s_1 < s_2 < ...（任意脚）
stride_i 窗口       = (s_i, s_{i+2}]                 # 两步一个周期，滑动
结算时刻            = s_{i+2}（第 i+2 次落地的那个 action step）
```

**滑动窗口**：每次落地都结算一次"最近两步"的净表现——每次落地都
有反馈（信号密度是逐帧的 ~1/16），且不用管"哪只脚先起步"。
代价：每一步的位移在相邻两个 stride 窗口里各计一次（等价于对
"每步奖励"做了一次窗长 2 的滑动平均——保留周期语义，密度翻倍）。
备选：非重叠 stride（每两次落地结算一次）——若稀疏信号不足可切。

## 3. 每个 stride 评估的三量（世界系，move_track 原始 xy/yaw）

```
Δxy   = xy(s_{i+2}) − xy(s_i)          # 净位移，无需平滑（周期消摆）
Δt    = (s_{i+2} − s_i) / 20           # 秒
v_rad = (Δxy · d_hat) / Δt             # 径向速度（沿指令方向）
v_off = |Δxy × d_hat| / Δt             # 切向速度绝对值（横向漂移）
Δyaw  = unwrap(yaw(s_{i+2}) − yaw(s_i))# 周期内净偏航
```

`d_hat` 语义不变：指令方向 × episode 起点 yaw。不用滑窗。

## 4. 通道改动（仅 move×4 episode）

| 旧 | 新 | 定义（stride 末单点结算） |
|---|---|---|
| `r_move` | `r_move_step` | `+2·v_rad·Δt`（旧稠密积分等效量） |
| `r_drift` | `r_drift_step` | `−v_off·Δt − 0.5·|Δyaw|`（净侧漂位移 + 每周期净偏航 0.5/rad） |

- **actor_weight**：结算帧乘 φ²(t_end)（落地站稳才算数），其余帧该
  通道 actor_weight=0
- **结算量纲对齐**：稀疏点值 = 旧稠密奖励在该 stride 上的积分
  （`r_move_step = 2·v_rad·Δt`，约 1.0/event），critic 回报尺度与旧
  通道相当，避免新增 critic 量级失衡
- **gamma 不变**（0.99）：只缩"评估窗口"，不动折扣因子
- turn（`r_turn`）、step（`r_hold`）、strike（`r_strike`/`r_std_clip`）、
  `r_potential`/`r_fall` 全部原样保留——本方案只动 move×4

## 5. 边界与退化情形

| 情形 | 处理 |
|---|---|
| 擦地拖行/不迈步（无 strike 事件） | episode 末兜底：用 ep 起点→终点整段窗口结算一次（拖行也能拿到"位移如何"的评价，但拿不到迈步红利——激励迈步） |
| 摔倒中 | φ² 门控零权重，同现状；恢复后 stride 检测照常 |
| episode 尾巴的半个 stride | 若距上次结算 ≥1 步且 ≤1 步未结算，ep 末按实际 Δt 补结算一次（待定，见 §7 开放点） |
| 同脚连落地（脚没交替） | 不检查左右脚交替——任一脚落地都算边界；拖行会反映为 v_rad 低+v_off 高，自然被惩罚 |
| warmup | 本实验为阶段一结构（无 warmup spec），步界只切 episode 帧 |

## 6. 实验结构

- 新实验 `skill_step_mbs`（`SkillTrainMBS` 子类，**只改 move ep 的
  build_trajectories + reward_channels + eval 指标**）
- 起点：`hier_s2_balance_s42i_b/checkpoint_u01635.pt`（与 repro 同，
  已验收的 S2 血统）
- 训练目标对齐阶段一：四向 move + 双 turn + step + 四 strike
- `skill_train_mbs`/`skill_trans_mbs`/`skill_precise_mbs` 零改动

## 7. eval 指标（新增）

每技能报：`strides_per_ep`、`v_rad_per_stride`、`d_off_per_stride`、
`|dyaw|_per_stride`、`no_strike_frac`（兜底占比——反映"不迈步"比例）。
旧指标全保留（v_along 等仍按旧公式算便于对比）。

## 8. 验收标准

- move v_along ≥ 旧终态（~0.95 m/s）且 v_off/dyaw 显著低于现状
- 视频定性：迈步动作是"走"而不是蹭/飘；方向与指令一致
- falls 不比旧终态恶化；step/turn/strike 指标无回归

## 9. 实施记录（2026-10-19）

实验 `skill_step_mbs`（`exp_skill_step_mbs.py`）：
`SkillTrainMBS` 子类，`_channel_names` 换 `r_move_step`/`r_drift_step`；
`_build_agent_trajectory` 调父类后把 move ep 的两个稠密通道 pop 掉、
注入按步结算通道（actor_weight = 旧 aw × settle 掩码——结算帧
3·φ²(t_end)，其余 0；零奖励 stride 仍带权重，"没进展"也是信号）。

验证：

- smoke：8 通道产出，`r_*_step` 在 move traj 上稀疏激活
- 真实策略实跑 move_fwd：feet_forces 0→3.3×体重振荡，strike 检测
  每 ~5-6 步/脚 → stride ~11 步（0.55s），70 步 ep ~14 次结算 ✓
- eval 指标：`{skill}_{strides,svrad,svoff,sdyaw,nostep}` 在线

**训练 run**：`train_skill_step_mbs_ppo_20261019_160546`（GPU0，
PID 1770374），从 `hier_s2_balance_s42i_b/checkpoint_u01635.pt`
（repro 同起点）续训。
