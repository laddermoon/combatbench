# STEPARM V1 — 步令 × 双臂独立软劫持 联合实验方案

> 状态：**已按本会话决策定稿**
> 实验名：`v2_steparm`，首轮 `v2_steparm_r1`

## 0. 一句话

在 V1 步令实验（D20 残差步令跟随）基础上叠加 ARMDUAL 逐臂
独立软劫持机制与奖励通道，以 stepcmd 最新 checkpoint 热启动，
验证策略**同时满足步令与臂令**的能力。

## 1. 动机与选型依据

- 臂劫持最终形态已定：**逐臂独立劫持**（ARMDUAL 线证实：
  err_stand 0.134 仅落后捆绑版 0.03，且自由臂保留活动度可作
  平衡器、零平衡代价）。
- stepcmd 线已学会残差步令跟随（arr 0.82，d_mv 0.106）且
  学会了"失衡忽略令"——两臂指令复用同一 phi 门涌现语义，
  联合任务期望两个模态各自成立。

## 2. 任务定义

两个独立指令流并行驱动：

| 子系统 | 机制 | 周期 | 计分 |
|---|---|---|---|
| 步令（继承） | V2StepCmdSimulator：D20 残差编码，obs[67:79] | 0.5s/10 帧固定 | 逐帧积分残差 |
| 臂令（新增） | ArmDualScript：同步周期边界，P(0.1) 双臂 / P(0.9) 单臂(50/50)，dur U(2,20)f，恒 ≥1 臂在令 | 与步令独立 | 被劫帧 −err |

- obs 指令段合一布局（两实验原设计即兼容，无形状冲突）：
  `obs[57:65]=[flag_r,tgt_r,flag_l,tgt_l]`，`obs[65:67]`腰预留，
  `obs[67:79]`步令 cur/next 残差。
- flag=0 臂 tgt 全零（"不存在目标"）+ err 不计分——同 ARMDUAL。
- 无推力、倒地起点插件保留、无站立门控（帧 0 施令）——
  两模态的"失衡忽略"均由策略自学（涌现，不硬编）。
- 软劫持：act[15:21] 全归策略，无覆写。

## 3. 奖励通道（7，全 gamma 0.99）

| 通道 | 内容 | aw |
|---|---|---|
| r_potential | 0.01·φ（h_cap 1.10 → φ≤0.889） | 3.0 |
| r_fall | 摔倒 onset + 持续靠墙 | 2.0 |
| r_step_mv | −‖Δx,Δy‖（obs 列，post-state 对齐） | 1.0 |
| r_step_st | −max(0, res_st−0.03)（保留：虽 EV=0 仍作滑移监控） | 0.5 |
| r_step_yaw | −max(0, \|Δψ\|−0.05) | 0.5 |
| r_arm_r | −‖act[15:18]−tgt_r‖（被劫帧） | **0.7** |
| r_arm_l | −‖act[18:21]−tgt_l‖（被劫帧） | **0.7** |

臂总权重 1.4 = 步总权重 2.0 的 70%（用户定稿）。

## 4. Eval 规则（并集 + 一个交叉指标）

步令侧照旧：arr_rate/d_mv/d_st/d_yaw/hold/skate/quality/
falls/max_pot/success。
臂令侧照搬 ARMDUAL：arm_err_r/l（被劫帧条件均值）、
err_stand/err_down（φ 分解于被劫帧；h_cap→φ_max≈0.889，
**站稳门 0.85**）、occ_dual/r/l。
**交叉指标（新）**：步令周期内平均被劫臂数分桶——
`arr_1arm` vs `arr_2arm`（恒 ≥1 臂被劫故无 free 桶）：
回答"双臂同劫是否更拖累迈步"。
早停关闭，人工盯（V1 惯例）。

## 5. 配置

- env：`v2_steparm_env.yaml` = stepcmd env 换 sim 类
  （V2StepArmSimulator）+ arm_dual_a/b observers +
  arm_hold_range/arm_tgt_range/dual_prob 配置项。
- **warm-start**：`v2_stepcmd_r1` 最新 trainer checkpoint +
  `--reset-update`；r_arm_r/l 两 critic 自动冷启（不在 ckpt）。
- 网络 79 维 obs 不变（arm 段由"全零死维"变活跃——
  观测分布迁移，无形状变化）。
- GPU1（armsoft 停机释放）。

## 6. 看板（健康判据 + 预期瞬态）

- **预期瞬态（非 bug）**：臂令段从全零变活跃 + 两 critic 冷启
  → 早期 arr/d_mv 回落、arm_err 从 ~1.3 起步——域迁移瞬态；
  若 ~u200 后步令指标不回升才需归因。
- err_stand/arm_err 应向 armdual 水位（~0.13-0.19）收敛；
  err_down 应钉 ~1.6-1.7（忽略语义跨模态复现）。
- arr_1arm vs arr_2arm：若裂口 >0.15 说明双臂同劫显著拖累
  迈步 → 需调权重或降 dual_prob（决策点，跑出数据再议）。
- occ 应贴 0.1/0.45/0.45（调度器回归测试）。
- falls 应 ≤0.05：联合任务不显著恶化平衡。

## 7. 实现清单

| 件 | 内容 |
|---|---|
| `v2_steparm_simulator.py` | `V2StepArmSimulator(V2StepCmdSimulator)`：加每机 ArmDualScript（_dst）、set_action 先结 err 再步进脚本、obs 写 seg[0:8]、`arm_dual` derived state |
| `v2_steparm_env.yaml` | stepcmd env + arm 参数 + arm_dual_a/b observers |
| `exp_v2_steparm.py` | `V2StepArm(V2StepCmd)`：7 通道装配 + 臂 eval 扩展 + arr_hij/free 交叉指标 |
| 复用 | `arm_dual_script.py` / `arm_dual_observer.py` / stepcmd 全套（obs 布局本就兼容） |
| 既有文件 | 零改动 |
