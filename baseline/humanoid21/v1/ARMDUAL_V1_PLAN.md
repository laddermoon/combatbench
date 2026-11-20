# 双臂独立软劫持实验（ARMDUAL V1）

> 动机：旧臂令方案两臂捆绑同令——一臂执行指令时另一臂同被
> 劫持，失去"平衡器"角色。**逐臂独立劫持**：一臂工作、
> 一臂自由（可作平衡器）。
> 参照实验 = v2_armsoft_r1（软劫持+捆绑臂）；本实验不动其
> 任何文件（全新 sim/env/exp）。

## 1 观测格式（obs 指令段，总量 22 维不变）

腰段让 1 位给臂：

```
obs[57]     = flag_r        obs[61]     = flag_l
obs[58:61]  = tgt_r (3)     obs[62:65]  = tgt_l (3)
obs[65:67]  = waist 预留(2)  obs[67:79]  = step_*（不变）
```

- **flag=1**：该臂有令（软劫持=err 计分；预留硬接管=覆写，
  同语义开关 arm_override）
- **flag=0**：该臂无令——**tgt 字段全零**（"不存在目标"），
  err 不计分，策略自由输出
- 旧单 flag 的"站立 settle"语义不再存在：flag 纯粹=
  "本臂此刻是否有令"，帧 0 即生效（armsoft 无门控先例）

## 2 指令调度（同步同相位）

两臂共享同一周期时钟——每周期到期**同时**重采，不存在
一臂中途换令另一臂延续（引入相位仅为简化问题：保证
任意时刻恰好约一臂在令）。

```
每周期采样（双臂同步边界）：
  ① P=0.1  → 双臂同劫（两臂各采独立随机目标）
  ② P=0.9  → 单臂：P(左)=P(右)=0.5，选中臂采目标
  ③ 时长 ~ U(2,20) 帧（沿用 arm_hold_range）
  ④ 被劫臂 tgt ~ U(-1,1)³；自由臂 flag=0、tgt=0
```

- 时间占比：≈10% 双臂 / ≈45% 仅右 / ≈45% 仅左；
  **恒有 ≥1 臂在令**（无双臂全释放期）
- release_prob 机制废弃——自由臂即天然"释放"

## 3 奖励（4 通道，同 armsoft）

| 通道 | 定义 | aw |
|---|---|---|
| r_potential | 0.01×φ（无 h_cap） | 3.0 |
| r_fall | 摔倒onset罚 + 靠墙罚 | 2.0 |
| r_arm_r | −‖act[15:18]−tgt_r‖ × flag_r | 0.5 |
| r_arm_l | −‖act[18:21]−tgt_l‖ × flag_l | 0.5 |

被劫臂计复现误差；自由臂忽略（err=0）。软语义：
"可忽略指令"——失衡时策略自行牺牲跟随保命。

## 4 其余沿用 armsoft 最小化件

- 无推力插件；RandomFallenStatePlugin 保留（失衡忽略
  的训练素材）
- r_std_clip 移除 + standard_clip observers 移除
- PPO 超参沿用 S2 系（lr 3e-5 / kl 0.03 / 512ep×400f）

## 5 Eval 与看板

- arm_err_r/l = **被劫帧条件均值**（sim 上报 hij_r/hij_l
  per-frame 标记，eval 归一化分母=被劫帧数，非全帧）
- 调度健康度：occ_dual / occ_r / occ_l ≈ 0.1/0.45/0.45
- **核心证据**（同 armsoft）：err_stand(φ≥0.9) vs
  err_down(φ<0.5)——站稳跟随 vs 失衡忽略；裂口=学会忽略
- **新看点**：自由臂是否被用为平衡器（先定性看曲线/
  视频，不急着量化）
- 基线：无感 err ~1.3；armsoft_r1 终点 err_stand 0.29 为
  参照上界（双臂同时劫持 vs 单臂——跟随应更容易）

## 6 训练配置

- warm-start：S2 `checkpoint_u01680.pt`，`--reset-update`
  （r_std_clip critic 自动丢弃；r_arm_r/l 冷启）
- obs 布局变化：左臂 tgt 从 [61:64] 挪至 [62:65]，右臂
  [58:61] 原位——S2 暖机下 cmd 段本就是死维，语义重排
  无包袱

## 7 实现清单（全新文件，不动 armsoft）

1. `arm_dual_script.py` — 双 scheduler（周期时钟+模式/
   选臂/时长/目标采样，`flag/tgt` 输出，自由臂全零）
2. `v2_armdual_simulator.py` — `V2ArmDualSimulator(V2Simulator)`：
   父类臂机制关闭（arm_script=False），seg[57:65] 新布局
   自写；set_action 计 err（被劫臂）；derived_state 扩
   "arm_dual"（err_r/err_l/hij_r/hij_l/flag_r/flag_l）
3. `arm_dual_observer.py` — 透传（复用 arm_follow 模式）
4. `v2_armdual_env.yaml` — armsoft env 壳 + 新 sim
5. `exp_v2_armdual.py` — `V2ArmDual(V2S3ArmCmd)`：4 通道
   + err 条件归一 eval + 占用率诊断

命名：`v2_armdual_r1`（GPU2/3 空闲位）。

## 8 已知风险

- **单臂目标的平衡代价**：被劫臂全力跟令时只剩一臂可作
  平衡器——这正是实验问题本身；若 falls 上升说明自由度
  代价显著，需降 arm_aw 或放宽 tgt 范围
- **flag=0 全零语义**：自由臂 tgt=0 与"目标恰好为零向量"
  不可分——但 flag 位已消歧，无实际风险
