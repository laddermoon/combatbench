# 手臂软劫持验证实验（V1）— 最小化臂令跟随

> 定位：`v2_s3_armcmd_r3`（D17 奖励制臂令）的最小化变体——
> 剥掉推力扰动与最后一处门控暗示，只验证一个问题：
> **奖励制软指令能否让策略学会"站稳跟随、失衡忽略"？**

## 1 与 r3 的差异（唯一变量：扰动 + flag 语义）

| 项 | r3 | 本实验 |
|---|---|---|
| 扰动 | 40N×31-40f 躯干推（单级） | **无**（推力插件移除） |
| 臂令起始 | reset 即流动（已是事实） | 同 |
| flag 语义 | 随站立状态跳变（信息位） | **恒 1**（令恒激活） |
| err 计分 | 每帧无条件 | 同（无改动） |
| 指令流 | ArmScript U(2,20)hold/U(-1,1)tgt/rel 0.25 | 同 |
| 通道 | 5 通道（下表） | 同 |
| warm | u1680（经 r1/r2 接力） | **u1680 直起**（干净起点，与 stepcmd_r1 同线） |

**"从开始就发"的现状**：脚本 reset 即跑、err 每帧计分——
臂令在 r3 本来就无门控。真正要清理的是 obs[57]=flag：它仍随
站立状态跳变，是"令是否激活"的最后一处暗示。`flag_settle: 0`
（timer≥0 恒真）使 flag 恒 1——obs 不再对令的存活性说谎，
策略对令的态度完全由奖励权衡自决。零代码改动。

## 2 语义（同 D17，无改动）

- arm_tgt 写 obs[58:64]，act[15:21] 不覆写归策略全权
- r_arm = −‖act−tgt‖ 逐帧稠密——"可忽略指令"：站稳跟随划算，
  失衡时策略自行牺牲跟随（aw 权衡天然表达优先级）
- 释放彩票 0.25：每臂换目标时 25% 概率本周期无令（tgt 记 0、
  err 不计分）——指令本就可忽略，释放只是"显式无令"标记

## 3 通道（沿用 r3 五通道）

| 通道 | 定义 | aw |
|---|---|---|
| r_potential | 0.01×φ（**无 h_cap**——保持 r3 变量单一） | 3.0 |
| r_fall | 摔倒onset罚 + 靠墙罚（受推clean赏随推力移除自然缺席） | 2.0 |
| r_std_clip | clip+pose+0.5cad+0.5h_term | 2φ²（无推→not_pushing恒真→恒2φ²） |
| r_arm_r | −‖act[15:18]−tgt[0:3]‖ | 0.5 |
| r_arm_l | −‖act[18:21]−tgt[3:6]‖ | 0.5 |

## 4 已知风险

- **std_pose 含臂关节**（_POSE_IDX=q 0..20）——r_std_clip 的
  pose 项把臂拉向 s42i 姿态，与臂令结构性冲突。r3 已承受且
  err 照降（−50%/−66%）；V1 **照旧**保持变量单一，若 err 在
  无扰动下仍卡高位再考虑 pose 剔除臂维。
- RandomFallenStatePlugin 保留——倒地起点恰是"学会忽略"的
  训练素材（失衡帧 err 合理牺牲 vs 站稳帧跟随）。

## 5 Eval 与看板

- 延续 r3 字段（arm_err_r/l、arm_rel、falls、max_pot、clip/cad），
  recovery/pushes/force 随推力移除失效（自动读 0，不炸）
- **基线**：无感水位 ~1.3（零输出 0.96，无关 1.32）；r3 终点
  r=0.66/l=0.41 为参照
- **成功**：err 显著低于 1.3 且 falls 不升 → 软劫持可行
- **新增分解（核心证据）**：err 按帧势位分组——
  `err_stand`(φ≥0.9) vs `err_down`(φ<0.5)。失衡帧 err 显著
  高于站稳帧 = 策略"选择性忽略"的直接证据；两组同步下降
  = 无脑跟随（没学会忽略）。
- **err 地板**：失衡帧的合理牺牲使 err>0 永续——<0.3 看板线
  需结合分解读，不苛求全局值。

## 6 实现清单（极小 diff）

1. `v2_armsoft_env.yaml` = v2_balance_armcmd_env.yaml
   − StandingTriggeredForcePlugin − push_state_a/b
   + `flag_settle: 0`
2. `exp_v2_armsoft.py`：`V2ArmSoft(V2S3ArmCmd)` 仅换 `_env_pb`
   + on_eval 追加 err_stand/err_down 分解字段
3. warm-start：S2 u1680 + --reset-update（r_std_clip critic
   承袭；r_arm_r/l 两个新 critic 冷启）

命名：`v2_armsoft_r1`（GPU 空闲位）。
