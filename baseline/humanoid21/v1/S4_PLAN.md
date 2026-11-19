# S4 步令跟随训练方案

> 状态：实施中 ｜ 关联：D12–D15（步令编码/几何/边界）、`STEP_GAIT.md`（姿态集）
>
> 一句话：在 S3 底座（`v2_s3_balance_arm_r3/checkpoint_u02315.pt`）上，
> 教策略**执行声明式步令**——每条指令指定"哪只脚→什么终态姿态·
> 转不转向"，机器人靠自己的运动能力把它走出来。

## 1. 指令与观测（obs 79 维不变）

57 本体段沿用。22 维指令段布局（D8）：

```
flag(1) | arm_tgt(6) | waist_tgt(3) | step_cur(6) | step_next(6)
```

- `step_cur` = 当前在飞指令 `[脚, 转, 前脚, 纵档, 横宽, age]`
  ——dim5=**步龄**：发令 0.0 起、每 action step +0.1、轮转归零
  （同脚同姿态连发消歧，D15）
- `step_next` = 下一条预览（dim5=0 不用）
- 步令语义（D13）：`脚→终态·转`，脚维无 0，终态 = 10 姿态之一
- 观测承诺=实际施加（D6 同源）：指令写入/轮转只在 `set_action`
  推进点发生，任意时刻 get_observation 无副作用

## 2. 指令流状态机（V2StepSimulator，每机独立）

```
reset:  随机姿态起链 → cur=sample → next=sample（age=0）
每 action step（set_action 推进点）:
    age += 0.1
    相位机吃 feet_forces（obs 同源那份）
    ├─ 边界/超时 → 量测结算 → step_report 落袋
    │            → cur←next, next=sample(collision_fn)
    └─ 否则继续等
collision_fn: apply_order 预测动脚落点 vs 24 边形墙 margin 0.3m
            （v0 不做对手碰撞）
timeout: age ≥ 6.0（60 帧）→ 判负结算并轮转（防指令流冻结）
```

摔倒/挣扎期间**指令流照走**（不挂起）——"不稳时可弃令"靠权重
表达，不靠环境关闸。

## 3. 步边界检测（step_boundary.py，纯 obs 函数）

训练/部署同源：输入 = obs 的 feet_forces（体重归一化）+ 内部
小状态，输出"本帧是否一步终点 + 执行脚"。每令一周期两相位：

```
相位A 候抬（选举摆动脚）:
    脚 X 连续 2 帧 force < 0.1 → X 当选 swing，转相位B
    （同帧双脚齐抬 → 指令脚优先）
    age≥6.0 → 超时（双脚未抬）
相位B 摆动:
    force[swing] > 0.4 → 落地边界（executed=swing）
    age≥6.0 → 超时（swing 悬空未落）
```

要点：脚必须**先被观测到卸力**才有落地权——已承重的指令脚不能
秒结（堵自环白捡/旧落地延续两个洞）；错脚先抬则错脚当选，
其落地照样结算判错脚；<2 帧点触漏判可接受。

## 4. 奖励（4 新通道，全部边界帧稀疏结算）

发令时刻存锚（双脚位+yaw），目标由 `apply_order` 预计算；
边界帧量测（脚位+yaw）对比结算：

| 通道 | 值 | 语义 |
|---|---|---|
| `r_step_foot` | 1 / 0 | 迈对脚：executed==order.foot（超时=0） |
| `r_step_mv` | ≤0 | −‖执行脚落点 − target_mv_foot‖（m） |
| `r_step_st` | ≤0 | −‖另一脚 − 发令原位‖（m，stance 滑移） |
| `r_step_yaw` | ≤0 | −\|wrap(yaw − target_yaw)\|（rad） |

保留 `r_potential`（aw 3.0）、`r_fall`（aw 2.0）；**删 r_std_clip**
（s42i 模仿与指令步态打架）。摔倒期步令奖励照发——靠权重让
r_fall/r_potential 主导，策略自己学会"先站稳再听令"。

## 5. env / 实验配置

- `v2_step_env.yaml`：V2StepSimulator + RandomFallenStatePlugin +
  TimeoutPlugin；**移除** StandingTriggeredForcePlugin、
  standard_clip ×2、push_state ×2；保留 standing_balance、
  foot_state、wall_contact；新增 step_report ×2
- `exp_v2_s4_step.py`（继承 V2S3BalanceArm）：
  通道 `r_potential`/`r_fall`/`r_step_foot`/`r_step_mv`/`r_step_st`/`r_step_yaw`；
  aw 初值 foot≈1.5、mv/st/yaw≈2.0（训练中看 adv 分布调）；
  obs_dim=79 → actor/blueprint 直接 warm-start，新通道 critic 冷启；
  `_no_improvement_limit` 继承关闭；eval quality =
  `2·foot率 − mean_mv − 0.5·mean_st − 0.3·mean_yaw`（暂定）
- 启动：`--resume-from u2315 --reset-update --run-name v2_s4_step`

## 6. 待校准参数

| 参数 | 初值 |
|---|---|
| lift 阈值 / 去抖 | 0.1 ×BW / 2 帧 |
| land 阈值 | 0.4 ×BW |
| 超时 age | 6.0（60 帧） |
| 墙 margin | 0.3 m |
| fit 权重 | w_mv=w_st=1.0、m⁻¹；w_ψ=0.3 rad⁻¹ |
| 脚位参考点 | foot body xpos 的 xy（全链路一致） |

## 7. 已知接受边界

- 双脚同摆竞态：swing 当选后另一脚先落 → 该步被忽略，罕见
- <lift 去抖帧的点触步漏判
- 目标 yaw 相对发令时刻，多步可漂移
- 调度器两遗留：|Δs|=4 飞步（低权稀有）、方向无持续性（漫游）
- stance 滑移项与动脚误差轻微重复计（w_st 小权）
