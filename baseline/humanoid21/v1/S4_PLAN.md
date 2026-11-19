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

- `step_cur` = 当前在飞指令 `[脚, 转, 前脚, 纵档, 横宽, age/H]`
  ——dim5=**归一化时限进度**：发令 0 起、到期帧达 1.0、轮转归零
  （同脚同姿态连发消歧 + 时间压力，D16）
- `step_next` = 下一条预览（dim5=0 不用）
- 步令语义（D13/D16）：`脚→终态·转`，脚维无 0，终态 = 10 姿态
  之一；**令=目标构型+时限**，到期须处于目标态，过程不测量
- 观测承诺=实际施加（D6 同源）：指令写入/轮转只在 `set_action`
  推进点发生，任意时刻 get_observation 无副作用

## 2. 指令流状态机（V2StepSimulator，每机独立）

```
reset:  随机姿态起链 → cur=sample → next=sample
发令时: 存锚(双脚位+yaw) → apply_order 算目标
        → 按动脚行程算时限预算 H（下节）
每个 action step（physical_step 末子步，obs 帧对齐点）:
    age_f += 1
    age_f ≥ H → 到期结算 → step_report 落袋
              → cur←next, next=sample(collision_fn)
collision_fn: apply_order 预测动脚落点 vs 24 边形墙 margin 0.3m
            （v0 不做对手碰撞）
```

摔倒/挣扎期间**指令流照走**（不挂起）——"不稳时可弃令"靠权重
表达，不靠环境关闸。

## 3. 时限预算（D16，取代相位机）

```
s_k = ‖指令脚位(发令时刻) − 目标脚位‖    # 水平行程
H_k = clip(h_min + (h_max−h_min)·s_k/s_max, h_min, h_max)

h_min = 0.15s → 3 帧（不动令下限）
h_max = 0.50s → 10 帧（满行程上限）
s_max = 0.53m（=2×STAGGER_UNIT）
```

部署 parity：指令源发令时拿同一对（动脚位, 目标位）算同一 H，
之后纯数帧——无检测项，训/部天然同帧。`step_boundary.py` 删除。

## 4. 奖励（4 新通道，全部到期帧稀疏结算）

发令时刻存锚（双脚位+yaw），目标由 `apply_order` 预计算；
到期帧量测（脚位+yaw）对比结算：

| 通道 | 值 | 语义 |
|---|---|---|
| `r_step_foot` | 1 / 0 | 到位：d_mv < 0.10m |
| `r_step_mv` | ≤0 | −‖指令脚位 − target_mv_foot‖（m） |
| `r_step_st` | ≤0 | −‖stance脚 − 发令原位‖（m，滑移） |
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
| 时限 h_min / h_max | 0.15s / 0.50s（3~10 帧） |
| 行程基准 s_max | 0.53 m（2×STAGGER_UNIT） |
| 到位判定 foot_tol | 0.10 m |
| 墙 margin | 0.3 m |
| fit 权重 | w_mv=w_st=1.0、m⁻¹；w_ψ=0.3 rad⁻¹ |
| 脚位参考点 | foot body xpos 的 xy（全链路一致） |

## 7. 已知接受边界

- 目标≈当前态的令免费得分（状态本已达成，语义诚实）
- 到期帧快照：提前到位又预摆下一单会被误伤（先观察；
  备选缓解=窗口尾部 min 结算）
- 目标 yaw 相对发令时刻，多步可漂移
- 调度器两遗留：|Δs|=4 飞步（低权稀有）、方向无持续性（漫游）
- stance 滑移项与动脚误差轻微重复计（w_st 小权）
