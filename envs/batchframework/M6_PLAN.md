# M6 计划：训练 pilot —— 设备路径能否学会任务

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

ROADMAP §9 的前半部分。本轮只回答一个问题：**DeviceRollouter 产出的
数据能否驱动与 CPU 等效的学习**（同一训练问题、同一配置、同一初始化）。
性能收敛（ROADMAP §9 工作包 4/5）不在本轮范围——设备路径当前慢 CPU
~37×，pilot 的目的恰恰是判断这条路径是否值得继续做性能工程。

## 验收语义

"能学会"的定义（对齐 ROADMAP §7 冻结门槛）：

- **学习进展等价**：device 臂的 eval max_pot/final_pot/success 曲线落在
  CPU seed-噪声带内（用第二个 CPU seed 定带宽，不能只靠单 seed 对照）
- **无系统性偏差**：最终 success 差异 ≤ 已声明的非劣效门限（M0 冻结值）；
  若 device 臂明显更快或更慢收敛，需归因到已声明差异（fp32 物理/
  摔倒分布统计等价），否则视为未通过
- **暂停条件（ROADMAP §9 原文）**：效果只能通过改任务或另调一套 PPO
  才成立 → 不算等价迁移成功

明确边界：逐轨迹不等价（RNG 流、fp32/fp64 物理差异 → 混沌发散），
比较的是**分布与学习曲线**，不是 trajectory identity。

## 实验设计

两臂完全同配置，唯一变量是 collector：

| 臂 | 命令关键参数 | 用途 |
|---|---|---|
| CPU-s42 | `--collector cpu --seed 42` | 主对照 |
| CPU-s43 | `--collector cpu --seed 43` | seed-噪声带（CPU 臂 ~6s rollout/update，便宜） |
| DEV-s42 | `--collector device --collector-batch-size 512 --seed 42` | 待验证臂 |

- 实验：`standup_floor04`，PPO 参数全默认（与参考 run 一致）
- 同 seed → `set_seed(cp.seed)` 保证两臂初始权重逐位相同；
  `rollout_seed = seed + u*512` 与 `eval_seed = seed+100000+u*97`
  两臂一致 → job 集合与评估集逐点相同
- B=512 单波（jax 预分配修复后 cap48 占 ~4.4GB，放得下）；跑 GPU1
  （GPU0 有常驻租户进程）
- `--dump-at` 预挂 u50/u150/u350 的 dump sentinel，偏差分析时现成

## 分阶段预算（对齐 CPU 参考曲线 u85/u210/u335/u460）

| 阶段 | 到 update | CPU 参考锚点 | 预计时长(device ~220s/up) | 通过判据 |
|---|---|---|---:|---|
| P0 | u5 + 首次 eval | 冒烟 | ~0.5h | eval 经 device collector 端到端跑通（M5 W4 用 eval_interval=9999 跳过了 eval，stochastic=False 的 eval job 只做过单元验证）；无漏帧/断言 |
| P1 | u50 | reward_mean 上升趋势 | ~3h | r_potential 均值/方差与 CPU 臂同量级；KL/clip_frac/EV 无异常 |
| P2 | u150 | eval85 success 0.008, max_pot 0.42 | ~9h | eval 曲线进入 CPU 噪声带；success 出现非零时点可比 |
| P3 | u460 | eval460 success 0.977 | ~28h | 最终 success/末帧能力与 CPU 两臂同区间 |

每阶段检查点不过夜硬等：stage 间人工查看曲线后再放行下一阶段
（同一 run 可用 `--resume-from` 续跑，checkpoint 每 update 落盘）。

## 偏差归因流程（命中任一异常时按序执行）

ROADMAP §9.3 的分类：reset / 物理 / 观测奖励 / 采样分布 / 数据时序 /
训练处理。工具已有：

1. **reset 分布**：两臂摔倒初始状态统计对照（z 高度、fall 步数直方图）
   —— M4 T3 的分布验收方法；`episode_metrics`/dump 里可取
2. **观测奖励**：把 device episode 的初始状态注入 CPU sim，逐字段比对
   observer 输出 —— M4 T2 的注入态 harness
3. **采样分布**：`evaluate_actions` 重放 rollout log_prob —— M5 W2
   工具；逐 update 抽 64 条轨迹验证差值仍是 fp32 噪声级
4. **数据时序**：dump 两臂同 update 的 Episode 流，比对帧序/终止记录/
   final_observation bootstrap —— M5 W1/W5 的结构对照工具
5. **训练处理**：同一份 device dump 喂 CPU buffer/GAE 重算 update，
   与 device run 的 stats 对比 —— 隔离"数据侧 vs 算法侧"

修一处跑一次受影响验证（ROADMAP §9.6），不叠多个变化。

## 交付物

- `M6_RESULTS.md`：两臂 eval 曲线对比表（u50/150/335/460 锚点）、
  seed-噪声带、各 stage 判据结论、偏差归因记录（如有）
- 运行目录：`m6_pilot_cpu_s42` / `m6_pilot_cpu_s43` / `m6_pilot_dev_s42`
- 若通过：冻结该配置为 R2 候选，转 M7（多 seed 正式验收）或先回
  性能收敛工作包

## 已知风险与对策

- **device 臂时长 ~28h**：可恢复（update 级 checkpoint）；若中途发现
  可安全开大的 batch/cap 调优，先记录不动——本轮不比性能
- **GPU 租户干扰**：GPU0/1 有常驻 ~1.9GB 进程；用 GPU1，启动前确认
  显存余量 >6GB
- **seed 噪声带太宽**：若 CPU s42 vs s43 自身差异很大，pilot 判据
  降级为"趋势同向 + 终局同档"，并在 M7 用更多 seed 解决统计功效
- **eval job 路径首次端到端**：P0 就是干这个的，失败先修再进 P1
