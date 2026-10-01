# M6 训练 pilot 结果：standup_floor04 设备 rollout 能否学会

日期：2026-09-30。计划见 `M6_PLAN.md`。回答一个问题：**设备（warp）
collector 产出的数据能否驱动与原实现相同的训练问题并学会任务**。
性能收敛不在本轮范围（见 §5 遗留）。

## 1. 实验设置

| run | collector | seed | 备注 |
|---|---|---|---|
| `m6_pilot_cpu_s42` | cpu，16 workers | 42 | 与主参考 Run B 同 seed |
| `m6_pilot_cpu_s43` | cpu，16 workers | 43 | 噪声带第二臂 |
| `m6_pilot_dev_s42` + `_s42b` | device（warp），B=512 单波 | 42 | u55 崩溃后从 u50 checkpoint 续跑（§4） |

三臂实验参数完全一致：512 ep/update、200 步、eval_interval=5、
max_updates=500、相同初始权重（同 seed）。

## 2. 前置验证：cpu_s42 与主参考 Run B 的一致性

`m6_pilot_cpu_s42`（16 workers、含全部 M3–M5 改动的新 HEAD）对
Run B（seed 42、96 workers、历史快照）的 `__RAW_STATS__` 逐字段比较
（排除 timing 与日志 schema 字段）：

- **u1–u68：所有共同数值字段逐位相同**（episode_stats、buffer_stats、
  逐 epoch KL、adv/ret 分布、EV/confidence 等）。
- u69 出现首个差异：`grad_norm_actor_mean` ~1e-9；u70 起 kl_mean 等
  在 5–7 位小数漂移，u100+ 被混沌放大。

结论：

- rollout worker 数只影响墙钟，**不影响数据**（per-job 种子确定性成立）。
- M3–M5 的设备路径改动对 CPU 路径数值零污染。
- **bit-identical 不是可行的长程验收标准**——同后端同 seed 也只撑到
  u68。两臂判据正确选择了"学习曲线与分布落在 seed 噪声带内"。

## 3. 学习曲线对照（eval：success / max_pot）

| u | cpu_s42 | cpu_s43 | dev_s42 |
|---:|---:|---:|---:|
| 50 | 0.000 / 0.371 | 0.000 / 0.380 | 0.000 / 0.334 |
| 150 | 0.000 / 0.579 | 0.008 / 0.560 | — |
| 300 | 0.000 / 0.740 | 0.695 / 0.903 | 0.000 / 0.736 |
| 405 | 0.000 / 0.857 | 1.000 / 0.972 | 0.047 / 0.892 |
| 450 | 0.000 / 0.874 | 1.000 / 0.987 | **0.977 / 0.913** |
| 460 | 0.000 / 0.880 | 1.000 / 0.990 | **0.914 / 0.910** |
| 470 | 0.008 / 0.886 | 0.992 / 0.984 | **0.992 / 0.923** |
| 500 | 0.445 / 0.899 | 1.000 / 0.992 | **1.000 / 0.936** |

- 噪声带实测很宽：同为 CPU、仅换 seed，s43 在 ~u330 完成跃迁，
  s42 到 u500 仍在 0.45 徘徊。
- dev_s42 在 ~u410–450 完成同一跃迁（跃迁期 eval 振荡形态与 CPU 臂
  一致），**终局落在两臂之间，判定在噪声带内**。
- rollout 侧 `final_potential_mean`、reward_mean、KL/EV/uncertainty、
  clip_frac 形态全程与 CPU 臂同构，无系统性偏移。

## 4. 后端交叉评估（M4 T4 冻结初态协议，64 初态，确定性策略）

`m6_xeval.py` / `m6_xeval_results.json`：

| 策略 | CPU env | warp env (B=64) | 跨后端差 |
|---|---:|---:|---:|
| cpu_s42@500 | suc 0.492 / mp 0.899 | suc 0.484 / mp 0.899 | ≤0.008 |
| cpu_s43@500 | suc 1.000 / mp 0.992 | suc 1.000 / mp 0.991 | ≤0.001 |
| dev_s42@475 | **suc 0.961 / mp 0.922** | suc 0.969 / mp 0.922 | ≤0.008 |

- **主验收通过**：设备训出的策略在 CPU 参考物理上 success 96%，
  能力可移植，非 warp 数值伪影。
- 反向检查通过：CPU 策略在 warp 上能力保持（含 s42 的 49% 如实重现）。
- 跨后端指标差 ≤0.008，在 eval 方差量级内。

## 5. 事故与修复（本次 pilot 暴露的真实缺陷）

**u55 dev 臂崩溃**（`nefc overflow - please increase njmax` → CUDA
device assert 710），修复于 `71b601c7`：

1. **`put_data` 参数语义 bug**：`nconmax` 是 per-world 语义，代码误传
   `B*48` → 总接触容量按 B²×48 增长。这是此前"B=512 OOM /
   B=2048 不可行"结论的真正成因之一，也放大了 u55 的约束溢出。
   修正为 `nconmax=48`、`njmax=512`。
2. **XLA 预分配**：warp 子类经父类 ctor 初始化 jax 后端，预占 ~18GB。
   修复为 `_init_jax=False` 跳过 jax 专属初始化（`790ecd3b` 起
   父类提供 `_init_jax` 开关与同构 numpy statics）。

修复后复测（正式 runtime 路径，非裸 probe）：

| B | cap48 显存 | env-substeps/s |
|---:|---:|---:|
| 512 | 3.0 GB | 58.8K |
| 2048 | 3.3 GB | 237K |
| **8192** | **4.4 GB** | **504K**（> CPU 96-worker 池 ~472K） |

续跑 `dev_s42b`（u50→500）全程 445 个 update 无 device 异常。

## 6. 计时分解（dev 臂稳态）

| 阶段 | 早期（摔倒态多） | 后期（站立态多） |
|---|---:|---:|
| rollout total | ~170s | ~77–84s |
| ├ reset | ~26s | ~11s |
| ├ step（物理+obs+插件） | ~136s | ~62–69s |
| ├ assemble | ~7s | ~3s |
| └ policy | <1s | <0.2s |

物理步耗时随接触数下降；单波 B=512 的 launch-latency 地板决定了
512 ep/update 形状下设备路径没有墙钟优势（≈10× 慢于本机 16-worker
CPU，更远逊于 96-worker 生产基线 10.85s）。warp 的价值区在
`episodes_per_update` ≫512 的大 batch 形状（§5 表）——按 ROADMAP
暂停条件，这属于另立的优化实验，不算本轮等价迁移的加速证据。

## 7. 判定

- P0（eval 经 device collector 端到端）✅
- P1（u50 趋势正常）✅
- P2（u150 进入噪声带）✅
- P3（终局同档）✅ —— dev_s42 u500 eval success **1.000**
  （max_pot 0.936 / final_pot 0.931），末段 10 次 eval 均值 ~0.96；
  介于 cpu_s42（u500 suc 0.445，慢 seed）与 cpu_s43（1.000）之间
- 交叉评估：✅（§4）

**R1→R2 的"能不能学会"判据通过**：设备 collector 与 CPU collector
在同一训练问题上产生分布等价的轨迹，PPO 消费无差异，策略能力可
移植回参考后端。

## 8. 遗留与下一步

- **M7 验收协议需先修订**：M0 冻结的"端到端加速 ≥2×"在 512 ep/update
  形状下物理不可达（延迟地板）；须把加速对象重定义为"同等墙钟下
  更大 batch"或承认原形状不可达。此决定应在跑正式验收前冻结。
- 大 batch 优化实验（`episodes_per_update` 2048–8192，可选多卡分波）
  是 warp 路径价值主张的正路，尚未做。
- dev 臂 timing 显示 reset 占 ~11–26s/update、assemble ~3–7s——
  大 batch 化后这些 host 段也需要进 profiling。
