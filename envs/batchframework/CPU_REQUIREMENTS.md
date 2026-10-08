# Batch Framework 的 Host CPU 需求分析

**结论**：device collector 路径下，一条训练 run 的 host CPU 需求约
**2–3 个核 + ~6GB RSS**；对照 CPU collector（`rollout_workers=96`）
同协议需要的 ~96 核，**host 侧压力下降约 40×**。Rollout 的 CPU
需求本质上是"每 GPU 约 1 核的协程调度"，与 batch 规模、episode
数量弱相关——host 只剩 Python 派发、同步屏障和 D2H/导出三个角色。

测量环境：2×Xeon 8463B（96 核/192 线程，1TB RAM），8×RTX 4090，
standup 实验（`episodes_per_update=512`、双 worker × B=256、
hooks-off CUDA-Graph 路径），2026-10-08 R2 seed 43/44 运行中实测。

## 实测数据（单条 device run = 1 main + 2 workers）

| 进程 | %CPU | RSS | 线程 | 角色 |
|---|---|---|---|---|
| worker ×2 | **~90% 各 ≈1 核** | ~2.0GB 各 | 131 | `DeviceRollouter`：step 循环 Python 派发 + 图重放提交 + 终止屏障 sync + D2H 导出 + `export_episode`（numpy） |
| main | **~15%** | ~1.9GB | 14 | jobs 构建（0.01s）、collect 分派/合并、buffer 轨迹构建、PPO orchestration、eval、日志 |
| spawn helper | ~0% | — | 1 | multiprocessing 资源守护 |
| **合计/run** | **~2.1 核** | **~6GB** | | |

线程数（131/worker）主要是 torch/warp OMP 线程池，绝大多数空闲；
有效需求按 %CPU 计。

### 相位细分（512 eps/update，~11s 周期）

- **collect 段（~9.5s）**：workers 各 ~90%；main 阻塞在 pipe 读，
  近 0%。worker 的单核消耗 = step 循环 Python 开销 + 图重放提交 +
  `check_every` 屏障的 sync 轮询 + 波末 D2H/导出。
- **buffer+PPO 段（~1s）**：main 短时突发（多线程 torch），均值
  摊薄后 15%；GPU0 上的 learner 计算为主。
- **export/jobs/log**：~0。

### 随规模的扩展性

| 量 | 随 B/eps 的关系 |
|---|---|
| worker 核数 | ≈GPU 数（每 worker ~1 核），**与 B 无关**——提交成本恒定 |
| worker RSS | ~2GB，B≤2048 实测基本持平（CUDA ctx + torch 缓存主导） |
| main 的 buffer 段 | ~线性于 episodes：0.17s@512eps，~1.2–9s@4–12K eps |
| main 的 ppo 段 | 线性于 transitions（GPU-bound，host 只派 kernel） |

## 对照：CPU collector 同协议

`train_standup_ppo_20260825_143109`（CPU 参照 run）：`ParallelRollouter`
`rollout_workers=96`——96 个 MuJoCo 进程 rollout 段各 ~100%，即
**~96 核** + main 进程；rollout=5.8s/update 的代价是几乎占满单
socket。device 路径同协议 rollout~10s/update 但只用 ~2 核。

## 推论

1. **8 卡满配单 run**：8 workers ≈ 8 核 + main ~1 核 < 10 核。
2. **多 run 并行**：3 条 device run（6 卡）实际只耗 ~6 核——
   96 核机器跑 4 组双卡 run 仍有 ~88 核富余；CPU 不再约束 GPU
   利用率，GPU 数才是并发上限。
3. **释放的 host 算力可再投入**：buffer/导出等 host 段目前是单
   点串行（线性于 episodes），大批量 run 的 buffer 段（~9s@12K
   eps）可用富余核并行化——此前是 CPU collector 下不可能的方向
   （核全被物理吃满）。
4. **worker 单核 ~90% 是下一个 host 侧瓶颈点**：CUDA-Graph 已把
   kernel 提交压到每步两次 replay，剩余 90% 主要是 step 循环的
   Python 派发 + sync 轮询。若要继续压低 host 需求或提速
   hooks-on 路径，优化对象是派发循环本身（或减少 sync 频率），
   而非核数。
5. **eval/视频渲染**：视频走 MuJoCo 子进程（短时单核突发）；
   eval 与训练同 collector 路径，无额外常驻需求。

## 测量方法

`ps %CPU`（进程瞬时/周期均值）+ `/proc/<pid>/status` VmRSS/Threads
+ 训练日志 `timing` 字段（rollout/buffer/ppo/eval 分段）交叉
验证；对照组为同实验 CPU collector 历史 run 的 `rollout_workers`
配置与进程模型。采样窗口覆盖 rollout/buffer/ppo 全相位。
