# E7-W0 计量基线

测量日期：2025-XX（E7-W0a 落地后）；
探针：`envs/batchframework/probe_e7_baseline.py`（`git 4fb62949` 起）。

## 0. 测量条件

| 项 | 值 |
|---|---|
| 机器 | instance-1f1igpaq（8×RTX 4090 24GB, driver 550.163.01, 1TB RAM） |
| 栈 | torch 2.7.1+cu126, warp 1.12.1, mujoco-warp 3.8.0.3 |
| env 蓝图 | `standup_4stage_dense_v2_env.yaml`（humanoid21 双机，25 子步/动作步） |
| 任务语义 | `max_steps=200`，每格 jobs=B（单波），policy=TruncatedNormal(96→256→256→21) |
| 子步 hook 格 | 蓝图追加 `SubstepProbePlugin`（空 `on_post_phy_step`，只计路径固定开销） |
| 口径 | 每格 3 次 collect，取第 2/3 次中位数；env-substeps = Σ各行累计物理子步；transitions = Σ各 agent 轨迹帧数。两口径分列，不混算 |
| 计时性质 | **host 提交时间**（perf_counter 包裹提交调用；GPU 执行经 sync 点隐式计入 wall）。physics_wall 内含子步 hook/屏障，kernel 估计 = physics_wall − hook_timing[substep] − barrier_time |

## 1. 总吞吐矩阵（env-substeps/s，wall 中位数）

| B | hooks | 1 卡 wall(s) | 1 卡 sub/s | 2 卡 wall | 2 卡 sub/s | 8 卡 wall | 8 卡 sub/s |
|---|-------|------------|-----------|----------|-----------|----------|-----------|
| 64 | off | 56.3 | 5,685 | — | — | — | — |
| 64 | on | 56.9 | 5,627 | — | — | — | — |
| 512 | off | 62.1 | 41,220 | 60.8 | 42,095 | 57.9 | 44,214 |
| 512 | on | 66.5 | 38,513 | — | — | — | — |
| 2048 | off | 67.4 | 151,847 | 65.9 | 155,289 | 61.4 | 166,772 |
| 2048 | on | 79.8 | 128,314 | — | — | — | — |

transitions/s ≈ sub/s ÷ 12.5（每动作步 25 子步 × 2 agent）。

对照：CPU 生产 rollout ~472K env-substeps/s（M2_RESULTS，192-worker
池）；历史 warp 直接探针 B=8192 ~504K（M5_RESULTS，无契约层）。

## 2. 单卡分项分解（hooks off / on，秒/collect）

| B | reset | policy | step | physics_wall | obs_build | barrier | post_action | d2h+export |
|---|-------|--------|------|--------------|-----------|---------|-------------|------------|
| 64 off | 7.31 | 0.25 | 49.8 | 48.0 | 0.80 | 0.09 | 0.90 | 0.01 |
| 64 on | 7.35 | 0.21 | 49.4 | 47.8 | 0.76 | 1.15 | 0.72 | 0.02 |
| 512 off | 10.85 | 0.20 | 50.8 | 49.2 | 0.85 | 0.05 | 0.70 | 0.13 |
| 512 on | 10.89 | 0.20 | 54.8 | 53.2 | 0.79 | 1.69 | 0.69 | 0.13 |
| 2048 off | 13.02 | 0.20 | 53.2 | 51.4 | 1.09 | 0.04 | 0.69 | 0.60 |
| 2048 on | 13.15 | 0.20 | 65.5 | 63.9 | 0.79 | **8.51** | 0.70 | 0.62 |

`reset` ≈ `on_pre_episode` hook（摔倒初始化物理循环，
`DeviceFallenResetPlugin`，≤1000 子步/行）——属**初始化物理**
口径，与 rollout 物理分列。

## 3. host sync 记账（每 collect）

| 点 | hooks off | hooks on | 说明 |
|---|---|---|---|
| rt.term_barrier 调用 | 400（2/步） | 10,400（52/步=2n+2） | 子步路径每子步前后各一屏障 |
| rt.term_barrier_syncs | 800 | 20,800 | 每屏障 2×`bool(.any())` |
| rt.freeze_check/syncs | 1/1 | 1/1 | 仅新 ENDED 行触发 |
| any_running 早退检查 | 25 | 25 | 每 8 步一次 |
| health_scan + d2h_export | 25+1 | 25+1 | — |

## 4. 瓶颈判读

**F1 — rollout 是 host 提交侧（kernel launch）bound，不是 GPU 算力 bound。**
`step` 段 B=64→2048 几乎不动（49.8→53.2s），per-worker B=64→1024
同样不动（49→52s）。每动作步 host 提交 ~245ms（25 子步 ≈ 10ms/子步，
即 mjw.step 内部几十次 kernel launch 的提交成本）。GPU 在中小批量
大量空转。这解释全部三个现象：吞吐随 B 近似线性爬升；同一总批量
分多卡无加速（见 F2）；子步 hook 的边际成本随 B 增长（见 F3）。

**F2 — 固定总批量下多卡几乎不加速。**
B=512：1 卡 41.2K → 8 卡 44.2K sub/s；B=2048：152K → 167K。
瓶颈在每 worker 的 host 提交速率，分卡只是把固定 host 成本复制 N
份。多卡的价值是"单卡装不下的批量/更多 jobs"，不是同量加速。

**F3 — 子步屏障成本随 B 与队列深度增长。**
hooks-on 增量：B=64 ≈ 0%，B=512 +7%，B=2048 +18%（barrier_time
8.5s，20,800 syncs ≈ 0.41ms/sync）。sync 等待 = host 排空 GPU 队
列的代价——队列越深（B 越大）单次 sync 越贵。当前生产蓝图（standup/
basic_balance）**均无子步插件**，走的已是 hooks-off 路径；该成本
只在启用子步插件时出现。

**F4 — reset（初始化物理）是第二大固定项。**
7.3s@B=64 → 13.1s@B=2048，占 collect 13–20%（200 步 episode）。
摔倒循环逐子步跑 mjw.step——与 rollout 同一 launch-bound 机制，
同样受 F1 影响。

**F5 — policy/d2h/export/obs_build/hooks 合计 <3%。**
不在优化面内。

## 5. 对 W1/W2 的决策依据

- **W1（barrier dirty-flag）**：hooks-off 路径本就只有 2 屏障/步
  （barrier_time ≤0.09s）——收益≈0；hooks-on 路径 8.5s@B=2048
  有约 10% 收益空间，**但前提是实验真用子步插件**。判定：W1 做
  （机制简单、语义零风险），但期望收益按"启用子步 hook 的蓝图"
  计，不是全局收益。
- **W2 头号候选 = CUDA Graph**（E2.1 快路径）：F1 表明 ~10ms/子步
  的 host 提交是总吞吐的主约束。`physical_step(n)` 无回调路径若
  能整段图捕获（控制序列在图外写入），理论上把 25 子步的几十×25
  次 launch 压成 1 次 replay——潜在量级是数倍而非百分之几。
  约束：仅无子步回调路径可用；warp graph 捕获可行性需先验证
  （mjw.step 是否有 host-side 分支/动态形状）。
- **次候选 = reset 编排**（E2.2）：13s/67s 的初始化物理值得单独
  看——摔倒循环能否与 rollout 重叠、或同样走图化。
- kernel 合并（E2.3）与图化同源（减少 launch 数），若 graph 不可
  行则退而求其次。
- 接触聚合/通信优化按现数据无依据，暂缓。

## 6. 已知限制

- `hook_timing`/`physics_wall` 是 host 提交时间，非 GPU kernel
  执行时间——F1 的"launch-bound"判读依赖"时间随 B 不变而提交
  量恒定"的推理，未做 nsys/ncu 佐证；若 W2 图化前需要更硬的证据
  可补一条 nsys trace。
- 8 卡格只测了 hooks-off（子步插件格意义在单卡已充分暴露）。
- `per_worker_reports` 中 hook/seg 计数跨 collect 累计，表中已
  按 collect 数平均。
- B=8192 未测（M5 历史数据 ~504K 可作趋势外推参考；本基线聚焦
  契约层当前形态，B 上限非瓶颈判读所必需）。
- 子步 hook 用空插件——测的是路径固定开销，真实子步插件（如
  CPU 侧 imbalance_termination 的逐子步判定）会在其上叠加自身
  计算。
