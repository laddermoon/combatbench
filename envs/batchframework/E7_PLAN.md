# E7：有依据的执行优化与规模检查 — 实施计划

> ROADMAP 原文：
> - 在接口不变的前提下评估 CUDA Graph、固定容量接触聚合、kernel 合并、reset 编排、批量导出和通信优化；每次只改变一类执行机制。
> - 正确区分 host 提交时间、GPU 实际执行与端到端时间；编译、启动、reset、policy、physics、observer、记录、导出、合并分别计量。
> - 物理 world-substeps = world 数 × action steps × substeps；agent transitions 另算，双机器人不能让同一次 world 物理推进被计数两次。初始化物理步单列。
> - 固定模型与任务语义，不以削减接触、子步、输出或放宽容差换取吞吐。硬件拓扑、可用资源和同步路径随结果记录。
>
> **放行：** 容量与开销可解释，有 1/2/8 卡的工程规模记录及已知瓶颈；捕获/优化路径对 eager 语义回归通过。没有"必须比 CPU 快 2×"门槛，也不把吞吐增长直接称为 time-to-quality 改善。

## 定位

E1–E6 把正确性/契约/迁移/故障面建完了，代价是热路径上攒了一批
**未计量的固定开销**：每步插件 hook、终止屏障 host sync、obs 重建、
波末 D2H。E7 的任务不是"提速"——是先让这些开销**可解释**，再按
数据决定动哪一类机制。

### 已有基线（历史文档，不可直接当 E7 数字用）

| 来源 | 数字 | 备注 |
|---|---|---|
| CPU 生产池 | ~472K env-substeps/s | M2 §4 实测（10.85s/update ÷ 5.12M substeps） |
| warp B=8192 cap48 | ~504K env-substeps/s | M5，接触修复后 |
| warp B=512 | ~34K env-substeps/s | M5——固定 kernel/调度开销主导 |
| mjx-jax 峰值 | 13.4K env-substeps/s（B=128） | M2，已否决 |

这些数字测的是**裸物理**，不含 E3 以后的插件 hook/屏障/记录/导出。
E7-W0 的第一件事就是重测"框架态"基线。

### 新嫌疑点（终止帧契约引入，未计价）

`_consume_terminations()` 每次调用内含 `bool(rr.any())` +
`bool(newly.any())` 两个 host sync。含端点语义落地后每步调用次数：

- pre-action 屏障 ×1（新增）
- `_pre`/`_post` 子步屏障 ×2n（有子步 hook 时；`_pre` 屏障为新增）
- 步尾 ×1

→ 有子步 hook 的任务每 action step ~**4n+4 次 host sync**
（n=phy_substeps）。即使每 sync ~10μs，B 无关的固定成本对
小 B 工况是显著项。无子步 hook 的快路径为 2 次/步。

## 设计判断

- **O1 — 先分项，后优化。** 现有 `timing` 只分
  reset/policy/step/assemble 四段，"step" 是黑盒。W0 先加**只读
  计量**（不改执行语义）：physics kernel、plugin hooks、observer
  刷新、recorder 写、屏障 sync、obs build、D2H 导出分别计量——
  区分 host 提交时间 / GPU 执行 / 端到端（`torch.cuda.Event` +
  wall + sync 计数三视角）。没有分项数据不动任何机制。
- **O2 — 两类速率分开报。** `env-substeps/s`（world 物理）与
  `agent-transitions/s`（B×2×steps）分列；reset 初始化物理步单列。
  沿用 M 系列口径，不新造指标。
- **O3 — 每次只改一类机制。** 候选按 W0 数据排序，一次落地一类
  并回归 eager 语义。禁止"顺手"跨类改动（ROADMAP 原话）。
- **O4 — 屏障收敛优先于微优化。** 若 W0 证实 host sync 是小 B 主导
  项：先做**提议聚合**——`request_termination` 置一个 device 端
  dirty flag，consume 快路径只读 `dirty.any()` 一次（有提议才走
  完整屏障）；多 consume 合并为一个 flag 检查。这比 CUDA Graph
  收益大得多且语义风险可控。**前提：sealed-ENDED 行为逐位不变。**
- **O5 — CUDA Graph 只包可包段。** 插件 hook 是 Python 回调——
  graph 至多包 physics 块/固定 observer 刷新，不能包整步。
  若 W0 显示 kernel-launch 开销主导（小 B 典型），评估
  `torch.cuda.graph` 包 `physical_step(n)` 无回调路径；有子步
  hook 的路径天然不可 graph（如实记录为"不支持"，不做语义牺牲）。
- **O6 — 不付语义税。** 接触容量、子步数、observer 输出、容差
  全部冻结；任何"优化"导致的物理/数据差异视为回归而非权衡。

## 工作包拆分

### W0 — 计量基线（只读，不改语义）

- 细分计时：`step` 拆 physics / hooks / observer / recorder /
  barrier / obs_build；assemble 拆 D2H / export / metrics。
  实现为可选开关（`sync_stats`/`timing` 扩展，默认开——
  `perf_counter` 调用本身在同步点之后近零成本）。
- host sync 计数细化：`_consume_terminations` 内每个 `bool(.any())`
  单独记数（已有 `term_barrier` 调用次数，补每次调用实际 sync 数）。
- **基线矩阵**：单卡 × B∈{64, 512, 2048} × {standup 无子步hook,
  basic_balance 有子步hook}；2 卡 B=512；8 卡 B=512。指标：
  env-substeps/s、agent-transitions/s、collect wall 分解、
  host sync 次数/step、GPU util。
- 产出：`E7_BASELINE.md`（机器/驱动/拓扑随记录）。

### W1 — 同步点收敛（按 W0 结果执行，预计最高收益）

- `request_termination`/`reset_request` 置 device dirty flag；
  consume 快路径 = 单次 `bool(dirty.any())`，无提议零额外 sync；
  2 个 any() 合并为 1 个 fused mask 检查。
- `any_running()` 检查 cadence 复核（check_every=8 是否最优——
  早退收益 vs sync 成本，按波均步数定）。
- 契约回归：substep barrier / sealed-ENDED / records 归档逐位
  不变（现有 `test_device_lifecycle_contract` + `test_wave_contract`
  即断言集）。

### W2 — 逐项机制评估（每类独立可回滚）

按 W0 排序执行，每项：机制 → eager 回归 → 分项收益记录：

| 候选 | 评估点 |
|---|---|
| kernel 合并 | consume/seal 的 elementwise 链、obs build 拼接 |
| reset 编排 | `reset_rows` host 同步批量化（E1 遗留债） |
| 批量导出 | 波末 D2H 逐叶→单块；`host_export_bytes` 已有记账 |
| CUDA Graph | 无子步hook 快路径 `physical_step(n)`（O5 边界） |
| 接触聚合 | 固定容量下 contact 计数/饱和检查的成本 |
| 多卡通信 | worker→learner Episode 传输（E4 已证 pickle 无瓶颈，复核） |

### W3 — 规模记录与回归

- 1/2/8 卡工程规模记录表（吞吐/显存/同步路径随记）。
- 全量 tests/ 回归 + eager 语义对拍 + device 冒烟训练。
- `E7_RESULTS.md` + ROADMAP 状态更新。

## 放行条件核对

1. **容量与开销可解释** ← W0 分项基线 + W1/W2 后复测表；
2. **1/2/8 卡工程规模记录及已知瓶颈** ← W0 矩阵 + W3 记录；
3. **捕获/优化路径对 eager 语义回归通过** ← 每类机制独立回归 +
   W3 全量；
4. **无"必须比 CPU 快 2×"门槛** ← 如实报数，不宣称
   time-to-quality。

## 不做

- 不重新跑旧 M7 多 seed 长训练验收；
- 不动接触容量/子步数/observer 输出/数值容差；
- 不引入新后端或多机调度；
- 不要求所有候选机制都落地——W0 数据说"不值得做"本身
  是合格结论，记入 E7_RESULTS。
