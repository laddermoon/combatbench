# Batch Framework 时间审计 —— 深度分析

> 类型：记录  
> 日期：2026-10-09  
> 对象：standup 训练负载（512 eps/update，2×B=256，25 substep/step，
> T=200）  
> 方法：训练日志分段计时 + `probe_e7_baseline` 计量探针 +
> 隔离微基准 + nsys kernel 分解  
> 结论先行：**wave 的 77–85% 是物理 GPU 执行，且物理是
> 延迟/占用率受限而非算力受限**——"observer 慢"是同步点记账的
> 假象。真正的优化空间按节 4 排序。

---

## 1. 每 update 顶层分解（稳态均值）

| 段 | device s42 | CPU s42（参照） |
|---|---|---|
| rollout | **9.87s** | 10.91s |
| ppo | 0.74s | 0.68s |
| buffer | 0.21s | 0.23s |
| eval（摊销，每 5 更新一次） | **1.84s** | 0.38s |
| total | 12.68s（中位 10.9，p90 19.8） | 12.22s |

rollout 内部（`probe_e7_baseline` B=256 单卡 collect，median wall 8.94s）：

| 段 | 时间 | 占比 | 说明 |
|---|---|---|---|
| step（200 步） | 7.04s | 79% | 见下 |
| reset | 1.42s | 16% | `device_random_fallen_state` 摔倒模拟 |
| policy | 0.25s | 3% | executor 前向 |
| finalize+d2h+export | 0.07s | 1% | — |

step 内部（隔离测量，B=256 含 sync）：

| 段 | ms/step | 占比 |
|---|---|---|
| physical_step(25) exec | 29.1 | **77%** |
| obs_build | 0.5 | 1%（已图化） |
| barrier/hook/计数/记录残差 | ~8.3 | 22% |

## 2. 关键澄清：observer 不是瓶颈

`probe` 显示 `obs::standing_balance_a` 6.28s（b 仅 0.38s）——**这是
记账假象**：`contact_forces_flat` 内 `torch.nonzero` 变长输出强制
host 同步，排在队列里的物理 kernel 执行时间全部记入第一个 sync
点（observer A）。真实 observer 数学 ≈ 0.4s/wave（observer B 的
测量值 ≈ 真实成本）。

**隔离验证**：裸物理 200×25 子步 graph replay wall=7.11s ≈ 实测
wave 的 step 段 7.04s——物理 exec 就是 wave 本体。

## 3. 物理 exec 的结构（B=256，1.42ms/substep）

### 吞吐–batch 缩放（graph replay）

| B | ms/substep | sub/s |
|---|---|---|
| 256 | 1.42 | 0.18M |
| 512 | 1.65 | 0.31M |
| 1024 | 2.15 | 0.48M |
| 2048 | 3.44 | 0.60M |

8× 行数 → 墙钟只涨 2.4×：**kernel 延迟受限，非算力受限**。

### kernel 分解（nsys eager，15 子步，busy≈0.87ms/substep）

| 族 | µs/substep | 占比 |
|---|---|---|
| **solver**（cholesky 123 + linesearch ~95 + constraint update ~92 + LD/search/jaref ~120） | ~440 | **~50%** |
| 质量矩阵 ops（qLD_acc 108 + mul_m 26 + JTDAJ 30 + qM 6） | ~170 | ~20% |
| 前向动力学树（kinematics/crb/cfrc/comvel/cacc/transmission ~103） | ~103 | ~12% |
| 接触+碰撞（efc_init/jac 25 + narrow 9.5 + broad 6） | ~40 | ~5% |
| 其它（padding/zero/cost/misc） | ~120 | ~14% |

- solver 每子步实际迭代 ~4–5 次（linesearch_iterative ×4.5/substep）；
  **收紧 iterations 上限 100→5 无收益**（1.31→1.51ms，噪声），
  已经在用早停——再压迭代属语义变更需 golden 验证。
- 单 kernel 均值 ~4µs@B=256 → 每子步 ~60+ 小 kernel，
  延迟主导；wall 1.42ms vs busy 0.87ms 的差为提交气泡
  （graph 已大幅压缩，残差是 replay 间隙）。

## 4. 优化空间（按预期收益排序）

| # | 项 | 估计收益 | 工作量 | 语义风险 |
|---|---|---|---|---|
| O1 | **eval 路径重叠/改尺寸**：64 eval jobs 跑 B=256 runtime → 75% 物理浪费在 pad 行（9.1s/eval≈train wave）。方案 a：第二卡异步 eval rollouter（训练环改异步）；b：eval 专用 B=64 runtime（kernel 延迟同 → 只省 ~30%） | ~1.8s/update 摊销（total 14%） | 中 | 无 |
| O2 | **摔倒模拟去同步**：`_draw_actions` 后每物理子步 `bool(newly.any())` host 同步（~1000 sync/reset）。改为设备侧 masked capture + 每 sync_chunk 才同步早退检查 | 1.4s→~0.4s/reset（wave ~10%） | 中 | 无（同数学） |
| O3 | **每步 sync 合并**：~7+ sync/step（barrier×2 + 2 observer×nonzero + timeout nonzero）。observer 接触表改定形 dense 路径（复用 `_feet_forces_dense` 式），termination/timeout 打包进一次 .any() 读 | 残差 8.3ms/step 收一半 → ~0.8s/wave（~9%） | 中-高 | 无 |
| O4 | **波密度/打包**：B=2048 吞吐 3.3×（0.60M sub/s）。同协议下单 update 墙钟收益小（2×256 已并行）；真实收益=**每卡多 run 复用**或更少卡 | 每 GPU 吞吐 2–3× | 低（配置） | 无 |
| O5 | **物理 kernel 深挖**：solver 族 ~50% busy——MJWarp 版本升级/`solver` 参数变体（CG/PG）/nconmax 池尺寸/nsight compute 细剖 | 未知，潜在最大 | 大 | solver 参数=语义变更，需 golden |
| O6 | **ENDED 行物理浪费**：ENDED 行仍全员推进（冻结写回） | standup 零收益（全 timeout）；早终任务大 | 中 | 无 |
| O7 | policy MLP / PPO / export | 已 <3%，略 | — | — |

## 5. 单 run 加速的现实边界

- 512eps wave 地板 ≈ 物理 7.1s + reset + 残差 ≈ 9s——**当前
  9.87s 已贴近**；
- 要把 rollout 压到 CPU 的一半（~5.5s）需要物理 exec ~4s——
  唯一可达路径是 O5（kernel 级）+ O2/O3 清干净非物理残差；
- O4 的方向才是结构性答案：**GPU 在 B=256 只跑到 0.18M sub/s，
  B=2048 达 0.60M**——standup 单 run 用不满卡，收益要按
  "多 run/多实验共享一卡"或"吞吐受限任务（大批量/自博弈）"
  兑现。

## 6. 建议的下一步（可选）

1. O2+O3 落地（纯框架优化，无语义风险，wave 估 ~19%）；
2. O1a eval 异步化（训练环改动，total ~14%）；
3. O5 spike：nsys compute 细剖 solver 族 + MJWarp 新版本
   changelog 对照——若 solver 时间可降，是唯一能把单 run
   rollout 打对折的路径；
4. O4 多 run 复用一卡的调度层（若走"并行实验"路线）。

*测量数据：`/tmp/e7_b256.json`（probe）、训练 run
`train_standup_ppo_20261008_131639`（device）、
`train_standup_ppo_20261009_105422`（CPU）、`/tmp/eager_prof`（nsys）。*

---

## 7. O2/O3 落地结果（2026-01 后续轮次）

已提交：`73a6927e`（O3a dense 接触表）、`59d6f77b`（O3b mask
终止 + freeze 契约修复）。

### 实测（B=256，干净 GPU4，probe 3 次中位）

| 指标 | 优化前 | O3a 后 | O3b 后 |
|---|---|---|---|
| wave wall | 8.94s | 7.98s | 8.03s（噪声内） |
| observer a/b | 6.28s / 0.38s | **0.29s / 0.26s** | 0.39s / 0.32s |
| device_timeout | ~0.01s | 5.52s（承接物理等待） | **0.15s** |
| barrier_time | ~0.04s | ~0.04s | 5.11s（承接物理等待） |
| reset（摔倒模拟） | 1.42s | 1.35s | 1.29s |
| `term_barrier_syncs`/波 | — | ~800 | **402**（=2/步，fused） |

**结论**：observer/timeout 的隐式 sync 全部清除，物理 exec 等待
现在干净地显形在 `barrier_time`（5.1s ≈ 排队中的物理执行）。
wave 8.03s ≈ 物理地板 7.1s + reset 残余 + ~0.9s 框架残差——
**离裸物理地板只剩 ~13%，O3 的 sync 收益已基本拿完**；再往下
需要动物理本身（O5）或波密度（O4）。

### O2 修正：摔倒模拟是物理主导不是 sync 主导

去 per-子步同步后 reset 仅从 1.42→1.29s——隔离测量证实该循环
本身就是 ~900 子步物理 exec（~1.4ms/substep），sync 开销是次要
项。继续压缩需减少子步数（语义变更）或独立 runtime 并行化。

### 附带发现：sealed-ENDED 冻结契约实际未生效（已修复）

`_freeze_ended_rows` 原仅在 new-ENDED 分支内调用——已 ENDED
行在后续步的漂移从未被写回，契约注释"每步写回封存态"不成立。
`test_ended_row_frozen_no_drift` 空转通过（FakeBackend 的
qpos 漂移依赖 qvel，测试从未 seed）。导出与健康扫描均有 mask
保护故无实际数据污染，但属于：① 契约违约；② ended 行白烧
solver；③ 无 mask 读活状态的插件/debug capture 拿脏数据。
已修复：freeze=True 屏障（post-substep/post-observer）将
sealed 检查与 pending 检查融合进同一次 sync，无条件写回；
测试 seed qvel 后成为真测试（64 项设备测试全绿）。

### O1a' 去 padding（eval 小波）——已落地（`215d6b64`）

eval 异步化否（语义要求同步内联），改为**尺寸匹配 runtime**：
`DeviceRollouter` 按 B 缓存 bundle（env 级共享
binding/io_schema/manifest 不变），波派发到 ≥job 数的最小 2
次幂尺寸 runtime。warp 无 masked-step，这是唯一正确形态。

实测小 B 子步成本（GPU4，graph replay）：

| B | ms/substep |
|---|---|
| 32 | 0.994 |
| 64 | 1.023 |
| 128 | 1.106 |
| 256 | 1.229 |

延迟受限确认：B=64 仅比 B=256 便宜 ~17%——但 eval 收益**大于**
物理占比：**64-job collect（B=256 rollouter）：9.1s → 6.4s
（−30%）**。除物理外，摔倒 reset 的 scratch sim 与
export/observer 也随行数缩（reset 1.3→0.88s）。按
eval_every=5 摊销 ≈ 0.5s/update（~4% total）。多卡路径透明受益
（每 worker 按自己分到的 job 数选 bundle）。

### O4 波密度/并行路线——已探明（2026-01）

物理子步延迟曲线（GPU4，graph replay，干净卡）：

| B | ms/substep | 吞吐（裸物理） |
|---|---|---|
| 64 | 1.02 | 0.06M sub/s |
| 256 | 1.23 | 0.21M |
| 512 | 1.41 | 0.36M |
| 1024 | 1.72 | 0.60M |
| 2048 | 2.57 | 0.80M |
| 4096 | 4.21 | 0.97M（未到拐点） |

并行路线实测：

| 路线 | 结果 |
|---|---|
| 多进程同卡（K=2/4 proc × B=512） | ❌ 聚合恒 ~0.32M sub/s——CUDA context 时间片轮转，零重叠收益 |
| 进程内多 stream（2 lane × B=512） | ✅ 真重叠 agg 0.54M（每 lane ~82% 效率），但 < 同行数单波 B=1024 的 0.60M |
| 端到端 B=2048 collect | **120 eps/s/GPU（3.75× B=256 的 32 eps/s）**；含 reset 2.8s + export 0.9s |
| 512 jobs 摊 4 worker ×128 | **端到端 7.76s**（vs 2 worker ×256 ~8.5-9s）|

**结论——延迟与吞吐是两个不同最优解**：

1. **吞吐模式**（seed farm、大批量 eval、自博弈）：**大 B
   单波**。B=2048/4096 仍未饱和；多 lane 被同价大 B 压制；
   多进程 lane 完全无效。部署：`batch_size_per_worker` 直接
   调大即可（bundle 机制保证小波仍按 pow2 匹配）。
2. **延迟模式**（单 run update 更快）：**摊到更多 worker、
   每 worker 更小 B**。512 eps：2×256≈8.5-9s → 4×128=7.76s；
   极限是 B≈64 的延迟地板（子步 ~1.0ms）。**部署零代码**——
   `--collector-devices` 撒满空闲卡即可（512/8=64→B=64
   bundle 自动命中）。
3. **进程内 lane** 的生态位只剩"异构 group 并发"（不同
   env_bp/policy/stochastic 无法合并为一波时的流重叠），
   ~1.5× 上限，实现复杂，暂记备选。
