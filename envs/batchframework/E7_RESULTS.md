# E7 结果：有依据的执行优化

状态：完成（W0 计量 / W1 屏障收敛 / W2b 物理图化 / W2c 观测图化 / W3 规模记录）。
探针：`probe_e7_baseline.py`；机器 `instance-1f1igpaq`（8×RTX 4090,
torch 2.7.1+cu126, warp 1.12.1, mujoco-warp 3.8.0.3）。
测量条件、口径（env-substeps 与 transitions 分列）、负载噪声说明
见 `E7_BASELINE.md`——本文只记 E7 落地后的净结果与判读。

## 1. 总账（1 卡，hooks-off 图化路径，B 行 × 200 步 × 25 子步）

| B | 基线 sub/s | E7 后 sub/s | 总加速 | wall(s) |
|---|-----------|------------|-------|---------|
| 512  | 41,220  | **251,946** | **6.1×** | 10.2 |
| 2048 | 151,847 | **581,892** | **3.8×** | 17.6 |

对照：CPU 生产 rollout ~472K env-substeps/s（192-worker 池，M2_RESULTS）
——**单 4090 已超整个 CPU 池**，且这是含契约层/生命周期/记录/导出
全开销的 `DeviceRollouter.collect` 路径。

机制分解（B=2048, wall 67.4s → 17.6s）：

| 工作包 | 机制 | 增量 |
|---|---|---|
| W1 | 终止屏障融合单检查 | syncs/collect：hooks-on 20,800→10,402，hooks-off 800→402（结构性减半；wall 收益在共享机负载噪声下不可分辨，如实记录） |
| W2b | 无回调 `advance` 整段 CUDA Graph（wp.ScopedCapture） | physics_wall 51.4s→0.15s；collect 3.2–4.9×；reset 连带 13.0→3.1s（摔倒初始化走同一路径） |
| W2c | `obs_build` torch.cuda.graph 重放 | obs_build 14.0s→0.005s；collect 21.3→17.6s（+21%） |

## 2. 规模记录（hooks-off 图化路径，`0afab335`，GPU 空闲时段）

| 配置 | 每卡行数 | wall(s) | env-substeps/s | transitions/s |
|---|---|---|---|---|
| 1卡 × B=512  | 512  | 10.2 | 251,946 | 20,156 |
| 1卡 × B=2048 | 2048 | 17.6 | 581,892 | 46,551 |
| 2卡 × B=2048 | 1024 | 14.9 | 688,920 | 55,114 |
| 8卡 × B=2048 | 256  | 9.95 | 1,029,596 | 82,368 |
| 8卡 × B=8192 | 1024 | 19.2 | **2,136,162** | **170,893** |

判读：同总批量多卡仍非线性（每卡行数下降 → 单卡欠载，B=256 时
launch/GPU 利用双低）；**多卡的价值是装下更大总批量**——8 卡喂饱
后聚合 2.1M sub/s ≈ 4.5× CPU 池。这与 F2 判读一致，图化没有改变
"分卡复制固定成本"的结构，只是把固定成本压小了。

## 3. 瓶颈演化（W0 → W2c 后）

1. **W0 基线**：host 提交 bound（F1），~10ms/子步 launch 成本，
   step 时间与 B 几乎无关。
2. **W2b 后**：physics 图化 → obs_build（torch 小算子序列）成新
   launch-bound 段（66%）。
3. **W2c 后**：obs 图化 → **系统转为 GPU-work-bound**。判读证据：
   `obs::standing_balance_a`（第一个含 `nonzero` 隐式 sync 的
   observer）吸收 13.0s 的队列排空等待，而 hooks-on 格（物理
   eager、队列始终很浅）同一 observer 实测仅 0.58s——13s 是 GPU
   排空等待不是 observer 自身开销。B=2048 每动作步 ~88ms 几乎
   全是真实 GPU 计算。

## 4. 图化资格与限制

| 路径 | 资格 | 说明 |
|---|---|---|
| `advance(n)` 无子步回调 | ✅ wp graph | PD target/xfrc/sched 写固定 buffer，图外更新 replay 可见；图按 (n_steps, sched) 键缓存，`mujoco_warp.Data` 重建时失效 |
| `advance` 带子步回调 | ❌ 回退 eager | pre/post_step 回调是 Python 侧逻辑，不可捕获——hooks-on 格实测 B=2048 142K sub/s（物理仍 launch-bound，obs 图仍生效） |
| obs_build | ✅ torch graph | `_feet_forces_dense` 固定形状变体（dense+mask 替代 `nonzero`）；按 state 身份惰性捕获/失效；中间分配走 torch 图池 |
| 终止屏障/observer/记录 | — eager | 含 host 读数/动态形状，按设计不图化 |

开关：`CB_WARP_GRAPH=0` 关物理图；`CB_OBS_GRAPH=0` 关观测图。
捕获异常均永久回退 eager（`_graph_broken`）。

## 5. 语义等价性

- physics 图：eager-vs-graph dqpos 差 3.7e-3，与 mjw 自身
  run-to-run 噪声（contact 原子序，6.0e-3）同量级；
- obs dense 变体：wave 契约测试 golden 对照全绿（逐帧逐字段）；
- 子步数/接触/容差/模型保真度零削减；终止屏障语义不变
  （W1 只合并检查点不减少屏障）。

## 6. "不值得做"清单（按测量判定）

- **observer 接触提取去重**（standing_balance_a/_b 各自走一遍
  `contact_forces_flat`）：hooks-on 格实测真实 observer 总开销
  ~0.9s/collect，去重省 ~3%——收益在噪声内，还引入跨 observer
  共享状态耦合。不做。
- **kernel 合并（E2.3）**：与图化同目标（减 launch 数），图化已
  覆盖且更彻底。不做。
- **导出批量化/D2H 压缩**：d2h+export ≈ 0.6s（3.4%）。不做。
- **接触容量聚合/通信**：无测量依据。不做。
- **reset 编排**：2.9s（含 1000 子步摔倒初始化物理的真实 GPU
  工作）——进一步收益须改初始化语义，超出 E7 范围。记录为已知项。

## 7. 回归

- `pytest tests/ envs/framework/tests/ envs/humanoid21/tests/` →
  **414 passed, 4 skipped**（deselect 1 项：
  `test_m2_cross_backend_fixtures`——**既有失败**，fixture 依赖
  哈希自 `17d031fd` 起 stale，与 E7 无关；ignore
  `test_stage_seg_rewards.py`——既有 collection 错误）。
- GPU 契约套件（wave/lifecycle/rollouter/runtime/balance/standup/
  multi/warp）全绿，且 wave golden 对照确认 dense 观测路径逐帧
  数值等价。

## 8. 遗留认知项

- `hook_timing`/分项是 host 提交墙钟：图化后 post_action 里首个
  隐式 sync 的 observer 名下会累计"排空等待"——归因以"谁发起
  同步"为准（§3 已用 hooks-on 对照解耦）。
- 流水线深度结构上限 1 步：每步终止屏障/observer 各有一次必要
  host sync，host 无法超前提交下一步——GPU-bound 后剩余优化空间
  在减少 GPU 工作本身（模型/接触预算），已出 E7 范围。
- `test_m2_cross_backend_fixtures` 的 stale 需要一次 fixture
  重录（独立维护项，非 E7 回归）。
