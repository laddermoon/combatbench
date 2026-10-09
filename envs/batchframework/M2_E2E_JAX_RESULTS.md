# M2 补充实验：纯 JAX 端到端性能上限探测

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

> 目的：回答"如果策略也在 JAX 里、全链路零 host 往返、不经任何框架
> 转换，MJX 的吞吐天花板是多少"。**这是性能上限探测，不是语义验证**
> ——不要求与主路径行为一致，只要求复杂度同量级。

## 1. 实验设置

| 项 | 配置 |
|---|---|
| 物理模型 | 同一 `battle_circular_v2.xml`（双 humanoid，nq=56, nv=54, nu=42, condim=3, impratio=10, 64 geoms，圆形墙） |
| 控制 | 与主路径一致：每动作步 25 物理子步 + PD 控制（复用 `mjx_simulator` 已编译 jit 函数，同源） |
| 策略 | JAX 原生 MLP `96→256(tanh)→256(tanh)→21(tanh)` + log_std(21)，≈95K 参数 —— 与 `TruncatedNormalPolicy` 同参数量/FLOPs；双机共享权重（与 self-play 一致），含高斯采样 |
| 观测 | 设备端 96 维提取（自身 48 + 对手 48），逐元素算子，与真实观测同量级 |
| 执行结构 | `obs→policy→采样` 融合为一个 jit（每机一个 launch），物理走 `physical_step(25)` jit；Python 循环驱动动作步，**全程零 host 往返** |
| 硬件 | RTX 4090（GPU3），fp32，系统负载 ~115 |

探针：`envs/batchframework/probe_e2e_jax.py`

## 2. 结果

| 配置 | policy | physics(25 子步) | env-substeps/s | agent-transitions/s |
|---|---|---|---|---|
| B=8，随机动作 | ~2ms | 1,820ms | 110 | 9 |
| B=128，随机动作 | 2ms | 4,122ms | **355** | 28 |
| B=256，随机动作 | 2ms | 7,109ms | **408** | 33 |
| B=128，**零动作**（站立） | — | 708ms | 4,523 | — |

编译耗时：reset+step-jit ~45s，scan(25) 首次 ~50-60s（负载高时）。
policy jit 自身编译秒级。

## 3. 结论

### 3.1 瓶颈就是 `mjx.step` 本身，不是集成层

- 策略+观测（95K 参数 MLP + 96 维提取）：**2ms/步，占比 <0.1%**
- 物理：4-8s/步（B=128-256）
- Python 循环 launch 开销：每步 3 个 launch，相对 4s+ 的物理步可忽略
- **把策略搬进 JAX 不能 unlock 任何隐藏吞吐**——MJX-JAX 的慢是
  `mjx.step` 内禀的，与本探测的纯设备端结构无关

### 3.2 接触数量是吞吐的主导变量（新发现）

- 站立（零动作、接触少）：4,523 env-substeps/s
- 缠斗（随机动作、接触密集）：**435 env-substeps/s —— 慢 10×**
- 早期报告的"B=128 峰值 13.4K/s"对应的是低接触状态；**真实对抗
  rollout 的接触密集工况下，吞吐比该峰值低 ~30-40×**
- B=128→256 无 scaling（355→408）—— GPU 已饱和，batch 摊薄无效

### 3.3 与 CPU 生产吞吐对比

| | env-substeps/s | 相对 CPU |
|---|---|---|
| CPU 生产 rollout（实测含策略/观测/插件） | ~472,000 | 1× |
| 纯 JAX e2e，接触密集（真实工况） | ~355-435 | **慢 ~1,100-1,300×** |
| 纯 JAX e2e，站立（低接触） | ~4,523 | 慢 ~104× |
| 早前报告（低接触峰值口径） | ~13,400 | 慢 ~35× |

> 早前 "~35×" 的数字是乐观口径（低接触）。本探测表明在训练真实
> 访问的接触密集状态下差距是 **~3 个数量级**。

### 3.4 对路线图的含义

- mjx-jax 后端在本模型（接触密集、condim=3、impratio=10）× 4090 上
  **没有**加速前景，差距不是集成造成，无法靠工程优化弥合
- 唯一未评估的翻盘候选仍是 `mujoco-warp`（NWORLDS 布局，需独立
  接入层，见 M2_RESULTS.md §6）
- 若后续评估 warp，应直接用**接触密集动作**测试，零动作数字会
  高估一个数量级

## 4. mujoco-warp 对照实验（`probe_e2e_warp.py`）

同一实验换 warp 后端：`mjw.put_data(mjm, mjd, nworld=B)` 原生批量
（NWORLDS 布局，非 vmap），PD 用 `wp.kernel` 写在 warp stream 上
（每子步 1 kernel + `mjw.step`），policy 仍走 JAX——`wp.to_jax` 零拷贝
视图读 qpos/qvel，结果经 `wp.from_jax` 写回。每动作步一次 host sync。

| B | warp env-substeps/s | mjx-jax env-substeps/s | 倍数 |
|---|---|---|---|
| 8 | 514 | 110 | ~4.7× |
| 128 | 10,482 | 355 | ~30× |
| 256 | 24,758 | 408 | ~61× |
| 512 | 45,172 | — | — |
| 1,024 | 75,403 | — | — |
| 2,048 | 146,813 | — | — |
| 4,096 | 254,083 | — | — |
| 8,192 | **308,100** | — | — |

补充：kernel 编译一次 ~93s（缓存于 `~/.cache/warp`，二次启动 ~3s）；
200 子步随机大力矩下 qpos 全部有限，物理未发散；接触容量默认
48/world（真实工况下可能截断——warp 对超容量接触做丢弃处理，语义
验证阶段需单独确认影响）。

### 结论修正

- **warp 仍有 scaling**：B=8192 才趋缓（254K→308K），mjx-jax 在
  B=128 就已饱和——验证了"瓶颈是 jax 后端的稠密 vmap 模型，而非
  任务本身不可 GPU 化"
- 单张 4090 上 warp ≈ **0.65× CPU 生产吞吐**（308K vs 472K/s），
  且未完全饱和；更大 batch 或更强卡（A100/H100）大概率反超
- **mjx-jax vs warp 差 ~200×**：同样语义、同样模型，后端差异是
  决定性的。mujoco-warp 是唯一有翻盘潜力的候选
- 但 warp 接入不是免费的：NWORLDS 布局与现有 vmap 代码不兼容，
  `mjx.Data` 接口不通用，需要独立适配层；且 warp 的 MuJoCo 覆盖是
  子集（验证语义等价性需要重新跑 M1/M2 fixture 体系）

## 5. 方法学备注

- 不用 `jax.jit` 套全 episode：外层 jit 会把 25 子步 scan 内联进整个
  200 步图，XLA 编译 >15min 超时。改为逐动作步驱动（编译 ~100s，
  launch 开销 <0.1%）。
- `XLA_FLAGS=--xla_gpu_autotune_level=1` 对结果无实质影响。
- 高系统负载（~115）主要拖慢编译；稳态 GPU 数字受影响有限，但仍应
  视为有噪环境下的下界估计。
- `sim.set_action` 走 `np.asarray` 会强制 host sync——纯设备端探针
  需直接写 `sim._action_jax`。
