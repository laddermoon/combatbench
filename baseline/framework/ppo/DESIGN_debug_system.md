# 调试体系实现设计与规划

**定位**：`DEBUG_GUIDE.md` 定义了**用户看到什么**；本文档定义**怎么实现**。实现时以本文档为总纲。

**阅读顺序**：先读 `DEBUG_GUIDE.md`（产品定义），再读本文档。本文档中的每个组件都必须能追溯到 `DEBUG_GUIDE.md` 里的某个工具或纪律。

**关联**：`GUIDE.md`、`DESIGN_unified_exploration_control.md`、`trajectory.py`、`trainer.py`、`loop.py`

---

## 1. 设计原则（不可妥协的五条）

这五条是所有后续决策的约束。违反其中任何一条的方案直接否决。

| # | 原则 | 理由 |
|---|---|---|
| **P1** | **调试路径不得改变训练动力学** | 尤其**不得消耗 RNG**。否则观测行为改变被观测对象，得到的是"另一个训练" |
| **P2** | **不复制生产逻辑** | 离线重算必须**调用**真实的 `build_trajectories` / `ppo_update`，不能有第二份 GAE 实现。否则调试路径会漂移，最终调试的是不存在的 bug |
| **P3** | **关闭时零开销** | 所有钩子形态为 `Optional[...] = None` + `if x is not None`。默认路径不增加一次分支之外的成本 |
| **P4** | **聚合量常态开启，逐帧量按需开启** | 聚合量成本 O(n·C) 相对前向传播可忽略 → 始终开。逐帧数组体积大 → 只在 sink 存在时写 |
| **P5** | **框架不解释语义，实验提供语义** | 沿用 `explore_factor` / `floor_weight` 已确立的模式。框架搬运，实验解释 |

---

## 2. 现状事实（实现前必须知道的锚点）

| 事实 | 位置 | 对设计的影响 |
|---|---|---|
| `Trajectory` 已有 `explore_factor` / `floor_weight` 两个 per-frame 可选字段 | `trajectory.py:112-113` | **provenance 沿用同一范式**：`Optional[...] = None`，buffer 填默认 |
| `policy_exports/u{u:05d}` 在 rollout **之前**导出 | `loop.py:459-462` | ✅ 它确实是 θ_old，可直接用于重算 |
| 但导出**只含 actor，不含 critic** | `to_blueprint()` | ⚠ 快照必须**额外保存 critics**，否则算不出 `V(s)` |
| `PPOBuffer` 直接接收 `List[Trajectory]` 并整批拼接 | `trainer.py:112-118, 156-185` | provenance 在此转成「按轨迹的段表」 |
| `ppo_update` 已收敛为单一入口，返回 `UpdateStats` | `trainer.py:408-420` | debug sink 作为新增可选参数挂在这里 |
| `ppo_update` 的返回类型标注是过期的 `Dict[str, float]` | `trainer.py:420` | 顺手修正为 `UpdateStats` |
| `aw_normed` / `conf` / `normed` 已在合成处同时存在 | `trainer.py:704-726` | 新聚合量在此计算，无需额外前向 |
| `Episode` / `EpisodeCollection` 已有完整 save/load + hash 校验 | `episode.py:408-552`, `episode_collection.py:123-194` | 快照的落盘层**已完成**，只需接线 |
| `Episode` 自带 `episode_index` / `base_seed` | `episode.py` | provenance 引用它，**不要用列表下标** |
| `analyze_training.py` 已有 `TrainingLogAnalyzer` 类 + 9 条规则 | `analyze_training.py` | `debug.py health` **导入复用**，不新写日志解析器 |
| `trainer.py` 已有 8 类 `diagnostics.append` 告警 | `trainer.py:582,665,738,903,1005,1028-1055,1097` | 不变量守卫与之合流，统一前缀 |

---

## 3. 核心数据模型决策

### 3.1 帧溯源（阻塞级，最先做）

**问题**：`PPOBuffer` 拼平后无法把索引映回 `(episode, agent, t)`。框架**原理上无法推断**——`build_trajectories(episodes) -> List[Trajectory]` 是实验拥有的多对多映射。

**决策**：由实验提供，**按轨迹**而非按帧存储。

```python
# trajectory.py
@dataclass(frozen=True)
class TrajectoryProvenance:
    """轨迹来源。调试功能的前置依赖；生产路径不读取。"""
    episode_index: int          # Episode.episode_index（不是 episodes 列表下标）
    agent_id: str
    t_start: int                # 本段首帧在 episode 内的步号
    termination_reason: str = ""    # 该 agent 的终止原因，"" = 未终止

@dataclass
class Trajectory:
    ...
    provenance: Optional[TrajectoryProvenance] = None
```

**为什么按轨迹而不按帧**：轨迹是连续段，`(episode, agent, t_start) + 段内偏移` 已足够定位任意帧。按帧存要多 3 个长度为 n 的数组，纯浪费。

**帧 ID 规范**（全体工具统一）：

```
ep{episode_index:04d}:{agent_id}:{t}          例：ep0003:robot_a:137
```

**PPOBuffer 侧**：新增 `self.seg_provenance: List[Optional[TrajectoryProvenance]]`（与 `ep_lengths` 平行）。提供

```python
def frame_id(self, flat_index: int) -> str        # 平坦索引 → 帧 ID
def frame_ids(self) -> Optional[np.ndarray]        # 全部帧 ID（有 provenance 时）
def find_frames(self, expr: str) -> np.ndarray     # 供 frame --where 使用
```

`provenance=None` 时 `frame_id` 返回 `flat:{i}` 并在调试工具中提示「该实验未提供 provenance，逐帧溯源不可用」——**明确失败，不静默给出错误 ID**。

**实验侧成本**：`exp_standup_step_v3._build_agent_trajectory` 已经知道 `agent_id`，加两行即可。**不强制**——只有需要逐帧调试的实验才填。

### 3.2 调试数据出口（DebugSink）

**决策**：`ppo_update` 增加可选 `debug_sink` 参数，**只写不读**，trainer 不得因 sink 存在与否改变算法分支（P1/P3）。

```python
# debug/sink.py
class DebugSink(Protocol):
    def record(self, stage: str, name: str, value: Any) -> None: ...
    def record_minibatch(self, epoch: int, mb: int, name: str, value: Any) -> None: ...

class NpzSink:
    """累积到内存，close() 时按 stage 分文件落盘。"""
```

`ppo_update` 内的记录点（stage 名固定，作为工具与实现之间的契约）：

| stage | 记录内容 | 记录位置 |
|---|---|---|
| `buffer` | `old_log_prob`、`explore_factor`、`floor_weight`、`sample_weights`、`frame_ids` | 入口 |
| `gae` | 每通道 `values` / `deltas` / `advantages` / `returns` / `bootstrap_value` / `active_mask` | 各通道 GAE 后 |
| `combine` | 每通道 `aw_frame` / `aw_normed` / `conf` / `normed_adv` / `contribution` / `norm_mask`；全局 `aw_l1_sum` / `combined_adv` | `trainer.py:704-726` 附近 |
| `update` | 逐 minibatch `ratio` / `clip_mask` / `policy_loss` / `floor_loss` / `grad_norm`；`--full-grad` 时单个 minibatch 的全量梯度 | epoch×mb 循环内 |

**关键约束**：`stage` 与 `name` 是**工具依赖的契约**，改名等于破坏工具。写进 docstring。

### 3.3 常态聚合量（无需 sink）

在 `trainer.py:704-726` 计算，加入 `UpdateStats`。这四项对应 `DEBUG_GUIDE.md` §3.3 的归因工具。

| 字段 | 类型 | 定义 |
|---|---|---|
| `actor_weight_normed` | `Dict[str, float]` | `aw_normed` 的均值（逐帧 L1 归一化**后**） |
| `influence_share` | `Dict[str, float]` | `Σ_frames \|aw_normed × conf × normed_adv\|`，跨通道归一化到和为 1 |
| `dead_frame_ratio` | `float` | `mean(aw_l1_sum <= 1e-12)` |
| `action_dim_grad_norms` | `Optional[np.ndarray]` | `(action_dim,)`，见 3.4 |

**兼容性**：`UpdateStats.to_log_dict()` 需要把这些展平进 `__RAW_STATS__`。加完后用 `analyze_training.py --list-metrics` 验证能被自动发现。`action_dim_grad_norms` 是数组，展平为 `grad_dim_00..NN` 或单独存，**不要**塞进 `stats` 破坏解析。

### 3.4 按动作维度的梯度（策略自有）

框架无法从 `actor.parameters()` 泛化地映射到动作维度（不同策略族结构不同）。按 P5，**由策略提供**：

```python
# TrainablePolicy 新增可选方法
def action_dim_grad_norms(self) -> Optional[np.ndarray]:
    """返回 (action_dim,) 的梯度范数；不支持则返回 None。

    须在 backward() 之后、zero_grad() 之前调用。
    """
    return None
```

`TruncatedNormalPolicy` 实现：取输出层权重梯度按行（动作维）求范数。

**调用时机是硬约束**：`trainer.py` 的 actor `backward()` 之后、`optimizer.step()`/`zero_grad()` 之前。放错位置会得到全零或上一轮的值。

---

## 4. 快照与离线重算

### 4.1 触发（进程内，唯一侵入点）

在 `loop.py:442` 循环顶部、`exploration()` 之前插入一次检查：

```python
debug_req = debug.poll_request(run_dir)   # 无请求时仅一次 os.path.exists
```

**哨兵文件语义**：

- 路径 `<run_dir>/debug_request.json`
- 必填 `hypothesis`（缺失则拒绝并在日志报错——`DEBUG_GUIDE.md` §6 纪律 6 的机制化）
- 读到后立刻 `os.rename` 到 `<run_dir>/debug/u{u:05d}/request.json`（同文件系统，原子）→ 天然一次性消费
- **不用信号**：无法携带参数、多进程投递复杂、不留请求记录

### 4.2 快照内容

```
<run_dir>/debug/u00250/
├── request.json          # 原始请求（含 hypothesis）
├── manifest.json         # update / 时间 / git commit / config / episodes 是否全量
├── episodes/             # EpisodeCollection.save()  ← 组件已存在
├── actor/                # θ_old（可复制 policy_exports/u00250，或直接引用）
├── critics.pt            # ⚠ 必须单独存：policy_exports 不含 critic
├── rng_state.pt          # torch / numpy / cuda RNG 状态
└── replay/               # 离线重算产物（由 replay 写入）
```

**两种模式，语义不同，必须在 manifest 中标明**：

| 模式 | `--episodes N`（默认 8） | `--episodes all` |
|---|---|---|
| 用途 | 逐帧检查、`frame`、`chain` 分位数 | 自校验、`whatif` |
| 聚合量 | **与训练日志不可比**（子集） | 必须与日志逐字段一致 |
| 体积 | ~数 MB | 数百 MB |

子集选取**必须确定性**（取前 N 条），不得用随机采样——否则违反 P1。

### 4.3 离线重算（`replay.py`）

严格复用生产代码（P2）：

```
加载 manifest / EpisodeCollection / actor / critics / rng_state
  → experiment = registry.get(manifest.experiment)（从 config.json 重建）
  → trajectories = experiment.build_trajectories(episodes)        ← 真实调用
  → experiment.debug_arrays(episodes, trajectories)               ← 实验中间量
  → buf = PPOBuffer(trajectories, actor, ...)                     ← 真实调用
  → set_rng_state(rng_state)
  → ppo_update(..., debug_sink=NpzSink(out))                      ← 真实调用
```

**actor/critics 必须深拷贝**，重算不得回写快照（否则重复运行结果不同）。

**自校验不变量**：`--episodes all` 模式下，重算得到的 `UpdateStats` 必须与日志中该 update 的 `__RAW_STATS__` 逐字段一致（浮点容差）。不一致 = 快照不完整或存在未捕获的非确定性，**本身是必须先修的 bug**。这条是整个 L2 可信度的地基。

---

## 5. 声明式扩展点（实验侧）

四个新扩展点，全部**可选、默认空**，不破坏现有实验。

> 这里的静默默认是合法的：「本实验没有声明探针」是完整答案，不是信息缺失。与 `CLAUDE.md` 的 fail-loud 规则不冲突。

### 5.1 `debug_arrays` — 实验中间量

```python
def debug_arrays(self, episodes, trajectories) -> Dict[str, np.ndarray]:
    """逐帧命名数组，供快照落盘。默认 {}。

    契约：第 0 维长度必须等于 trajectories 拼接后的总帧数，
    且顺序一致（即已按轨迹截断，与框架侧数组对齐）。
    """
    return {}
```

**实现纪律**：必须调用与生产同一批 helper（如 `_compute_phase_mask`、`_compute_foot_weights_masked`），**不得重写逻辑**（P2）。`exp_standup_step_v3` 已把这些抽成 staticmethod，可直接复用。

**时间基准**：契约规定「已截断、与轨迹对齐」。实验若在 `T_full` 上计算，需自行切片后返回。这是刻意选择——把对齐责任放在**知道怎么切**的一侧。

### 5.2 `probe_suites` — 行为探针

```python
@dataclass(frozen=True)
class BehaviorProbe:
    name: str
    predicate: Callable[[Episode, str], bool]    # (episode, agent_id) -> 是否通过

@dataclass(frozen=True)
class ProbeSuite:
    name: str
    probes: Tuple[BehaviorProbe, ...]
    seeds: Tuple[int, ...]                        # 固定种子 → 可跨 update 比较
    episode_options: Dict[str, Any]

def probe_suites(self) -> Tuple[ProbeSuite, ...]:
    return ()
```

**设计要点**：

- 谓词签名是 `(Episode, agent_id) -> bool`，因此**可在快照上离线运行、可单元测试**
- 探针 rollout 用 `stochastic=False` + 固定种子 → 无采样噪声、可跨 update 比较
- 探针是回答「链条①环是否断裂」的**唯一直接手段**，不依赖 provenance / sink / 快照
- **推荐纪律**：开训前先写谓词。写不出谓词说明目标没定义清楚

### 5.3 `metric_verifiers` — 指标严格定义

```python
def metric_verifiers(self) -> Dict[str, Callable[[Episode, str], float]]:
    """指标名 → 严格重算函数。供 `debug.py metric --verify` 对比。"""
    return {}
```

对 `exp_standup_step_v3` 应立刻提供 `steps` 的严格版本（要求最小支撑/摆动时长），以证伪当前基于接触翻转的计数。

### 5.4 `knob_checks` — 旋钮通路实测

```python
@dataclass(frozen=True)
class KnobCheck:
    name: str
    configured: Callable[[], Any]
    observed: Callable[[UpdateStats, PPOBuffer], Any]
    agree: Callable[[Any, Any], bool]
```

框架注册内建项：`explore_factor`（`eff_std_mean/std_mean` 比值）、`uncertainty_floor`（floor loss 梯度）、`floor_weight`（buffer 内非 1 占比）、resume（参数指纹）、observer 一致性。实验可追加自己的旋钮。

**目的**：`intervene-check` 不硬编码，新旋钮加了就自动被覆盖——否则这个工具会随框架演进而失效。

---

## 6. CLI 结构

单入口 `baseline/framework/ppo/debug.py`，子命令与 `DEBUG_GUIDE.md` §3 一一对应。

| 子命令 | 依赖 | 数据来源 |
|---|---|---|
| `health` | — | 训练日志（复用 `TrainingLogAnalyzer`） |
| `timeline` | — | 训练日志 |
| `compare` / `noise` | — | 多个训练日志 |
| `probe` | 5.2 | 现场 rollout 或快照 |
| `metric --verify` | 5.3 | eval episodes 或快照 |
| `intervene-check` | 5.4 | 日志 + 可选快照 |
| `snapshot` | 4.1 | 写哨兵文件 |
| `attribute` | 3.3 | 训练日志（新聚合量） |
| `chain` | 3.3 + 快照 | 日志 + 快照（分位数、①环需 probe） |
| `frame` | 3.1 + 4.3 | 快照 replay 产物 |
| `whatif` | 4.3 + `noise` | 快照 replay |

**复用要求**：`health` 必须 `import` 现有 `TrainingLogAnalyzer` 而非重写解析。日志解析器只能有一份。

---

## 7. 实施分期（含依赖与推荐顺序）

依赖关系决定了必须的先后，但**推荐顺序不是依赖拓扑序**——优先能立刻回答当前问题的项。

```
S0 聚合量+不变量 ─────────────┐
S5 探针+指标验证 ─────────────┤（二者互不依赖，可并行）
                              ↓
S1 provenance ──→ S2 sink+快照+replay ──→ S3 chain/attribute/frame ──→ S4 whatif
                                                                        ↑
                                            S6 intervene/compare/noise ─┘
```

| 期 | 内容 | 依赖 | 产出的用户能力 |
|---|---|---|---|
| **S0** | 3.3 四个聚合量；6 条不变量守卫；修 `ppo_update` 返回标注 | 无 | `attribute` 雏形；⑥⑦⑧环可见 |
| **S5** | 5.2 探针 + 5.3 指标验证 | 无 | **①环与⑨环可判定** |
| **S1** | 3.1 provenance（`Trajectory` + `PPOBuffer` + 实验填充） | 无 | 逐帧溯源成为可能 |
| **S2** | 3.2 sink；4.1 哨兵；4.2 快照（含 critics/RNG）；4.3 replay + 自校验 | S1 | 全链路逐帧数据 |
| **S3** | `chain` / `attribute` / `frame`（含 `--where`） | S2 | 九环剖面、帧检查器 |
| **S4** | `whatif`（含 `--sweep`）+ `noise` 基线 | S2, S6 | **离线证伪，打断盲调** |
| **S6** | 5.4 旋钮注册；`compare` / `noise` / `timeline` | 无（S4 用其结果） | 显著性判定、干预验证 |

### 7.1 为什么 S0 + S5 优先于 S1–S4

当前的实际问题是「stepping 学不出来」，而九环表告诉我们**必须先确定断点是否在①环**——因为①环断裂时所有 reward/权重类改动都无效。

- **S5 的探针**直接回答①环（行为是否曾经发生），且**不依赖 provenance、sink、快照中的任何一个**
- **S5 的指标验证**直接回答⑨环（`steps` 是否只是接触抖动）
- **S0 的影响份额与按维度梯度**回答⑥⑦⑧环

也就是说：**S0 + S5 就能定位九环中的断点位置**，而它们是整个计划里依赖最少的两项。S1–S4 是在断点已知之后用来看细节的。

先做 S1–S2 会花掉大量工作却仍不知道断在哪一环。

---

## 8. 对现有代码的改动清单

| 文件 | 改动 | 期 |
|---|---|---|
| `trajectory.py` | `TrajectoryProvenance`；`Trajectory.provenance` | S1 |
| `trainer.py` | `PPOBuffer.seg_provenance` + `frame_id/frame_ids/find_frames` | S1 |
| `trainer.py` | 3.3 聚合量；`action_dim_grad_norms` 调用点；返回标注修正 | S0 |
| `trainer.py` | `ppo_update(debug_sink=None)` + 各 stage 记录点 | S2 |
| `experiment.py` | `UpdateStats` 新字段 + `to_log_dict`；`TrainablePolicy.action_dim_grad_norms`；`ExperimentPPO` 四个可选扩展点 | S0/S2/S5/S6 |
| `loop.py` | 循环顶部 `debug.poll_request`；快照落盘（含 critics/RNG） | S2 |
| `policies/truncated_normal_mlp.py` | `action_dim_grad_norms` 实现 | S0 |
| `debug/`（新目录） | `sink.py` / `snapshot.py` / `replay.py` / `probes.py` / `knobs.py` / `invariants.py` | S0–S6 |
| `debug.py`（新，CLI） | 子命令 | S0–S6 |
| `analyze_training.py` | 抽出可复用入口供 `debug.py health` 导入 | S0 |
| `experiments_ppo/exp_standup_step_v3.py` | 填 `provenance`；实现 `debug_arrays` / `probe_suites` / `metric_verifiers` | S1/S5 |

---

## 9. 风险与对策

| 风险 | 后果 | 对策 |
|---|---|---|
| 调试代码消耗 RNG | **训练轨迹偏离**，观测改变被观测对象 | 独立 `Generator`；快照前后 `get/set_rng_state` 包夹；快照子集**确定性**选取；CI 加一条「开/关调试，训练结果逐位相同」的测试 |
| `debug_arrays` 与生产逻辑漂移 | 调试看到的不是真实计算 | 强制调用同一批 helper；对齐长度断言 |
| 离线重算与线上不一致 | L2 结论不可信 | `--episodes all` 自校验不变量；不一致视为阻塞级 bug |
| 快照缺 critics | 算不出 `V(s)`，④环不可见 | 4.2 明确单独保存（`policy_exports` 只有 actor） |
| 子集快照被误用于聚合比较 | 得出错误结论 | manifest 标注模式；工具在子集模式下**拒绝**输出聚合对比 |
| 磁盘膨胀 | `runs/` 已很大 | 默认 8 episodes；`include_grads` 默认关；目录总量上限 |
| 新增 stats 键破坏日志分析 | `analyze_training.py` 解析失败 | 加完即用 `--list-metrics` 验证；数组类量不进 `stats` |
| sink stage/name 改名 | 所有工具静默失效 | 契约写进 docstring；加一条名称清单测试 |
| 探针 rollout 成本 | 拖慢训练 | 探针默认**不在训练循环内**跑；由 CLI 按需触发 |

---

## 10. 完成判据

每一期都必须能用 `DEBUG_GUIDE.md` 的用户语言验收，而不是「代码写完了」。

| 期 | 验收标准 |
|---|---|
| S0 | 能对当前任一运行中的实验说出「哪个通道占多少影响份额」「哪些关节没有梯度」 |
| S5 | 能对 stepping 给出「历史上是否曾经发生」的是/否结论，并证伪或确认 `steps` 指标 |
| S1 | 任取一个平坦索引，能得到正确的 `ep{N}:{agent}:{t}` |
| S2 | `--episodes all` 快照的重算聚合量与训练日志逐字段一致 |
| S3 | `chain` 能自动指出主断点，并给出「不该做什么」 |
| S4 | 对一个候选改动，能在一分钟内给出「效果是否超过噪声带」的结论 |
| S6 | 任一已配置旋钮都能被自动验证是否进入数据通路 |

---

## 11. 明确不做的事

避免范围蔓延。以下**当前不做**，需要时再单独立项：

- **实时 Web 面板 / TensorBoard 集成**。日志 + CLI 已足够；面板会引入依赖与状态同步复杂度。
- **自动调参 / 自动修复**。本体系只负责**看清**，决策权保留在人手里。自动化建立在错误的因果判断上会加速走错方向。
- **跨实验的历史数据库**。`compare` 直接读日志文件即可。
- **训练中在线的逐帧记录**。违反 P4 与磁盘预算；快照 + 离线重算已覆盖。
- **SAC 侧的对等实现**。先在 PPO 上验证设计，稳定后再考虑抽取共用部分。
