# Rollout 模块详细设计

> 作用域：`baseline/common/rollout/` —— 为 humanoid21 1v1 搏击项目提供
> 训练侧的 episode 数据采集 / 序列化 / 并行 rollout 抽象。
>
> 上游依赖：`envs/framework/` 已有的 `EnvBlueprint`、`EnvRuntime`、
> `EpisodeRunner`、`PostActionRecorder`、`EpisodeBufferRecorder`、
> `BaseFrameRecorder`、SEED 协议。
>
> 本文档的目标：在 `redesign.md` 草案基础上把每个组件落到可实现的接口、
> 数据结构和决策粒度，并指出原思路里需要再确认的几处问题。

---

## 0. 对原始思路的几点提醒（Push-back）

原思路（`redesign.md`）方向是对的：把 `EnvBlueprint` 当 rollout 唯一入口、
通过更换 `Recorder` 区分训练 / 调试、把 `EpisodeCollection` 作为训练
边界。但有几个隐含假设需要在落地前先明确，否则会走回老路：

1. **"Rollout 和调试只有 Recorder 不同" 不完全成立。**
   除了 Recorder，至少还有两处会差：
   - **policy 模式**：训练时通常需要 stochastic + `want_extras=True`
     （拿 `log_prob`/`value`），调试 / 复现时往往希望 deterministic、
     不带 extras。这是 policy 的属性，不是 recorder 的属性。
   - **debug plugins**：视频录制 (`VideoRecorderPlugin`)、扰动注入
     等 `BLUEPRINT_EXCLUDE=True` 的插件不属于 blueprint，但属于
     "这次跑要不要"。
   建议把这条结论改写成：**`EnvBlueprint` + 种子完全决定环境与转移；
   policy 的 deterministic 开关 + 附加 debug plugins + recorder 决定
   "这次跑产出什么"。** 这样训练 / 调试的差异可以精确归约到三件事。

2. **Policy 不能直接作为 ParallelRollouter 的输入。**
   `Policy` 实例几乎都不可 pickle（持有 GPU tensor、文件句柄等）。
   现行 `baseline/common/rollout/collector.py` 的做法是传
   `policy_factories: Mapping[agent_id, Callable[[], Policy]]` +
   每轮广播 `state_dict`，这是必须保留的设计。`redesign.md` 写的
   "输入 ... Policy" 应该改为 "policy factory + 可选权重快照"。

3. **EpisodeCollection 的"全量序列化"会迅速失控。**
   Humanoid21 单步 `derived_state` 序列化后约 1-2 MB（已在
   `recorder.py` 模块文档里有警告）。一个 1000 episode × 500 step
   的训练批次若直接存所有访问器输出会到 ~TB 级。
   所以 EpisodeCollection 必须**显式区分两种用途**：
   - **训练态（轻量）**：只存 `observation`、`action`、`action_extras`、
     `observer_outputs`、`termination`、`base_seed`。
   - **调试 / 复现态（重）**：完整 `core_state` / `derived_state` /
     image —— 这条路应该走 `BaseFrameRecorder`，**不要**再让
     `EpisodeRecorder` 重复实现。

4. **Episode 不是裸 list[dict]，应当是带 schema 的数据类。**
   `EpisodeBufferRecorder.get_episode_data()` 现在返回的就是裸 dict，
   足够当 buffer，但作为跨进程 / 落盘 / 训练 batch 的边界还需要：
   显式字段、shape 校验、惰性数组化、空 episode 处理。

5. **EpisodeCollection 的"管理多个 episode"要明确 in-memory 还是 lazy。**
   训练时常常 episode 数大但每条不长（GRPO group size），全部 in-memory
   OK；离线 dataset 场景（几十万 episode）必须是 lazy / streaming。
   建议先做 in-memory 版（简单清楚），但磁盘格式从一开始就 episode-per-file，
   方便后续不动 API 升级到 lazy。

下面的设计基于以上五点修正。

---

## 1. 模块边界图

```
┌────────────────────────────────────────────────────────────────────┐
│ envs/framework/  (已有, 不动)                                       │
│   EnvBlueprint  ─── build() ──▶  EnvRuntime ◀── attach ── Recorder │
│   EpisodeRunner (驱动单 episode)                                    │
│   EpisodeBufferRecorder (in-memory 原始 buffer, 通用)               │
│   BaseFrameRecorder    (磁盘 image+JSON, 调试 / 复现专用)           │
└──────────────┬─────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────┐
│ baseline/common/rollout/  (本模块)                                  │
│                                                                    │
│   Episode              ── 单 episode 的强类型数据对象                │
│   EpisodeCollection    ── 多 episode 容器, save/load                │
│   EpisodeRecorder      ── PostActionRecorder, 每集生成 1 个 Episode │
│   ParallelRollouter    ── blueprint × policy_factories × seeds     │
│                          ──▶ EpisodeCollection                     │
└────────────────────────────────────────────────────────────────────┘
```

- 训练侧（PPO / GRPO）：`ParallelRollouter.collect(seeds)` →
  `EpisodeCollection` → trainer 自己解释 reward / advantage。
- 调试侧：`python -m envs.framework.round_runner --blueprint X.yaml
  --recorder envs.framework.recorder:BaseFrameRecorder?output_dir=...`
  → 磁盘录像 + JSON，再用 `recorder_viewer` 看。
- **二者共享 blueprint + seed**，差异精确落在 "Recorder + policy 模式 +
  debug plugins"，符合 §0.1 的修正。

---

## 2. 数据模型

### 2.1 `Episode` —— 单集数据

```python
@dataclass(frozen=True)
class Episode:
    # ---- 元信息 ----
    base_seed: int                       # 来自 ctx.base_seed, 必填
    episode_index: int                   # 在所属 collection 中的下标
    blueprint_hash: str                  # EnvBlueprint 内容哈希, 校验一致性
    num_frames: int                      # = len(frames)
    termination_proposals: Tuple[str, ...]  # 终态原因(末帧的 ctx 快照)
    is_terminated: bool                  # 是否提前终止 (vs 截断)
    episode_options: Mapping[str, Any]   # ctx.episode_options 快照, 可空

    # ---- 时序数据 (按 agent_id 分桶, 每个值是 (T, *) ndarray 或 (T,) list) ----
    # 约定: 任意 agent_id 下的所有数组第一维都等于 num_frames.
    # 同一 episode 下不同 agent 的 T 必须一致.
    observations:    Mapping[str, np.ndarray]            # {agent_id: (T, obs_dim)}
    actions:         Mapping[str, np.ndarray]            # {agent_id: (T, act_dim)}
    action_extras:   Mapping[str, Mapping[str, np.ndarray]]
                     # {agent_id: {"log_prob": (T,), "value": (T,), ...}}
                     # 全部 None 时整个 agent 槽位允许缺失
    observer_outputs: Mapping[str, Any]
                     # {observer_name: 时间堆叠后的结构}, 见 §2.1.1

    # ---- 末态附加 (非时序) ----
    final_observation: Mapping[str, np.ndarray]          # bootstrap 用 obs_{T+1}
```

**字段约定（强制）**：

- `T = num_frames` 是"动作步数"，不含初始 reset 帧（与
  `EpisodeBufferRecorder.on_post_action_step` 帧定义一致）。
- `final_observation` 单独存一份 `obs_{T+1}`（即末步动作之后的下一步 obs），
  RL bootstrap 必需；这一项 **必须显式获取并存储**，
  因为 EpisodeRunner 的循环在终止时不会再读 obs。
  →  这意味着 `EpisodeRecorder` 比 `EpisodeBufferRecorder` 多做一件事：
  在 `on_post_episode` 时再读一次 `runtime.get_observation()`。
- 任何 `action_extras` 子键缺失（policy 没产生）→ 整个键不出现，**不**
  填 NaN，避免训练侧误把 NaN 当样本。
- `blueprint_hash`：用 `hash(json.dumps(blueprint.to_dict(), sort_keys=True))`
  级别的稳定 hash；用于 `EpisodeCollection` 拒绝混入异构 blueprint。

#### 2.1.1 `observer_outputs` 时序化策略

观察器输出可能是任意嵌套结构（dict of ndarray of dict ...）。
两种处理方式：

| 策略 | 优点 | 缺点 |
|------|------|------|
| (A) 不动结构，逐键 stack 叶子 ndarray | 训练侧拿到 `(T, *)` 直接用 | 嵌套深时 stack 易出错 |
| (B) 整体存 `list[dict]`（每帧一个 dict） | 实现最简单，零假设 | 训练侧每次都要再 stack |

**决策：默认 (A)，提供 (B) 作为 fallback。**
具体规则：

1. 若所有帧在同一路径下都是 `np.ndarray` 且 shape 一致 → stack 成
   `(T, *original_shape)`。
2. 若 dtype/shape 在帧间不一致或非 ndarray → 退化为 `list[原值]`，长度为 T。
3. 这一逻辑封装在 `Episode.from_buffer_frames(...)` 工厂里，
   不污染 trainer。

### 2.2 `EpisodeCollection` —— 多集容器

```python
class EpisodeCollection:
    """In-memory, 顺序访问 + 随机访问. 可 save/load."""

    # ---- 构造 ----
    def __init__(
        self,
        blueprint: EnvBlueprint,        # 所有 episode 共享; 写入 metadata
        episodes: Sequence[Episode] = (),
    ): ...

    # ---- 容器接口 ----
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Episode: ...
    def __iter__(self) -> Iterator[Episode]: ...
    def append(self, episode: Episode) -> None: ...
    def extend(self, episodes: Iterable[Episode]) -> None: ...

    # ---- 元信息 ----
    @property
    def blueprint(self) -> EnvBlueprint: ...
    @property
    def total_frames(self) -> int: ...                  # sum of num_frames

    # ---- 训练侧便利 (不强制使用) ----
    def stack_field(self, getter, axis=0) -> np.ndarray:
        """跨 episode 堆叠某字段, 用于 PPO 的 (N*T, ...) flatten."""
    def split_by_termination(self) -> tuple["EpisodeCollection",
                                            "EpisodeCollection"]:
        """按 is_terminated / 截断切两份, 调试 KO 率等指标用."""

    # ---- 持久化 ----
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "EpisodeCollection": ...
```

**append 的一致性校验**：每次 `append/extend` 必须验证
`episode.blueprint_hash == self.blueprint_hash`，否则抛错。
这是这个 collection 唯一硬约束 —— "一个 collection 对应一份 blueprint"。

### 2.3 序列化格式

> 目标：人类可粗看（meta 可读），数组紧凑高效，episode 之间互相独立
> （未来可改 lazy 不破坏 API）。

落盘为目录（不是单个 zip，方便流式增量写、断点续传）：

```
<path>/                                  ← EpisodeCollection.save(path) 产物
  collection.json                        ← 全局元数据 (见下)
  blueprint.yaml                         ← EnvBlueprint, 直接 EnvBlueprint.save
  episodes/
    episode_00000.npz                    ← 单集二进制
    episode_00001.npz
    ...
```

- `collection.json` 含：`format_version`, `blueprint_hash`,
  `num_episodes`, `total_frames`, 创建时间，框架版本等。
- 每个 `episode_*.npz` 用 `np.savez_compressed`，键名采用
  扁平化路径，例如：
  - `meta__base_seed`, `meta__num_frames`, `meta__termination_proposals`,
  - `obs__robot_a`, `obs__robot_b`,
  - `act__robot_a`, `act__robot_b`,
  - `extras__robot_a__log_prob`, ...,
  - `obs_outputs__robot_a_reward__total`, ...
  - `final_obs__robot_a`, `final_obs__robot_b`.
- **不使用 pickle**（跨版本不安全 + 混入任意 Python 对象）。
- `episode_options` 这种自由 dict → 落进每集的 `meta__options.json`
  作为附加文本（罕用、值多样、不值得二进制）。

> **拒绝的替代方案**：
> - **单个 .pkl**：跨 numpy/torch 版本翻车风险，且无法增量写。
> - **HDF5 / Zarr**：依赖重，monorepo 暂时没引入。
> - **Parquet**：表格化对嵌套 observer_outputs 难展开。
> 等真的有 100 万 episode 量级训练再换 zarr，目前 npz 足够。

---

## 3. EpisodeRecorder

### 3.1 角色定位

`EpisodeRecorder` 是 `PostActionRecorder` 子类，**唯一职责**：
在每个 episode 结束时把 framework 收到的原始数据组装成 1 个
`Episode` 对象并通过 `get_last_episode()` 暴露。

参照 `envs/framework/recorder.py:EpisodeBufferRecorder`，但有三处变化：

| 项 | EpisodeBufferRecorder | EpisodeRecorder (本模块) |
|----|----------------------|--------------------------|
| 输出 | 裸 dict（`get_episode_data()`） | `Episode` 强类型对象 |
| `obs_{T+1}` | 不存（buffer 语义） | 在 `on_post_episode` 显式读 |
| 跨 episode 持有 | 仅最近一集 | 同样仅最近一集（队列由外层维护） |
| 数组堆叠 | 不堆叠（每帧一份） | `Episode` 内部自动 stack |
| blueprint_hash | 无 | 在 `attach` 时由外层注入 |

### 3.2 接口

```python
class EpisodeRecorder(PostActionRecorder):
    def __init__(
        self,
        blueprint_hash: str,
        observer_names_to_keep: Optional[Sequence[str]] = None,
        snapshot_arrays: bool = True,
    ) -> None: ...

    # 单集生命周期 (复用 EpisodeBufferRecorder 的逻辑, 但末尾多读 final obs)
    def on_pre_episode(self, ctx) -> None: ...
    def on_post_action_step(self, ctx, observation, action,
                            observer_outputs, action_extras=None) -> None: ...
    def on_post_episode(self, ctx) -> None: ...

    # 输出
    def get_last_episode(self) -> Episode: ...
```

**关键实现细节**：

- `observer_names_to_keep`：白名单，默认 `None` = 全留。训练时传
  `["robot_a_reward", "robot_b_reward", "robot_a_obs", "robot_b_obs"]`
  之类，避免把 KO/scoring 等大体量 observer 全塞进每集。
- `final_observation` 在 `on_post_episode` 通过 `ctx.accessor.get_observation()`
  再读一次。注意：`ReadOnlySimContext` 是否暴露这个调用要先确认；如果
  不行，需要 `EpisodeRecorder` 在 `on_attach` 时拿到 runtime 的弱引用，
  或扩展 ctx —— **这是设计中的一个待定项**，见 §6.
- 不复用 `EpisodeBufferRecorder` 的代码（继承会让两者紧耦合），而是
  把帧 snapshot 的 `_snapshot()` 抽成 module-level 函数共用。

---

## 4. ParallelRollouter

### 4.1 输入 / 输出

```python
class ParallelRollouter:
    def __init__(
        self,
        blueprint: EnvBlueprint,                        # 必填, 唯一 env 来源
        policy_factories: Mapping[str, Callable[[], Policy]],
                                                        # {agent_id: factory}
        num_workers: int = 1,                           # <=1 走 in-process
        observer_names_to_keep: Optional[Sequence[str]] = None,
        deterministic: bool = False,                    # policy.act 模式
        debug_plugins_factory: Optional[Callable[[], Sequence[BasePlugin]]] = None,
                                                        # 仅 BLUEPRINT_EXCLUDE=True 的
    ) -> None: ...

    def collect(
        self,
        seeds: Sequence[int],                           # 每个 seed = 一个 episode
        policy_state_dicts: Optional[Mapping[str, Any]] = None,
                                                        # 可选热更新
        episode_options_fn: Optional[Callable[[int], Mapping[str, Any]]] = None,
                                                        # 第 i 个 seed -> options
    ) -> EpisodeCollection: ...

    def close(self) -> None: ...
```

**说明**：

- **输入是 seeds 列表，不是 episode 数。** 显式 seed 让 collection 可
  按需重放；训练侧调用方负责 `seeds = rng.integers(...)`。
- `policy_state_dicts` 是热更新通道：worker 进程持久存活，每次
  `collect()` 把新权重广播过去（CPU 端 detached tensor），避免反复
  rebuild policy / 重连 GPU。**这是 §0.2 明确保留的能力。**
- `debug_plugins_factory` 限制只能产 `BLUEPRINT_EXCLUDE=True` 的
  插件，跟 `EnvBlueprint.build` 的契约一致。
- `num_workers <= 1` 直接在主进程跑，便于 pdb 调试 —— 这条原则
  借鉴当前 `RolloutCollector` 的实现。

### 4.2 Worker 职责（每个进程）

每个 worker 启动时执行一次：
1. `runtime = blueprint.build(recorders=[EpisodeRecorder(...)],
   debug_plugins=debug_plugins_factory() or ())`
2. `policies = {a: factories[a]() for a in factories}`
3. `runner = EpisodeRunner(runtime, policies)`

每次任务（一个 seed）：
1. 若有新 `state_dict`，调用 `policy.load_state_dict(...)` / 等价方法。
2. `runner.run_episode(seed=seed, options=options_fn(idx))`.
3. 取 `recorder.get_last_episode()` 通过 pickle 回主进程。
4. **不要**在 worker 里维护跨 episode 的累积，避免 worker 崩溃丢数据。

### 4.3 进程通信

- 主进程 → worker：seeds（int），可选 state_dict（CPU tensor /
  ndarray dict）。
- worker → 主进程：`Episode`（数组 + 基本类型，pickle 安全）。
- 用 `multiprocessing.Pool.imap_unordered` 拿到顺序无关的结果再
  按 `episode_index = seed 在输入列表中的位置` 重排。
- 如有 worker 崩溃：默认 `strict=True` 整批失败抛出（行为对齐
  obsolete 的 `ParallelRunner`）。

### 4.4 与现行 `RolloutCollector` 的关系

现行 `baseline/common/rollout/collector.py` 里的 `RolloutCollector`
功能比 `ParallelRollouter` 多（observer 绑定、`as_rollout_batch`
转换）。**短期路线**：

- 新建 `ParallelRollouter` + `Episode`/`EpisodeCollection`，定位为
  "项目无关的通用层"。
- `RolloutCollector` 保持不动，作为 PPO/GRPO 训练的胶水层；其
  内部把"调 ParallelRollouter 拿 EpisodeCollection → 拆 reward
  observer → 装 RolloutBatch" 这条链路接起来。
- 等多家训练脚本都迁过来后再决定是否合并。

---

## 5. 与外部世界的接合

### 5.1 调试 / 录像
```bash
python -m envs.framework.round_runner \
  --blueprint envs/humanoid21/rule_blueprint.yaml \
  --policy-a ... --policy-b ... \
  --video out.mp4 \
  --recorder envs.framework.recorder:BaseFrameRecorder?output_dir=debug/run01&save_accessor_state=true
```
→ 走 `BaseFrameRecorder`，磁盘可直接喂 `recorder_viewer`。
**不动 rollout 模块。**

### 5.2 离线训练数据复用
```python
collection = EpisodeCollection.load("data/run_2026_05_20")
for ep in collection:
    advantages = compute_gae(ep.observer_outputs["robot_a_reward"]["total"], ...)
    ...
```
→ 训练脚本不再依赖活跃 simulator，纯数据迭代，便于在 CPU 节点做
ablation / 离线 eval。

### 5.3 Rollout ↔ 调试 一致性
- 同一 `blueprint` + 同一 `seed` + 同一 policy 权重 →
  无论用 `EpisodeRecorder` 还是 `BaseFrameRecorder`，每帧的
  observation/action 必须 bit-equal。
- 这条做一个回归测试 (`tests/test_rollout_consistency.py`)：
  跑同一个 blueprint+seed 两次，一次拿 Episode，一次拿
  BaseFrameRecorder 的 step JSON，对齐 obs/action 完全相等。
- 该测试是这套设计 "训练 / 调试同源" 主张的唯一物证，**不能省**。

---

## 6. 待定 / 需要确认的问题

1. **`ReadOnlySimContext.accessor` 是否在 `on_post_episode` 时仍可读
   `get_observation()`？** 若不行，`EpisodeRecorder` 拿
   `final_observation` 的方式要改（最坏情况：让外层 caller 在
   `runner.run_episode` 后立刻读一次，再注入到 episode 里 —— 这
   会污染 EpisodeRunner 的 "无返回值" 协议，需要权衡）。
2. **`Episode` 是否需要存 `physics_step` 序列？** 当前训练不用，
   但调试有用。建议作为可选字段（`include_physics_steps=True`）。
3. **多 controlled agent 之外，单 agent + scripted opponent 怎么写？**
   `EpisodeRunner` 需要两个 agent 的 obs 名都有；`policy_factories`
   可以传 scripted policy（已是 `Policy` 子类，不是问题），但
   `Episode.observations` 是否要存 `robot_b` 的也要约定 ——
   建议**默认全存**，训练侧自己挑要哪一个 agent 的轨迹。
4. **`EpisodeCollection.save` 在 1000 集×500 步 的实际尺寸？**
   先做 in-memory 估算，超过 ~5GB 再考虑 lazy。
5. **是否需要 schema version？** 建议从 `format_version=1` 开始，
   load 时严格校验，未来加字段走 minor bump。

---

## 7. 实施顺序建议

1. `Episode` + `EpisodeCollection` + npz save/load + 一组 unit test。
   （纯数据，无 framework 依赖，最容易写。）
2. `EpisodeRecorder` + 一组 framework 集成 test
   （单进程，跑 humanoid21 random policy 1 集）。
3. `ParallelRollouter` 单进程版（`num_workers=1` 路径）。
4. `ParallelRollouter` 多进程版 + `state_dict` 热更新。
5. 训练 / 调试一致性回归 test (§5.3)。
6. 把 `RolloutCollector` 的内部改造为 `ParallelRollouter` 包装。

每一步都应当 commit + push 后再进下一步（按项目规则）。
