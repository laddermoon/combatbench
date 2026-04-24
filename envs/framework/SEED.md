# Seed Architecture

`combatbench` 所有随机性都从**单一 `base_seed`**派生。本文定义派生规则、
API 与记录约定。

## 两条原则

1. **独立可设**：每个需要随机性的组件（simulator / policy / plugin）必
   须能被独立赋予一个 `int` 种子。
2. **父向子链路通**：父组件能确定性地派生出子组件的种子；这条链路从
   `ParallelRunner` 一直到最叶子的 plugin RNG 全程闭合，不允许有任何
   "自己 `np.random.default_rng()`" 的孤岛。

## 随机性消费者

| 层 | 消费者                     | 数量 | 设置入口                                  |
|----|----------------------------|------|-------------------------------------------|
| 1  | `EnvRuntime.simulator`     | 1    | `runtime.reset(seed=...)`                 |
| 2  | `Policy`（每个 agent）     | N    | `policy.reset(seed=...)`                  |
| 3  | `Plugin`（可选带 RNG）     | K    | `plugin.set_episode_seed(seed)` + `on_pre_episode` |

**K** 运行时确定（取决于挂了哪些插件）。不带 RNG 的 plugin 不需要实现
`set_episode_seed`。

## 派生树（`SeedSequence.spawn` 全程）

```
base_seed : int                       ← 用户给，禁止为 None
    │
    SeedSequence(base_seed)
    │
    .spawn(n_episodes)                ← batch 层
    │
episode_ss[i] : SeedSequence
    │
    .spawn(1 + N_agents + K_plugins)  ← episode 层
    │
┌───┴────────────┬────────────────┬──────────────┐
runtime_ss      policy_ss[a]     plugin_ss[p]    （无 runner-internal，预留）
│               │                 │
to_int(1)       to_int(1)         to_int(1)      ← 叶子取 uint32
│               │                 │
simulator       policy.reset      plugin.set_episode_seed
```

**为什么全程用 `spawn` 而不是 `generate_state(n)`**：
- `spawn` 返回子 `SeedSequence`，子组件可以**继续**往下派生（比如某个
  plugin 内部有 2 个独立 RNG）。这是原则 2（链路通）的硬性要求。
- `generate_state` 返回 `uint32` 是终点值，切断了链路。
- 需要 `int` 时，对子 `SeedSequence` 调 `.generate_state(1, dtype=uint32)[0]`
  取出即可——链路首段保留 `SeedSequence`，叶子取 `int`。

## 禁止事项

1. **不允许 `None` 向下传播**。`run_episode(seed=None)` 与
   `run_n_episodes(base_seed=None)` 在 runner 入口立即解析为一个具体
   `int`（用 `secrets.randbits(32)` 或 `np.random.SeedSequence().entropy`），
   并把解析出的值写回 `EpisodeResult.seed` 与日志。这样**任何 episode
   都是可复现的**，哪怕调用方没显式给种子。
2. **不允许算术推导** `seed + i`, `seed * k`, `hash(seed, i)` 等——引入
   隐性相关。**只用 `SeedSequence.spawn`**。
3. **不允许 `np.random` 全局 RNG**。所有组件各自持有 `Generator` /
   `RandomState` 实例，由接收到的 seed 构造。

## API 契约

### `BasePlugin.set_episode_seed(seed: int) -> None`

默认实现 no-op。持有 RNG 的 plugin 必须实现：**在这个方法里
立即重建 RNG**（不推迟到 `on_pre_episode`）。这样实现最简洁：
`set_episode_seed` 是唯一的 RNG 重建入口，on_pre_episode 只负责 episode
级其他状态的 reset。

```python
def set_episode_seed(self, seed: int) -> None:
    self._rng = np.random.RandomState(int(seed))
```

### `Policy.reset(seed: int) -> None`

已有约定——`EpisodeRunner` 每个 episode 前调用一次，传
per-policy 子种子。实现方如持有 RNG，需用 `seed` 重建。

### `EnvRuntime.reset(seed: int, ...)`

已有约定——透传给 `simulator.reset`。

### `EpisodeRunner` 内部派生函数

```python
def _derive_seeds(self, base_seed: int) -> EpisodeSeeds:
    """Return a structured bundle with concrete int seeds for every consumer."""
```

其中 `EpisodeSeeds` 至少包含：

```python
@dataclass(frozen=True)
class EpisodeSeeds:
    base: int                       # 入口 seed（已解析，非 None）
    runtime: int
    policies: Dict[str, int]        # agent_id -> int
    plugins: Dict[int, int]         # id(plugin) -> int
```

Plugin 字典的 key 用 `id(plugin)` 而非 `plugin.name`：
- 进程内唯一，免去 plugin 名重复时的歧义；
- 该字典仅 runner 内部用于跟踪分发，不落盘、不跨进程传递，因此 id 不
  稳定跨进程的缺点不影响语义；
- 需要可读日志时，在打日志处拼 `f"{type(plugin).__name__}(id={id(plugin)})"`
  即可。

## 记录

### `EpisodeResult`
只记 `base_seed: int`（单个数）。派生规则是确定的 → 只要 base_seed 与
代码版本一致，任何子种子都可重算。

### Recorder 落盘（`episode_manifest.json`）
同理——只写 `base_seed`。Replay 时调用方按同样的 `_derive_seeds` 重算
即可。不记录派生树，避免「派生逻辑改了但老 manifest 仍写着旧派生值」
的误导。

### 运行时日志
`EpisodeRunner` / `ParallelRunner` 入口打一条 INFO：
`"episode base_seed=<int>"`。这是调试可复现性的第一落点。

## `ParallelRunner`

已经与 `EpisodeRunner.run_n_episodes` 派生同样的 batch-level 子种子
（见 `parallel_runner._derive_seeds`）。**此处保持不变**——只需把批内每
个 episode 的 seed 从 `uint32` 改为 `SeedSequence` 在 episode 层 spawn。

## 实现顺序

1. `BasePlugin.set_episode_seed` 默认 no-op + 把 humanoid21 现有
   `RandomPushPlugin` / `InitialStatePerturbationPlugin` 的私有方法提升到基类契约。
2. `EpisodeRunner._derive_seeds` 改为返回 `EpisodeSeeds`，全程 `spawn`；
   `_reset_all` 给所有 seedable plugin 打种子，再 `runtime.reset`，再
   `policy.reset`。入口解析 `None → int`。
3. 给无 seed 的 plugin（`PeriodicUpwardForcePlugin` 等）加 `random_seed`
   ctor 参数 + `set_episode_seed` 实现。
4. 修 `match_runner.py` 的加法 seed → `SeedSequence.spawn`。
5. `EpisodeResult` 保持 `seed` 字段不变（就是 base_seed）；不加
   `derived_seeds`（按原则，子种子可重算，不需持久化）。
6. `Recorder.on_post_episode` 写 `base_seed` 到 manifest。
7. 测试：同 `base_seed` 两次跑出来的所有派生值 byte-equal；缺 seed 的
   `base_seed=None` 路径自动解析为 `int` 并写回 result。
