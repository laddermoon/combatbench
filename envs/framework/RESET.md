# Reset Architecture

本文定义一个 episode 开始时，`reset` 如何在 **runner → runtime → simulator
→ plugins → observers → recorders → policies** 之间传导。配套文档：
`SEED.md`（seed 派生）、`plugin.md`（hook 权限）。

---

## 1. 目标

- `reset(seed, options)` 是进入一个新 episode 的**唯一入口**。调用返回后，
  整个 runtime 必须处于"第 0 步可 `step`"的干净状态。
- 所有**随机性消费者**按 `SEED.md` 拿到派生种子。
- 所有**每-episode 可变参数**（课程化扰动强度、对手快照、初始 HP、初始
  姿态、…）通过 `options` 一个通道传导，不再由 caller 反复重建 runtime
  / plugin 实例来"带参数"。
- 传导顺序可见、可测；任何一环读不到它该读到的东西 = bug。

---

## 2. 参与者与 reset 接口

| 层 | 类型 | 接口 | 谁负责调用 |
|---|---|---|---|
| L0 | `BaseSimulator` | `reset(seed: int, options: dict \| None)` | `_RuntimeCore.reset` |
| L0+ | `SimContext` | `clear_episode_state()` | `_RuntimeCore.reset` |
| L1 | `BasePlugin` | `set_episode_seed(seed: int)`<br>`on_pre_episode(ctx)`<br>`on_post_episode(ctx)` | `EpisodeRunner` / `_RuntimeCore` |
| L1 | `BaseObserverPlugin` | `on_reset(ctx_ro)`<br>`on_post_episode(ctx_ro)` | `_ObserverDispatcherPlugin`（内部） |
| L2 | `PostActionRecorder` | `on_pre_episode(ctx_ro, obs)`<br>`on_post_episode(ctx_ro, obs)` | `EnvRuntime._invoke_recorders` |
| L3 | `Policy` | `reset(seed: int)` | `EpisodeRunner._reset_all` |
| L4 | `EnvRuntime` | `reset(seed, options)` | 外部 |
| L4 | `EpisodeRunner` | `run_episode(seed, options)` | 外部 |

**关键约定**：`seed` 沿 L4 → L3 / L1 / L0 派生下发（见 `SEED.md`）；
`options` 沿 L4 → L0 透传 + 经由 `ctx.episode_options` 供 L1 / L1-obs /
L2 读取。

---

## 3. 规范的 reset 调用链

从外到内、按时间顺序。**❗** 标出的是**当前实现与本规范不一致的地方**，
需在后续 patch 中对齐。

### 3.1 外部入口 — `EpisodeRunner.run_episode(seed, options)`

```
1) base_seed = _resolve_seed(seed)                      # None → concrete int
2) episode_seeds = _derive_seeds(base_seed)             # SeedSequence.spawn 全程
3) for plugin in seedable_plugins:
       plugin.set_episode_seed(episode_seeds.plugins[id(plugin)])   # 重建 RNG
4) runtime.ctx.base_seed = base_seed                    # 给 recorder 读
5) runtime.reset(seed=episode_seeds.runtime,
                 options=options)                       # ❗ 当前 `options` 没透传
6) for agent, policy in policies:
       policy.reset(episode_seeds.policies[agent])
```

**步骤 3 必须发生在 `runtime.reset` 之前**（亦即 `on_pre_episode` 之前）——
持有 RNG 的 plugin 在 `on_pre_episode` 里就会用 RNG 采样初值，此时 RNG 必
须已按本 episode 重建。

### 3.2 `EnvRuntime.reset(seed, options)` → `_RuntimeCore.reset`

```
1) ctx.clear_episode_state()
     - episode_step = 0, physics_step = 0
     - metrics.clear(), events.clear(), termination_proposals.clear()
     - episode_options.clear()                          # ❗ 当前 ctx 无此字段
     - base_seed 保留不动                                # 由上层在 runtime.reset 之前写入
2) ctx.episode_options = dict(options or {})           # ❗ 当前无此步骤
3) _is_episode_active = True
4) simulator.reset(seed=seed, options=options)         # backend 消费 sim 相关 key
5) plugin_manager.invoke("on_pre_episode", ctx, allow_mutator=True)
     - 内置观察者派发器（priority=1e6）先跑：对每个 observer 调
       observer.on_reset(ReadOnlySimContext(ctx))
     - 其他 plugin 按 priority 递减顺序跑 on_pre_episode；
       可读 ctx.episode_options 取本 episode 参数，可读 ctx.metrics /
       ctx.accessor，可写 ctx.metrics / ctx.events / ctx.mutator（若
       require_mutator=True）
6) if ctx.is_terminated: _handle_termination()         # 立即触发 on_post_episode
```

### 3.3 `EnvRuntime.reset` 的收尾（recorder 阶段）

```
7) _invoke_recorders("on_pre_episode")                 # 读 ReadOnlySimContext
     - 此时 observer 已刷新、plugin metrics 已初始化；
     - recorder 可以拍一张"episode 起始快照"并据此写 manifest。
8) 如 6) 已触发终止 → 顺带 _invoke_recorders("on_post_episode")
```

### 3.4 顺序不变式

- `simulator.reset`（backend 写初态） **先于** plugin `on_pre_episode`
  （plugin 读初态）。
- observer `on_reset` **先于** 其他 plugin 的 `on_pre_episode`（由
  dispatcher 的 priority=1_000_000 保证）。后果：其他 plugin 在
  `on_pre_episode` 里写的 `ctx.metrics`，observer 在本步**看不到**——
  需要看到 → 移到 `on_post_action_step`，不要指望 on_reset。
- recorder `on_pre_episode` **晚于** 所有 plugin 的 `on_pre_episode`。
  recorder 看到的是 plugin 已处理完的完整 `ctx`。
- `policy.reset` 在 `runtime.reset` **返回之后**。policy 不参与 runtime
  生命周期，reset 只用于重置策略自身的 RNG / LSTM state。

---

## 4. `options` 通道语义

### 4.1 数据形状

单一 `dict[str, Any]`，在 `runtime.reset` 入口被**原封不动**地：

- 传给 `simulator.reset(seed, options)`——backend 按自己的 schema 取需要的 key；
- 写到 `ctx.episode_options`——plugin / observer / recorder 只读地访问。

**不做 schema 强校验**。key 命名由约定治理；未知 key 静默忽略。

### 4.2 现有已知 key（按 backend / 插件）

| Key | 消费者 | 含义 |
|---|---|---|
| `initial_distance` | `humanoid21.Humanoid21Simulator` | 两机器人初始水平距离 (m) |
| `initial_pose_a` / `initial_pose_b` | 同上 | `"standing"` / `"squat"` / `"prone"` / ... |
| `episode` | `ReplaySimulator` | 跳到第 N 个录像 episode |
| *（新增）* `initial_health_a` / `initial_health_b` | `CombatScoringPlugin` | HP 延续 ❗ 当前靠重建 runtime 实现 |
| *（新增）* `push_force` | `CurriculumPushPlugin`（示例 03） | 本 episode 的扰动力大小 |
| *（新增）* `opponent_snapshot_id` | `OpponentPoolPlugin`（未实现） | 对手池里拉哪份快照 |

后三个示例说明：**课程化 / 对手池 / HP 延续**都应当走 `options`，而不是
"每回合拆掉 runtime 重建一个"。这是本次改造最重要的收益。

### 4.3 插件消费 `options` 的写法

```python
class CurriculumPushPlugin(BasePlugin):
    def on_pre_episode(self, ctx):
        self._push_force = float(
            ctx.episode_options.get("push_force", 0.0)
        )
```

**一律在 `on_pre_episode` 里读**；不要在 `set_episode_seed` 里读——
那个钩子只负责 RNG 重建。

---

## 5. `seed` 通道语义

参见 `SEED.md`。本节仅记录它在 reset 链路上的落点：

| 消费者 | 接收入口 | 由谁赋值 |
|---|---|---|
| `simulator` | `simulator.reset(seed, options)` 的 `seed` | `_RuntimeCore.reset` |
| plugin | `plugin.set_episode_seed(seed)` | `EpisodeRunner._reset_all` |
| policy | `policy.reset(seed)` | `EpisodeRunner._reset_all` |
| recorder | `ctx.base_seed`（只读） | `EpisodeRunner._reset_all` 写入 ctx |

裸用 `EnvRuntime.reset`（不经 `EpisodeRunner`）时，plugin 和 policy 的
`set_episode_seed` / `reset` 不会被自动调用；caller 需要自己负责
——这也是 `SEED.md` 把 `EpisodeRunner` 定义为**唯一正确的 seed 派生点**的理由。

---

## 6. 钩子生命周期名字对照

**一件事三个名字**是当前代码里最容易迷惑的地方，先摆清楚：

| 事件 | `BasePlugin` | `BaseObserverPlugin` | `PostActionRecorder` |
|---|---|---|---|
| episode 开始 | `on_pre_episode(ctx)` | `on_reset(ctx_ro)` *(经 dispatcher)* | `on_pre_episode(ctx_ro, obs)` |
| 每 action step 末 | `on_post_action_step(ctx)` | `on_post_step(ctx_ro)` *(经 dispatcher)* | `on_post_action_step(ctx_ro, obs)` |
| episode 结束 | `on_post_episode(ctx)` | `on_post_episode(ctx_ro)` *(经 dispatcher)* | `on_post_episode(ctx_ro, obs)` |

**名字不统一是历史遗留**，短期内保留兼容；长期应当统一为
`on_pre_episode / on_post_step / on_post_episode` 三个名字，dispatcher
透明转发。此处不做强制，但新插件建议按 `BasePlugin` 命名实现，以便将来
一次性收口。

---

## 7. 当前实现 vs 本规范的 gap（改动清单）

> **状态（2026-04-26）**：G1–G6 已全部落地；§9 计划完成。不变式 I1–I6
> + G4/G5 的回归测试见
> `@/data1/mono/things/combatbench/envs/framework/tests/test_reset_chain.py`。
> 下面保留每条 gap 的原始描述与改法，作为审计与回滚依据。

按影响面由大到小：

### G1. `options` 没有透传到 `EpisodeRunner` — ✅ 已落地

`@/data1/mono/things/combatbench/envs/framework/episode_runner.py:582`
调用 `self.runtime.reset(seed=seeds.runtime)` 时没传 `options`。

**改法**：
- `EpisodeRunner.run_episode` 增加 `options: dict | None = None` 入参；
- `_reset_all` 接收并传给 `runtime.reset`；
- `run_n_episodes` 增加 `options_fn: Callable[[int], dict] | None`（可选，
  每个 episode 派发不同 options——课程化的关键入口）；
- `ParallelRunner` 对应增加同名入参，走 pickle 通道（要求 options 可
  picklable，由 caller 保证）。

### G2. `ctx.episode_options` 字段不存在 — ✅ 已落地

`@/data1/mono/things/combatbench/envs/framework/context.py:171`
`SimContext.__init__` 没有 `episode_options`；`clear_episode_state()`
也没有清它。

**改法**：
- `SimContext.episode_options: Dict[str, Any] = {}`；
- `clear_episode_state` 里加 `self.episode_options.clear()`；
- `_RuntimeCore.reset` 在 `clear_episode_state()` 之后、`simulator.reset`
  之前写 `ctx.episode_options = dict(options or {})`。

### G3. `MatchRunner` 用"重建 runtime"模拟 HP 延续 — ✅ 已落地

`@/data1/mono/things/combatbench/envs/framework/match_runner.py:166`
每回合 `self.runtime_factory(initial_health_a=..., initial_health_b=...)`
重建整个 runtime（含 simulator、MuJoCo 模型、plugin 实例）。

**改法（G1/G2 落地后自然可行）**：
- `runtime_factory` 改成只接受**一次**（`env_factory()` 无参）；
- `MatchRunner` 循环内只做 `runtime.reset(seed=..., options={"initial_health_a": hp_a, "initial_health_b": hp_b})`；
- `CombatScoringPlugin.on_pre_episode` 从 `ctx.episode_options` 读 HP，
  不再从 ctor 读（ctor 值作为默认）。

收益：一场 6 回合比赛从 6× MuJoCo cold-start 变成 6× reset（快 ≈ 10 倍），
也消除"配置藏在 factory 闭包里"的问题。

### G4. `reset` 在 episode 进行中被再次调用 → 静默丢弃当前 episode — ✅ 已落地

`_RuntimeCore.reset` 没有检查 `_is_episode_active`。若上一个 episode
还没 `terminated/truncated` 就 `reset`，`on_post_episode` **不会触发**，
recorder 写一半的 manifest 直接失效。

**改法**：
- 如 `_is_episode_active` 为 True，直接 `_handle_termination(reason="abandoned")`
  再继续 reset。理由：rerun / IPython 环境下这很常见，静默修复比报错友好；
  但要显式记录终止原因以便排查。
- 或者加严格模式 `strict_reset=True` raise。**选前者作默认。**

### G5. `base_seed` 在 `clear_episode_state` 里没被清，行为微妙 — ✅ 已落地

裸用 `EnvRuntime.reset`（不走 EpisodeRunner）时，`ctx.base_seed` 可能
保留上一次 EpisodeRunner 跑过的值，误导 recorder。

**改法**：把 `base_seed` 的所有权明确为"由 reset 的 caller 负责"——
`clear_episode_state` 里清掉（设为 `None`）；`EpisodeRunner` 在 `runtime.reset`
**之前**写入（当前就是这样，只需确保 clear 不再保留）。

### G6. `on_reset` / `on_pre_episode` 命名不一致（见 §6） — ✅ 已统一

Observer 侧的 `on_reset` / `on_post_step` 已重命名为
`on_pre_episode` / `on_post_action_step`，与 Plugin 侧一致（无
向后兼容）。详见 `@/data1/mono/things/combatbench/envs/framework/observer_plugin.py`。

---

## 8. 不变式（供测试固化）

写成测试时，以下断言都应恒成立：

- **I1**：`reset` 返回后 `ctx.episode_step == 0 and ctx.physics_step == 0`。
- **I2**：`reset` 返回后 `ctx.termination_proposals == []`，除非
  `on_pre_episode` 里显式 `request_termination`（此时 episode 立即终止
  并已触发 `on_post_episode`，外部看到 `is_episode_active == False`）。
- **I3**：同一 `base_seed` 两次调用 `EpisodeRunner.run_episode(seed, options)`，
  所有 plugin RNG / simulator RNG / policy RNG 按位一致；trajectory
  bit-equal。
- **I4**：`options` 里的 key 在 plugin 的 `on_pre_episode` 里 via
  `ctx.episode_options[key]` 可见；在 observer 的 `on_pre_episode` 里 via
  `ctx.episode_options[key]` **也**可见（只读）。
- **I5**：`observer.on_pre_episode` 的调用时机早于**所有**非 dispatcher plugin
  的 `on_pre_episode`。
- **I6**：`recorder.on_pre_episode` 的调用时机晚于所有 plugin 的
  `on_pre_episode`；此时 `ctx.metrics` / observer outputs 都是"episode
  起始快照"。

---

## 9. 落地顺序（建议）

1. **G2 + G5**：`ctx.episode_options` 字段 + `base_seed` 归属收紧。
   （纯加字段 + 清理，零行为变更，先落。）
2. **G1**：`EpisodeRunner.run_episode(options=...)` + `run_n_episodes(options_fn=...)`
   + `ParallelRunner` 对应入参。
3. **G4**：mid-episode reset 的"优雅终止" 处理，加不变式 I1/I2 的测试。
4. **G3**：`MatchRunner` 重构——`env_factory` 无参 + 循环内改 reset。
   `CombatScoringPlugin` 从 `ctx.episode_options` 读 HP。
5. 给 `examples/03_training_aids.py` 的 `CurriculumPushPlugin` 换成
   从 `ctx.episode_options` 读 `push_force`，作为对外 demo。

改动全部落地后，策略构建层规划文档里的 **R1（per-episode params 注入）**
前置需求就自然消解了。
