# BLACKBOARD_DESIGN — 设备侧 metrics/events 通道设计（审计 G1/G2）

> 类型：记录

**状态**：设计稿（未实现）。解决 BATCHFRAMEWORK_AUDIT 的两个高优先级
缺口：`ctx.metrics` 与 `ctx.events` 在 `DeviceCtx` 无对应物，阻塞
scoring/damage/事件驱动类实验迁移。

**设计原则**（与既有框架一致）：
- **名称对齐 CPU**：暴露 `ctx.metrics` / `ctx.events`——迁移代码改动
  面最小化（dict→张量池的差异是语义必然，不是命名差异）。
- **声明式**：容量/键名先声明后使用，与 `declare_state`/`output_schema`/
  `term_history_k` 同族；溢出显式 flag，不静默截断。
- **行级生命周期**：与 episode 簿记同生死——partial reset 清零行并
  递增 epoch（对齐 CPU `EventJournal._reset` 语义）。
- **可审计**：读/写纳入 declared_reads/writes 族声明面。

---

## 1. G1 — `ctx.metrics`：共享标量黑板

### 1.1 模型

CPU 语义：`Dict[str, Any]`，插件写、observer 同帧读、跨 hook 持久到
episode 边界、`clear_episode_state` 清空。

设备化：**键→张量的声明式共享池**，存于 `state.board`：

```python
state.declare_shared(key, shape, dtype, init=0.0, owner=unit.name)
    → (B, *shape) Tensor      # 冲突声明（同 key 异型）→ ValueError
ctx.metrics                      # MetricsView（dict 风格只读映射）
    ["hp"]                       → (B,) Tensor 本体（写= .copy_/.fill_/行掩码）
    ["nope"]                     → KeyError（未声明）
```

- **写 = 原位张量操作**——`ctx.metrics` 返回张量本体而非副本；
  没有 `__setitem__`（设备侧"赋值"就是行掩码原位写）。
- **读无声明亦可**（与 declared_reads 同哲学：声明是审计增强非门禁）；
  推荐但非强制声明 `shared_reads`/`shared_writes` 类属性。
- **多写者**：允许（CPU 同语义），顺序由 priority 保证——文档注明
  同键多写是排序敏感用法。

### 1.2 生命周期

```
declare_shared（attach/declare_state 阶段）→ 分配 (B,*shape)
reset_episode_rows / reset_plugin_rows 同位 → reset_shared_rows(ids)
    rows 填 init 值
```

### 1.3 导出

`export_episode_metrics` 通道不变——插件从 `ctx.metrics` 读取共享值
汇入自己的指标表；**不做隐式全池导出**（键空间不归单一单元所有，
隐式导出会污染 Episode.metrics 归属语义）。可选增强：`declare_shared`
加 `export=True` 让 runtime 代导——列入开放问题 §4.1。

### 1.4 迁移映射

| CPU | 设备 |
|---|---|
| `ctx.metrics["hp"] = x` | `ctx.metrics["hp"][ids] = x` / `.fill_` / `.copy_` |
| `ctx.metrics.get("dmg", 0)` | `ctx.metrics["dmg"]`（未声明 KeyError——改用 try 或先声明） |
| dict/嵌套结构 | **拆成多个 (B,*shape) 张量键**——无任意对象容器 |
| `metrics.clear()` | 框架所有——插件无权清（与 events 同约束） |

---

## 2. G2 — `ctx.events`：事件池

### 2.1 模型

CPU 语义（EventJournal，2026-10 落地）：append-only；episode 内只增；
`since(mark)` 游标差分；`(epoch, len)` 游标；框架独占 `_reset`
（epoch++）。事件是任意 Python 对象。

设备化：**容量声明的 per-row padded journal** + 类型 code 注册表——
`term_history` 机制的同构泛化（该机制已验证）： 

```python
# state.events —— EventNamespace（episode 簿记族，runtime 拥有）
records  (B, K, 4) f32   # [code, agent, value, aux]——定长数值记录
steps    (B, K) i32      # action_call_index 快照（帧归属）
count    (B,) i32        # 本 epoch 已写入条数
epoch    (B,) i32        # 行 reset 递增（游标失效检测）
overflow (B,) bool       # 超 K 显式截断标志
event_registry: Dict[str, int]  # 事件类型字符串→确定性 code
```

### 2.2 API

```python
# 生产者（插件）
ctx.events.emit(env_ids, "hit", agent=0, value=dmg, aux=0)
    # 向量化：每调用每行追加一条；超 K 置 overflow 不写入（不丢旧）
    # code 未注册 → event_registry 首见序分配 ≥0

# 消费者（observer/插件，同帧或跨帧差分）
ctx.events.since(env_ids, marks) → (records_view, valid_mask)
    # marks (M,) i32：上次 len 快照；返回 mark..count 段的 padded 视图
ctx.events.epoch / .len(env_ids)   # 游标组成
```

- **epoch 游标语义照搬 CPU**：消费者存 `(epoch, len)`；`epoch` 变
  → journal 被框架重置 → 整段重取（mark 作废）。
- **emit 时间戳**：自动写 `action_call_index`——与 term_history
  归档同一边界语义（含端点）。
- **append-only 的设备实现**：API 只暴露 emit/since/len/epoch；
  无 remove/clear——行复位由 `reset_events_rows` 在框架 reset
  路径执行（epoch++）。
- **payload 定长**：`[code, agent, value, aux]`——任意对象不可表达，
  这是设备化必然损失；str→code registry 保住可读性（导出反查）。

### 2.3 容量

`events_cap`（K）——EpisodeNamespace 级参数，默认建议 **64**
（term_history 的 8 是 reason 专用；通用事件含高频 hit 类，K 需
余量；overflow flag 保证超限可观测不静默）。

### 2.4 与 term_history 的关系

term_history 是**终止事件的专用归档**（含端点 step、去重规则特殊）。
不合并——events 池是通用通道；终止提议继续走 term_history（语义已
稳定且有去重规则）。将来 events 池落地后 term_history 可作为其
"订阅者"重构，**非本次范围**。

### 2.5 导出

CPU 侧 events 不进 Episode（ctx-only，observer 同帧消费）。设备
保持同语义——**不导出**。需要事件进轨迹数据的实验另行声明
（走 reward_channels/observer outputs 通道）。

---

## 3. 实现面（落地清单，供执行时核对）

| 件 | 改动 |
|---|---|
| `device_state.py` | `state.board`（shared dict 池）+ `declare_shared`/`reset_shared_rows`；`state.events`（EventNamespace 六字段 + registry）；reset 路径挂接两者 |
| `device_plugin.py` | `ctx.metrics`（MetricsView）、`ctx.events`（EventPoolView，emit/since/len/epoch）；声明面加 `shared_reads`/`shared_writes`（可选审计） |
| `device_runtime.py` | `_invoke` 无改动（ctx 视图常设）；reset/reset_rows 序列插入 shared+events 行复位（在 reset_plugin_rows 同位） |
| `episode_exporter.py` | 无改动（不导出） |
| 测试 | FakeBackend 契约用例：声明/冲突/行复位/epoch 失效/overflow/emit-since 差分；`test_wave_contract` 补断言 |
| 文档 | DESIGN §3 命名空间表 + SEMANTICS §2.3/§4 + MIGRATION_GUIDE §0 阻塞表更新 |

## 4. 开放问题

1. **`export=True` 自动导出** shared 键到 Episode.metrics——方便 vs
   归属语义清晰，倾向不做（由 export_episode_metrics 显式导）。
2. **events 高频写入性能**：hit 类事件每步每行可能多条——emit
   每调用每行一条的 API 是否够用？（备选：emit_batch 接受
   (M,K2) 压缩段。）先用单条 emit，profile 后按需扩。
3. **aux 语义**：第二个 f32 payload 够不够？（hit={agent,value}够；
   复杂事件要 (geom pair, pos)——3 维位置可考虑 aux 升维 (…,6)
   但多数事件用不到。先 4 定长，超纲事件拆多条 emit。）
4. **G4 顺带**：`ctx.episode_options`——options 在 reset 已知，给
   ctx 加只读 options 视图（(B,dict)→张量化白名单键）成本低，
   建议随本包一起做（一次 reset 路径改动覆盖两个缺口）。
