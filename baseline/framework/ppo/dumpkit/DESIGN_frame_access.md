# DESIGN_frame_access — Dump 数据访问层设计（帧表 / FrameTable）

> 状态：**讨论稿 v1**（未实现）
> 关联：`DATA_FLOW.md`（数据流阶段语义）、`CONTEXT.md`（能力地图）、
> `viewer/server.py::DumpData`（将被本层取代的现有懒加载器）

## 1. 背景与动机

### 1.1 触发案例

step 实验 u50 dump 的根因诊断需要回答："在 `h_foot>3cm` 且 `φ≥0.9`
的帧上，各通道 `aw×adv` 贡献各是多少？"——现有命令面（inspect /
samples / trace / timeline / metrics）没有任何一个能表达这个
**条件化跨通道聚合**。最终靠 ~150 行 ad-hoc numpy 脚本完成，脚本里
80% 的代码在做 dump 里本应已解决的 join 与对齐。

### 1.2 现状问题：知识分散 + 重复实现

对 dump 数据的"访问知识"（哪个字段在哪个文件、扁平索引怎么切、
traj↔episode 怎么换算、dict-of-array 怎么拆、旧格式怎么兼容）目前
**散落在 5 处手写代码里**：

| 消费方 | 访问方式 | 重复了什么 |
|---|---|---|
| `viewer/server.py` DumpData + ~10 endpoint | 文件级懒加载 + 手写切片/join | seg_offsets、traj_map 反查、observer 双格式兼容、`f"reward.{ch}"` 拼 key |
| `dump_analysis.py`（4 个纯函数） | 经 `dd.*_npz` 访问 | `_dict_item` 拆 object-array、frame_map join、逐字段装配 |
| `dump_delta.py` | **绕过 DumpData 直接 np.load** | episode_frame_offsets 切片、traj_map agent 查找 |
| `dump_render.py` | **绕过 DumpData 直接 np.load** | episodes.npz 种子/obs/actions 直取 |
| 分析脚本（本次） | 直接 np.load + 手工 join | traj_map→ep 索引重建、φ 掩码、双空间对齐 |

两个框架内模块**绕过抽象直接读盘**，是抽象不足的直接证据。

### 1.3 目标

一个**统一的懒加载帧表访问层**，成为 dump 数据的唯一读取入口：

- server.py、dump_analysis.py、delta、render、debug.py、未来分析
  脚本全部建立在其上；
- 列访问、切片、join、兼容、派生列 —— 每类知识只有一个实现；
- 高级诊断（L3）以"派生列注册"方式外挂，不进框架本体。

## 2. 需求清单

来源 A = 现有消费方代码归纳；来源 B = 本次诊断的 ad-hoc 分析。

### 2.1 结构访问需求（来源 A：现有消费方）

| # | 需求 | 现状出处 |
|---|---|---|
| REQ-1 | **懒加载 + 缓存**：npz 文件/列首访问才读，读后驻留 | `DumpData._load_npz` |
| REQ-2 | **元数据**：manifest、traj_map、channel_names、agent_ids、observer_keys | DumpData properties |
| REQ-3 | **trajectory 切片**：`seg_offsets[i]:[i+1]` 同时切 ~15 列 | `_traj_overview` |
| REQ-4 | **episode 切片**：`ep_offsets[pos]:[pos+1]` | `_episode_frame`、delta、render |
| REQ-5 | **traj→ep 溯源**：traj_idx → (ep_pos, agent_id, t_start) | `_find_traj_provenance` |
| REQ-6 | **ep→traj 正向 + 帧换算**：ep frame → traj frame (`f - t_start`) | `_episode_frame` |
| REQ-7 | **单帧跨阶段装配**：buffer_idx → 6 个文件的同行数据 | `trace_frame` |
| REQ-8 | **dict-of-array 拆包**：gae/combine 的 object-dict → {ch: col} | `_dict_item` ×6 处 |
| REQ-9 | **字符串 key 规范化**：`reward.{ch}`、`obs.{aid}`、`observer_outputs.{o}.{f}`、`new_value.{e}.{ch}`、`ratio.{e}` | 全部消费方散写 |
| REQ-10 | **后向兼容**：`traj_lengths|ep_lengths`、observer flat|object-array、traj_map.json|frame_id 重建 | server.py 三处 fallback |
| REQ-11 | **gradsig→帧 join**：sampled_idx → frame_id → traj_map | `_frame_map` |
| REQ-12 | **timeline 独立行空间**（400 mb 步，非帧） | timeline_npz |
| REQ-13 | **epoch_frames 访问**：ratio.{e}/clip_mask.{e}/new_value.{e}.{ch}（帧行空间，按 epoch 分层） | `_traj_epoch_*` |
| REQ-14 | **派生产物路径**：record/ PNG、delta/ npz 存在性 | `image_path`/`episode_rendered` |
| REQ-15 | **JSON-safe 输出**：标量/数组 → list/float/bool | `_arr_to_list` 各处 |

### 2.2 分析需求（来源 B：ad-hoc 分析归纳）

| # | 需求 | 案例 |
|---|---|---|
| REQ-16 | **全帧条件聚合**：任意列布尔 mask → 任意列聚合 | veto 诊断（h>3cm & φ≥0.9 → contrib 分解） |
| REQ-17 | **跨通道贡献分解**：`contrib_c = aw_normed_c × conf_c × normed_adv_c` 作为一阶派生列 | 发现 r_potential 否决 |
| REQ-18 | **observer↔traj join**：episode 空间的 observer 列投影到 traj 帧空间 | φ 掩码、脚高列对齐 aw/adv |
| REQ-19 | **前瞻事件统计**：帧 i 满足 A → [i, i+H) 内是否满足 B（不跨 traj 边界） | +W 意图→15 帧内真抬脚转化率 |
| REQ-20 | **实验自定义派生列**：站立门、步态相位等注册即用 | `standing = φ_trail≥0.9` |

### 2.3 明确划界（非本层需求）

- run 级时间序列（`__RAW_STATS__`）—— `metrics`/`summary` 已覆盖；
- dump **写入**路径 —— dump_capture.py 不变，磁盘格式不变；
- 交互式可视化 —— viewer 是本层的消费者，不是本层的一部分。

## 3. 设计原则

**P1 列存语义**：`Table` ≈ `dict[str, ndarray]`。所有查询表达力交给
numpy 布尔 mask（where）与 numpy/pandas 聚合 —— **不自研 DSL**。

**P2 懒加载粒度 = 列**：npz 成员本身可按 key 解压，持有 `NpzFile`
句柄、`col()` 首访问才读对应 member 并缓存。**不做行级懒加载**
（逐帧解压会性能抖动；单列最大 ~160MB 的 obs 按需整列读可接受）。

**P3 单一 schema 源**：列名 → (文件, key, 行空间, 变换) 的映射是
一张集中注册的 SCHEMA 表。通道列用 `{ch}` 模板，channel_names 从
trajectories.npz 动态展开。所有后向兼容分支收进 SCHEMA 解析处
——消费方代码里不再出现兼容逻辑。

**P4 L3 外挂**：`register_column(name, fn)` 让实验/脚本注册派生列
（contrib、standing gate、步态相位…）。dumpkit 自带最少派生列
（contrib），其余属于各实验的脚本或 `param_overrides` 级扩展。

**P5 只读**：本层永远只读；派生产物（record/delta）保持现有目录约定，
访问层只提供路径/存在性查询，不生成。

## 4. 数据模型

### 4.1 三个行空间

| 表 | 行数（本 dump） | 行含义 | 主要列来源 |
|---|---|---|---|
| `ds.frames` | 409600 | **canonical**：traj-帧（1024 traj × 400） | trajectories/buffer/gae/combine/epoch_frames + observer 投影 |
| `ds.episodes` | 204800 | ep-帧（512 ep × 400，双 agent 并列） | episodes.npz 原生（obs/actions/observer_outputs） |
| `ds.timeline` | 400 | minibatch 步 | timeline.npz |
| `ds.gradsig` | 2000 | 梯度采样帧（带 `sampled_idx` 回连 frames） | gradsig.npz |

**为什么 canonical 是 traj 帧**：训练的一切目标量（adv/aw/contrib）
都定义在这个空间；observer 列经 join 投影过来后，REQ-16/17 的
"条件×通道"分析变成纯列运算。

### 4.2 join 关系（一次性预计算）

```
traj_map.json:  traj_idx → (ep_pos, agent_id, t_start)
ep_flat_idx[traj_frame] = ep_offsets[ep_pos] + t_start + t_in_traj
                        → (409600,) int32，构建一次
observer 列访问 = ep_col[ep_flat_idx]          # gather
frame_id "flat:N"（obs 匹配失败的 traj）→ ep_flat_idx = -1 → 列值 NaN
```

gradsig 表：`sampled_idx` 直接索引 frames 行（2000 → 409600 回连）。

### 4.3 列命名规范（公开 schema）

模板 `{ch}` 按 channel_names 展开，`{e}` 按 n_epochs 展开：

| 公开列名 | 来源 | 行空间 |
|---|---|---|
| `reward.{ch}` `aw.{ch}` `terminated.{ch}` | trajectories.npz `reward.*`/`actor_weight.*`/`is_terminated.*` | frames |
| `adv.{ch}` `value.{ch}` `ret.{ch}` | gae.npz dict 键 | frames |
| `normed_adv.{ch}` `aw_normed.{ch}` `key_frame_mask.{ch}` `key_aw.{ch}` | combine.npz dict 键 | frames |
| `combined_adv` `aw_l1_sum` | combine.npz | frames |
| `obs` `action` `log_prob` `uncertainty` `sample_weight` `ef` `floor_weight` `frame_id` | buffer.npz / trajectories.npz | frames |
| `observer.{o}.{f}` | episodes.npz `observer_outputs.{o}.{f}` → join | frames |
| `epoch.{e}.ratio` `epoch.{e}.clip_mask` `epoch.{e}.log_prob` `epoch.{e}.value.{ch}` | epoch_frames.npz | frames |
| `contrib.{ch}` | **派生**：`aw_normed × conf × normed_adv` | frames |
| `obs.{aid}` `actions.{aid}` `ef.{aid}` `observer.{o}.{f}` `seed` `termination` | episodes.npz | episodes |
| `kl` `ratio_mean` `dtheta_norm` …（原生名） | timeline.npz | timeline |
| `proj` `cos` `grad_norm` `w_adv` `floor_pen` `sampled_idx` + join 列 | gradsig.npz | gradsig |

命名取舍：通道字段统一 `aw.`（actor_weight）/`adv.`/`value.`/`ret.`/
`normed_adv.`/`aw_normed.`/`contrib.` 前缀族 —— 一眼可辨"哪个量、
哪个通道"；原始文件 key 只在 SCHEMA 表内出现一次。

## 5. API 设计

```python
ds = DumpDataset("baseline/runs/<run>/dumps/u00050")   # 或 <run>:u50 简写解析

# ---- 元数据 ----
ds.manifest            # dict
ds.traj_map            # list[dict]（json 或 frame_id 重建，REQ-10 内化）
ds.channel_names       # ["r_potential", "r_left_foot", "r_right_foot"]
ds.agent_ids           # ["robot_a", "robot_b"]
ds.n_frames            # 409600

# ---- 列访问（懒加载，REQ-1/8/9）----
f = ds.frames                       # FrameTable
f.columns                           # 全部可解析公开列名
f["adv.r_left_foot"]                # np.ndarray (409600,) — 首访问才读 gae.npz
f["contrib.r_left_foot"]            # 派生列，首访问才计算
f["observer.standing_balance_a.potential"]
                                    # episodes.npz 列经 ep_flat_idx gather

# ---- 掩码/切片（REQ-16）----
stand = f["observer.standing_balance_a.potential"] >= 0.9
sub = f[stand]                      # FrameTable 视图，同列集
sub["contrib.r_potential"].mean()

# ---- traj/episode 视图（REQ-3/4/5/6）----
f.traj_slice(i)                     # (start, end) — 取代手写 seg_offsets
f.traj(i)                           # FrameTable 视图 = f[start:end]
ds.episodes[i]                      # EpisodeView：obs/actions/observer 于 ep 空间
ds.provenance(traj_idx)             # (ep_pos, agent_id, t_start)
ds.trajs_of_episode(ep_pos)         # [{traj_idx, agent_id, t_start, length}]
f.ep_flat_idx                       # join 列本体（调试可见）

# ---- 单帧跨阶段（REQ-7）----
f.row(buf_idx)                      # dict：该行全部已加载列（默认轻列，
                                    # obs/action 等大列需显式 include）

# ---- 独立行空间 ----
ds.timeline["kl"]                   # (400,)
ds.gradsig["proj"]                  # (2000,) + "frame_idx"/"traj_idx" join 列

# ---- epoch 分层（REQ-13）----
f["epoch.0.ratio"]                  # (409600,)
ds.n_epochs

# ---- 派生列注册（REQ-20，L3 外挂）----
ds.register_column("standing",
    lambda t: t["observer.standing_balance_a.potential"] >= 0.9)
f["standing"]                       # 此后与内建列同权

# ---- 派生产物（REQ-14）----
ds.record_dir / ds.episode_rendered(pos) / ds.image_path(pos, frame)
ds.delta_dir(ep_pos)

# ---- JSON 序列化（REQ-15）----
DumpDataset.to_jsonable(v)          # ndarray/scalar/bool → list/float/None
```

**显式不提供**：`where(...)` 链式 DSL、SQL 式聚合、行对象 ORM、
写接口。要迭代就用 `for col in f.cols([...])` 或直接 numpy。

## 6. 逐需求推演

**REQ-3 traj 切片**（`_traj_overview` 现状：手写 seg_offsets + 15 处
`[start:end]` + 6 处 `_dict_item`）

```python
t = ds.frames.traj(i)
t["reward.r_left_foot"]   # (T,) — 一行替代原 ~80 行装配代码
t.cols(["adv.r_potential","contrib.r_potential"])
```

**REQ-4/6 episode 帧装配**（`_episode_frame` 现状：offsets 换算 +
observer 双格式分支 + enriched_trajs 手工 join）

```python
ev = ds.episodes[ep_pos]
ev.col("observer.foot_state_a.h_left_foot")[frame]
for tinfo in ds.trajs_of_episode(ep_pos):
    tf = frame - tinfo["t_start"]
    if 0 <= tf < tinfo["length"]:
        row = ds.frames.traj(tinfo["traj_idx"]).row(tf)
        row["reward.r_potential"], row["aw.r_potential"]  # 已对齐
```

**REQ-7 单帧跨阶段**（`trace_frame` 现状：6 文件手工装配）

```python
row = ds.frames.row(buf_idx)   # 内部按 buf_idx 统一索引各文件列
# gae/combine/epoch 列与 buffer 同行空间 —— 天然对齐
ds.gradsig.row_by_frame(buf_idx)  # sampled_idx 反查
```

**REQ-10 后向兼容**：全部进 SCHEMA 解析 —
`traj_lengths|ep_lengths` 双名、observer flat/object-array 归一化为
flat 再进表、traj_map 缺失时 frame_id 重建。消费方零感知。

**REQ-11 gradsig join**

```python
g = ds.gradsig
g["traj_idx"], g["frame_in_traj"]   # join 列由 sampled_idx+traj_map 预生成
g[g["proj"] < 0]["contrib.r_left_foot"]  # join 回 frames 列亦可
```

**REQ-16 条件聚合**（veto 诊断 → 3 行）

```python
m = (ds.frames["observer.foot_state_a.h_left_foot"] > 0.03) & \
    (ds.frames["observer.standing_balance_a.potential"] >= 0.9)
sub = ds.frames[m]
sub["contrib.r_potential"].mean(), sub["contrib.r_left_foot"].mean()
```

**REQ-17 contrib 分解**：`contrib.{ch}` 内建派生列，定义
`aw_normed[ch] × conf[ch] × normed_adv[ch]`（conf 为 per-traj 标量广播）。

**REQ-18 observer join**：列名 `observer.{o}.{f}` 透明 gather；
ep 空间原名保留 `observer.standing_balance_a.*`（_a/_b 后缀含
agent 语义），另有 `observer.<traj_agent>` 语义等价。

**REQ-19 前瞻统计**：提供 traj 边界安全的工具函数

```python
hit = ds.frames.shift_within_traj("observer.foot_state_a.h_left_foot",
                                  k=15, reduce="max") > 0.03
conv = (hit & (f["aw.r_left_foot"] > 0)).mean()
```

**REQ-20 派生列**：见 §5 `register_column`。实验级诊断（如步态相位）
写成 `exp_step.py` 里的注册函数，dumpkit 无感知。

**REQ-12/13**：timeline/epoch 为独立表/模板列，API 同构。

**REQ-14/15**：路径与序列化是薄工具方法。

## 7. 迁移计划

| 阶段 | 内容 | 验证 |
|---|---|---|
| P0 | 新模块 `dumpkit/frame_access.py`：DumpDataset + 3 张表 + SCHEMA + join + 懒加载 | 单测：列解析、双格式 observer、flat: frame_id、traj 切片、contrib |
| P1 | `dump_analysis.py` 内部改调 ds（函数签名不变） | CLI/HTTP 输出逐字节不变 |
| P2 | `server.py` 端点逐个改 ds；DumpData 保留为薄 shim 委托 ds，最后删 | endpoint 对拍 |
| P3 | `dump_delta.py`/`dump_render.py` 接入 ds | render 校验流程不变 |
| P4 | （可选）`debug.py frames` 单命令：谓词+聚合 | — |
| P5 | （可选）SCHEMA 常量反哺 dump_capture 写侧，schema 单源化 | — |

每个阶段独立可交付，P1-P3 是纯内部重构（对外行为不变）。

## 8. 非目标

- 不做查询 DSL / ORM / 流式接口
- 不做行级懒加载、不做稀疏列、不做增量写
- 不把实验诊断逻辑（步态相位、转化率…）收进 dumpkit ——
  `register_column` 是唯一接口
- 不改磁盘格式、不动 dump_capture 写侧（P5 另议）
- 不替代 run 级 metrics/summary 时序层

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 181M episodes.npz 全列解压慢 | 列级懒加载；obs/actions 等重列不进默认 row()，显式 include |
| gae/combine object-dict 无法按列懒拆 | 整 dict 一次性加载（~12M），可接受 |
| 老 dump 格式漂移 | 兼容分支集中在 SCHEMA，加测试夹具固定行为 |
| 命名规范争议 | 公开列名表写进 CONTEXT.md 唯一信息源，不扩散 |
| `flat:` 无 join 的 traj 帧 | ep_flat_idx=-1 → observer 列 NaN，文档明示 |
| Table 抽象被误用为逐行循环 | API 只暴露列与 mask；row() 标注 debug 用途 |
