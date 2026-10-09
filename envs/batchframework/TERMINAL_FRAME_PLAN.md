# 终止帧契约修订 —— 定稿方案

> 类型：记录

> 已定稿语义已于 `082187be` 在 `_RuntimeCore.step` 落地；本文档是
> 外溢修改面的实施计划。

## 已定契约（runtime 层，已落地）

1. `episode_step` = `step()` 调用计数，**无条件 +1/次**。不隐含
   "完整物理步"。
2. `physics_step` = 实际物理子步计数。**动作是否物理生效由帧的
   physics_step 增量判定**，与 episode_step 无关。
3. `on_post_action_step` 对每个**进入的** step() 恰好触发一次，
   作用于该步到达的最终状态（完整步或中途终止态）；终止处理
   （`on_post_episode`）统一在 post_action 之后触发。
4. 记录层忠实不变——recorder 无条件收帧，取舍全部在轨迹层。

## 帧分类与轨迹判定

| 帧 | physics delta | observer 值 | 轨迹处理 |
|---|---|---|---|
| 正常帧 | S | 步末态 post_action 值 | 进 |
| 中途终止帧 | 1..S-1 | **终止态** post_action 值 | 进（终止 transition） |
| post_action 终止帧 | S | 终止态 post_action 值 | 进（同原行为） |
| 退化帧（pre_action/首子步 pre_phy 全员终止） | 0 | 未变状态重算 | **记录在案，轨迹排除** |

关键推论：`episode_step` 无条件递增后，recorder 在 post_action
扫描到的 records 值天然是**含端点边界**（中途终止帧记 i+1，
`frames[:term_step]` 自动含终止帧）。唯一需要额外判定的是
退化帧——靠末帧 physics delta==0 排除。

## 外溢修改面

### W1 — 契约文档与注释收口

- `context.py`：`episode_step`/`physics_step` 字段 docstring 改写
  新定义；
- `plugin.py`：`on_post_action_step` docstring——"每步恰好一次、
  作用于步末状态、不隐含完整步"；插件开发者须知：hook 可见
  `all_agents_terminated=True` 的终止态；
- `observer_plugin.py`：dispatcher 触发条件注释（post_action 现在
  覆盖终止帧——dedup token 已含 physics_step/proposals 天然兼容）；
- `recorder.py` 模块 docstring：原声明"recorders always run after
  dispatcher refreshed"在新语义下**由假变真**，修正表述；
- `CLAUDE.md` hook 表、`DESIGN.md`、`LIFECYCLE_TRACE.md` 同步；
- `AUDIT.md` P-FW-2 条目更新为"契约修订已解决"。

### W2 — Episode 数据面

- `from_buffer_frames`：导出逐帧 `physics_step`（帧 dict 已有该键，
  当前被丢弃）→ `Episode.physics_steps: np.ndarray (T,)`；
- `Episode` 增派生 property `agent_frame_boundary[aid]`：
  `min(records[aid][0].step, 最后物理帧索引+1)`——语义集中在一处，
  records 保持 provenance 不动；
- `save/load` 补 `physics_steps`；旧 npz 无该字段 → 回退为
  "全帧物理"假设（旧语义不可精确重建，如实标注）；
- `episode.py` docstring 切片示例改 boundary。

### W3 — 消费侧切换

- 6 个活跃实验（5 PPO `exp_basic_balance`/`exp_standup`/
  `exp_standup_step_v3`/`exp_step`/`exp_minimal` + 1 SAC
  `exp_sac_balance`）：`T = term_step if fell else T_full` →
  `T = ep.agent_frame_boundary[aid]`；`fell`/`is_terminated`
  语义不动；
- reason-only 消费（dump_capture、reason 查询工具）不动；
- `experiments_ppo/{todo,archive}/` 死代码不改，标注语义已漂。

### W4 — 设备侧对齐

- `ep.episode_steps += run_i64`（本步进入运行的行，无条件）替代
  `post_run`；核对 `action_call_index` 既有语义是否重复；
- `term_history` 记录值对齐 CPU（post_action 扫描时值 = 边界）：
  consume 时刻读值 vs seal 时换算，实现时定；
- `BatchRuntime.step`：`on_post_action_step` 插件对"本步 running
  的行（含中途 ENDED）"触发——mask 语义修订；
- `env_term_step` seal 读 post-increment 值 → 自动变含端点；
- `_RecorderAdapter.on_post_episode` 的末帧覆写变冗余——可删
  （终止帧已有 post_action 终止态值）；
- `episode_exporter`：`num_frames`/`t_use` 语义随 env_term_step
  变化，`agent_frame_boundary` 导出对齐 CPU property。

### W5 — 测试与回归

- `test_audit_terminal_frame.py` 重写为新契约断言（终止帧
  observer=终止态值、records=含端点、episode_step 唯一→文件名
  不撞）；
- `test_wave_contract`/`test_device_lifecycle`/`test_device_rollouter`/
  `test_multi_rollouter` 的 term_step/num_frames 预期更新；
- Episode 构造 fixtures（dump 测试等）补 `physics_steps`；
- CPU + device 冒烟训练，如实记录 reward 记账变化（终止
  transition 计入轨迹后 r_fall/r_cross 尾段变化预期内）。

### W6 — 已知风险项（审阅清单）

- **插件在终止态 post_action 的副作用**：已扫 7 处实现——
  `ScoringPlugin` 获益（KO 补判/终帧 hit event 入账）、push 计数器
  无害（episode 即结束）、`TimeoutPlugin` 在恰好打满 max 的终止帧
  会补录一条 timeout record（排在真 reason 后，良性）；
- **observer 在终止态计算**：终止态可能 NaN/极端——值如实记录，
  退化帧反正不进轨迹；
- **legacy npz**：旧数据无 `physics_steps`，boundary 派生回退旧
  语义，不复原旧切片；
- **post_episode 双刷新**：终止帧 post_action 刷新后 post_episode
  再刷一遍——幂等，可接受。

## 不做

- 不加 done-mask 新字段（`is_terminated`/reason 已足够）；
- 不改 `term_records` 既有 provenance 形状；
- archive/todo 实验不迁移；
- zero-frame episode 维持拒绝。

## 实施落地记录（未提交，待 review）

W1–W5 已全部落地。与计划相比的实现决定：

- **`action_call_index` 复用为"本步 1-based 帧序号"**：步首即增
  （`entered_i64`），`_archive_reason` 与 `seal_rows` 统一读它——
  步内任意提议点都等于 CPU 记录帧扫描到的 `episode_step`；
  `episode_steps` 仍在步尾无条件自增（步内读到上一步值，
  CPU parity）。
- **新增 pre-action 终止屏障**：`on_pre_batch_step` 后
  `_consume_terminations()`——pre_action 全员终止的行不跑物理，
  产生物理增量=0 的退化帧（CPU 同语义）。
- **`physics_steps`/`time` 改逐子步记账**：`on_post_phy_step` invoke
  前按 running mask 累加（CPU：physics_step++ 在终止检查前）；
  `_pre` 回调内也补了 consume——pre_phy 提议的行不再多跑一个
  子步（旧实现此处漏屏障，pre_phy 提议会被延迟一个子步才封存）。
- **`_RecorderAdapter.on_post_episode` 删掉 `_t-1` 覆写**：旧写法
  在子步屏障内 post_episode 先于 post_action 触发时会把值写到前一
  帧（且 `_t=0` 时越界写到 T-1）；新契约下终止帧值=post_action
  刷新值，post_action 写入已正确。
- **RecordStore 新增 `phys_steps (T,B)` 列**，exporter 导为
  `Episode.physics_steps`；`np_bufs` 缺 `phys` 键时回退 None
  （测试/手写缓冲兼容）。
- **序列化**：`physics_steps` 作为 v3 npz 的可选新增键——旧文件
  无该键 → load 为 None → boundary 回退"全帧物理"假设；format
  version 不升（双向兼容：旧代码 load 新文件会忽略多余键）。

验证：

- 258 项（framework + wave/device lifecycle/runtime/rollouter）+
  66 项（wave/lifecycle 重跑）全过；717 项大回归中仅
  `test_s1_provenance` 两个失败为本改动引入（SimpleNamespace mock
  缺 `agent_frame_boundary`，已修）；其余 5 个失败在干净树上
  复现（`test_trainer` resume_ctx/`test_critic_mlp` 模块缺失/
  viewer collection error——既有问题）。
- CPU 端到端核验：mid-physics KO → num_frames=2、records=(ko,2)、
  physics_steps=[10,15]、boundary=2；pre_action 全员终止 →
  num_frames=1、physics_steps=[0]、boundary=0（退化帧排除）。
