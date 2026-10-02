# 终止帧保留方案（P-FW-2 的契约级修复）

> 背景：AUDIT.md P-FW-2 报告子步内终止帧的 observer 输出陈旧 +
> recorder 文件名碰撞。核对后发现更深一层的问题：**终止 transition
> 被整体丢弃**——`term_step = episode_step`（未递增）使该帧被
> `frames[:term_step]` 排除，"动作导致终止"的 transition 不进入轨迹。

## 问题判定

子步内终止帧四要素齐全：`obs_t`（动作前观测）、`action_t`
（`get_action()` 执行值）、`s_{t+1}`（终止时刻物理状态，良定义）、
`terminated=True` 及完整 reason。它与 post_action 终止帧在 MDP 语义上
**无差别**，但二者边界判定不同：

| 提出时机 | episode_step 递增？ | term_step | 帧是否进轨迹 |
|---|---|---|---|
| post_action（Timeout 等） | 已递增 (i→i+1) | i+1 | ✅ 含 |
| 子步内（KO/imbalance） | 未递增 (=i) | i | ❌ 丢 |

边界结果取决于 `episode_step += 1` 写在物理循环之后这一**实现位置**
——是实现事故，不是语义设计。后果：导致终止的动作拿不到
reward credit（KO 奖励/摔倒惩罚缺最后一档信号），是系统性偏差。

**决定：终止帧保留，作为该 agent 轨迹的最后 transition。**

## 新契约（修订 D7 生命周期定义）

1. **帧边界含端点**：`frame_boundary[agent]` = 该 agent 首个
   termination proposal **所在帧索引 + 1**。轨迹消费 =
   `frames[:frame_boundary]`。该规则统一覆盖全部提出点
   （pre_episode/pre_action/子步前后/post_action），无需按 phase
   分支；env 未终止的 agent boundary = 本波帧数。
   - `term_records` 保持 `(reason, episode_step_at_proposal)` 不变
     ——降为 provenance，不再充当轨迹边界。
   - `num_frames` = 记录的帧数（含终止帧）。
   - zero-frame episode 维持拒绝（`on_pre_episode` 即终止 → 无帧可记）。
2. **终止帧 observer 输出 = 终止语义下最后刷新点的值**：
   - env 级终止帧（全 agent terminated）→ post_episode 刷新值；
   - 行内终止帧（A 死 B 活，env 续跑）→ 正常 post_action 刷新值
     （该帧本就完整走完）。
   - reward 类 observer 若要给终止 transition 计 reward，输出必须在
     `on_post_episode` 产生——这是单元的明确语义要求（见 W3 核对项）。
3. **记录帧 metadata**：终止帧 JSON/字段标 `partial_terminal`
   （首 proposal 在物理循环内提出）供 dump/viewer 区分。

## 工作包

### W1 — CPU 侧边界与帧

- `EpisodeRecorder`：每帧扫描 proposals 时同时记
  `frame_boundary[agent] = 当前帧索引 + 1`（首见 reason 帧）。
- `Episode` 增 `agent_frame_boundary: Mapping[str, int]`；num_frames
  含终止帧；docstring 更新 term_step→boundary 语义。
- 消费侧：`build_trajectories`/`extract_per_step_*` 改
  `[:frame_boundary]`（timeout/post_action 场景结果不变，
  子步内终止多含一帧）。
- `BaseFrameRecorder`：文件名改单调帧号（`step_{i:05d}`，
  `_frame_counter` 在 on_pre_episode 归零）；episode_step 写进
  JSON 不变；`partial_terminal` 标记；docstring 修正为
  "终止帧 observer 输出对应 post_episode 刷新点"。

### W2 — 设备侧边界与帧

- `EpisodeNamespace` 增 `term_frame` (B, A) i32：首 proposal 时刻的
  **帧序号**。实现：runtime 在 step() 开头对 running 行递增
  `frame_seq`（每行已记录帧数），proposal 时
  `term_frame = frame_seq`——帧序号空间统一，不受 episode_step
  递增时机影响。
- `RecordStore.env_term_step` 语义保持（provenance），导出侧改
  `t_use = frame_boundary`（ENDED 行 = term_frame+1 / env 级 =
  env_term_step+1；跑满行 = T）。`frame_valid[t]` 已含终止帧
  （写入时刻行仍 running），无需改标记。
- **修 `_RecorderAdapter.on_post_episode` off-by-one**：当前写
  `self._t - 1` 假设末帧 post_action 已执行——子步内终止时 `_t`
  未递增，覆写错帧（现被截断掩盖）。改为按帧序号定位（ended 行
  的末帧索引 = `term_frame`，由 store 记录而非 `_t` 推算）。
- `export_episode`：`num_frames = frame_boundary`；
  `agent_frame_boundary` 导出；term_records 原样透传。

### W3 — 终止帧 observer 输出核对

逐个核 standup + basic_balance 的 reward/状态 observer 的
`on_post_episode` 输出是否构成有效的终止 transition 值：

- `HeightPhiObserver`（phi/height/uprightness）：终止态可算 →
  应在 post_episode 产出真值；
- `CrossSupportBalanceRewarder`/`PostureRewarder`/
  `StandingBalance4StageRewarder`：当前 post_episode 为 no-op →
  终止帧携最后 post_action 值。逐单元决策：补 post_episode 计算
  或显式声明"终止帧 = 上一帧值"并在 manifest 记录理由；
- 双端一致性：CPU observer 的 post_episode 实现 vs 设备
  DeviceObserver 的覆写值需同注入态对拍（E5 模式）。

### W4 — 回归、冒烟与文档

- 契约测试更新：子步终止 → `num_frames = t+1`、帧含终止帧、
  observer 值 = post_episode 值；golden-path 等价重跑；
  部分终止（A 先死 B 续跑）per-agent boundary 各自正确。
- 全量 tests/ 回归 + device 冒烟；如实记录 r_fall/r_cross
  return 变化（reward 记账变化预期内）。
- AUDIT.md P-FW-2 条目更新；LIFECYCLE_TRACE.md 补契约修订；
  `episode.py`/`recorder.py` docstring 修正。

## 风险与如实声明

- **reward 记账变化**：所有实验的终止 transition 从此计入轨迹，
  历史 run 与改后不可直接比较。这是语义变更而非 bugfix 补丁，
  由本文件记录变更理由。
- **observer post_episode 语义依赖**：W3 是本次方案的最大不确定
  面——reward observer 若在终止态产不出正确值，终止帧的 channel
  值是"上一帧拷贝"语义，必须显式记录而非假装正确。
- **不设兼容开关**：单一边界语义；加 flag 等于双语义并存，比变更
  本身更难维护。
- 不做：trajectory 格式新增 done-mask 字段（现有
  `is_true_terminated` 逻辑足够）、zero-frame episode 放行。

## 与已完成工作的接口

- E2 sealed-ENDED/`term_history` 已保留全部 reason + step →
  provenance 不缺数据，只补帧序号；
- E3 `RecordStore.frame_valid` 已覆盖终止帧（写入时刻行 running）；
- E6 `debug_capture` 不受影响（捕获按帧索引）。
