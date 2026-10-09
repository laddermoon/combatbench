# EPISODE_OPTIONS — episode_options 键目录

> 类型：契约

`episode_options` 是**每回合配置通道**：`reset(seed, options)` →
`ctx.episode_options`（`EnvRuntime.reset` 原样转发，`Episode.options`
持久化于回合 manifest）。本表是全仓**唯一权威的键登记**——新增键
必须先在此登记，再到消费方实现。

## 分层

| 层 | 谁消费 | 键的语义 |
|---|---|---|
| simulator 层 | `simulator.reset(options)` | 物理初始化（位姿、距离） |
| plugin 层 | `ctx.episode_options`（`on_pre_episode` 读） | 规则/扰动/录制参数 |
| 元数据层 | 只进 manifest，无人消费 | 实验侧回读（`episode.episode_options`） |

**设备端白名单**：batchframework 只允许
`binding.episode_options_keys` 列出的键广播到设备插件
（`device_rollouter` 校验，未知键 fail-loud）。simulator 层键必须在
binding 里登记才会到设备侧。

## 键登记表

### simulator 层（Humanoid21Simulator）

| 键 | 类型 | 消费方 | 语义 |
|---|---|---|---|
| `initial_distance` | float | `Humanoid21Simulator.reset` | 双方初始间距（米），各偏移 ±d/2 |
| `initial_pose_a` / `initial_pose_b` | str | 同上 | 初始姿态名（`INITIAL_POSES` 键） |
| `episode` | int | `ReplaySimulator.reset` | **replay 保留命名空间**——回放第 N 回合；对插件可见（ctx.episode_options 原样持有），勿占用 |

### plugin 层

| 键 | 消费方 | 语义 |
|---|---|---|
| `video_output_path` | `VideoRecorderPlugin`（`OPTIONS_OUTPUT_PATH_KEY`） | 覆盖本回合 mp4 输出路径（当回合生效，P3-2 已修不回粘） |
| `initial_health_a` / `initial_health_b` | `CombatScoringPlugin` | 初始 HP（覆盖 ctor 默认） |
| `score_log_file` | `CombatScoringPlugin` | 逐物理步审计日志文件；`None` 关闭 |
| `impulse_params` | `ImpulsePerturbationPlugin`/`RelativeImpulsePlugin`/`ConstantForcePlugin` 等 disturbance 插件 | 扰动参数覆盖（当回合生效） |
| `state_bank_index` | StateBank 插件（disturbance_plugins.py） | 状态库索引选择 |

### 元数据层（实验侧写入、分析侧回读）

| 键 | 写方/读方 | 语义 |
|---|---|---|
| `agent_id` | experiments_ppo 各实验写入；分析代码从 `episode.episode_options` 回读 | 标识回合内关注智能体（`robot_a`/`robot_b`） |

## 契约条款

1. **插件消费 episode_options 时必须当回合生效、不留残**——ctor 存
   `self._default_*` 快照，`on_pre_episode` 读 override 而不改写实例属性
   （P3-2/P3-16 修复的污染模式）。
2. **未知键不报错但会进 manifest**——拼错的键静默落进回放清单，
   排查时先查 `episode.episode_options`。
3. 设备端（batchframework）仅转发白名单键；CPU 侧全量进 ctx。
4. `"episode"` 为 replay 保留命名空间——业务键勿以此开头。
