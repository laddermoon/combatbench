# Debug Viewer — 场景一：Episode → Trajectory 转换视图

## 目的

回答一个问题：**"这个 episode 的某一帧，环境看到了什么，产生了什么 reward，分给了哪些 channel？"**

用户选定一个 episode，拖动进度条，同步看到：
1. 当前帧的图片
2. 当前帧的 episode 数据（obs / action / observer outputs）
3. 当前帧关联的所有 trajectory 数据（per-channel reward / actor_weight / floor_weight）

## 核心难点：Episode → Trajectory 映射

### 为什么不能假设

Trajectory 的数量、长度、起始帧都由实验的 `build_trajectories()` 决定：

- 一个 episode 可能产生 **任意数量** 的 trajectory（不一定是 2 条）
- Trajectory 可能从 episode 的**中间**开始（`t_start > 0`）
- Trajectory 长度可能**不等**于 episode 长度

### FrameID 的语义

`frame_id` 格式 `ep{episode_pos:04d}:{agent_id}:{t}`，其中 `episode_pos` 是 episode 在 dump 批次中的**列表位置**（全局唯一）。由 `dump_capture._make_frame_ids` 用 obs 内容匹配生成，不在生产路径中产生。

匹配失败的 trajectory 用 `flat:{i}` 标记，viewer 应明确提示"溯源不可用"。

### 映射方式

dump capture 时 `_make_frame_ids` 已经通过 obs 内容匹配生成了 `frame_id` 数组。`traj_map.json` 直接从 `frame_id` 解析，不需要额外的 provenance 机制：

```json
{
  "n_episodes": 512,
  "n_trajectories": 1024,
  "episodes": [
    {
      "list_pos": 0,
      "seed": 4138,
      "num_frames": 200,
      "trajectories": [
        { "traj_idx": 0, "agent_id": "robot_a", "t_start": 0, "length": 200 },
        { "traj_idx": 1, "agent_id": "robot_b", "t_start": 0, "length": 200 }
      ]
    },
    {
      "list_pos": 1,
      "seed": 4139,
      "num_frames": 200,
      "trajectories": [
        { "traj_idx": 2, "agent_id": "robot_a", "t_start": 0, "length": 200 },
        { "traj_idx": 3, "agent_id": "robot_b", "t_start": 0, "length": 200 }
      ]
    }
  ]
}
```

### 映射构建算法

`_make_frame_ids` 已用 obs 内容匹配生成 `frame_id`（`ep{pos:04d}:{agent}:{t}`）。`traj_map.json` 从 `frame_id` 数组 + `ep_lengths` 直接解析，不需要 provenance：

```python
def build_traj_map(episodes, trajectories, frame_ids):
    """从 frame_id 数组解析 episode → trajectory 映射。

    frame_ids 由 _make_frame_ids 用 obs 内容匹配生成。
    每个 trajectory 的首帧 frame_id 包含 (episode_pos, agent_id, t_start)。
    """
    ep_map = [[] for _ in range(len(episodes))]
    offset = 0
    for traj_idx, traj in enumerate(trajectories):
        T = len(traj.obs)
        if T == 0:
            continue
        fid = str(frame_ids[offset])
        if fid.startswith("flat:"):
            # 匹配失败，无法关联到任何 episode
            # 不加入任何 ep_map 条目，viewer 显示"溯源不可用"
            offset += T
            continue
        # 解析 ep{pos:04d}:{agent_id}:{t_start}
        parts = fid.split(":")
        ep_pos = int(parts[0][2:])  # 去掉 "ep" 前缀
        agent_id = parts[1]
        t_start = int(parts[2])
        ep_map[ep_pos].append({
            "traj_idx": traj_idx,
            "agent_id": agent_id,
            "t_start": t_start,
            "length": T,
        })
        offset += T
    return [{
        "list_pos": i,
        "seed": ep.base_seed,
        "num_frames": ep.num_frames,
        "trajectories": trajs,
    } for i, (ep, trajs) in enumerate(zip(episodes, ep_map))]
```

## 数据关系

### Episode 数据（episodes.npz）

每帧的数据，按 episode **列表顺序**拼接，用 `episode_frame_offsets` 定位：

| 字段 | 形状 | 说明 |
|------|------|------|
| `obs.robot_a` | (N_frames, 96) | robot_a 的观测，96 维 |
| `obs.robot_b` | (N_frames, 96) | robot_b 的观测 |
| `actions.robot_a` | (N_frames, 21) | robot_a 的动作，21 维 |
| `actions.robot_b` | (N_frames, 21) | robot_b 的动作 |
| `explore_factors.robot_a` | (N_frames,) | 每帧的探索因子 |
| `explore_factors.robot_b` | (N_frames,) | |
| `observer_outputs.*` | (N_frames,) | 各 observer 的逐帧输出 |

Episode `i`（列表位置）的帧范围：`[episode_frame_offsets[i], episode_frame_offsets[i+1])`

### Trajectory 数据（trajectories.npz）

所有 trajectory 按返回顺序拼接。每条 trajectory 的长度由 `ep_lengths` 数组指定。

| 字段 | 形状 | 说明 |
|------|------|------|
| `ep_lengths` | (n_trajs,) | 每条 trajectory 的帧数 |
| `reward.{channel}` | (total_frames,) | 每个 channel 的逐帧 reward |
| `actor_weight.{channel}` | (total_frames,) | 每个 channel 的逐帧 actor_weight |
| `is_terminated.{channel}` | (n_trajs,) | 每个 channel 的终止标志（per-trajectory） |
| `floor_weight` | (total_frames,) | 逐帧 floor_weight |
| `explore_factor` | (total_frames,) | 逐帧探索因子 |
| `importance` | (n_trajs,) | per-trajectory 重要性权重 |
| `frame_id` | (total_frames,) | 帧标识 `ep{pos:04d}:{agent}:{t}`（pos=列表位置，全局唯一）；匹配失败为 `flat:{i}` |

Trajectory `j` 的全局帧范围：`[sum(ep_lengths[:j]), sum(ep_lengths[:j+1]))`

### 帧映射：Episode 帧 → Trajectory 帧

给定 episode 列表位置 `i`，episode 内帧号 `t`，找到关联的 trajectory 数据：

```python
def get_traj_frame(dump_dir, ep_list_pos, ep_frame, traj_map):
    """获取 episode 指定帧关联的所有 trajectory 数据。"""
    tr = np.load(dump_dir / "trajectories.npz", allow_pickle=True)
    ep_lengths = tr["ep_lengths"]
    # traj_map["episodes"][ep_list_pos]["trajectories"] = [
    #   {traj_idx, agent_id, t_start, length}, ...
    # ]
    trajs = traj_map["episodes"][ep_list_pos]["trajectories"]
    results = []
    for t_info in trajs:
        ti = t_info["traj_idx"]
        t_start = t_info["t_start"]
        length = t_info["length"]
        # episode frame t 对应 trajectory 内的帧 (t - t_start)
        local_frame = ep_frame - t_start
        if local_frame < 0 or local_frame >= length:
            results.append(None)  # 此 trajectory 不覆盖该帧
            continue
        global_offset = int(ep_lengths[:ti].sum())
        offset = global_offset + local_frame
        channels = {}
        for ch in tr["channel_names"]:
            channels[ch] = {
                "reward": float(tr[f"reward.{ch}"][offset]),
                "actor_weight": float(tr[f"actor_weight.{ch}"][offset]),
            }
        results.append({
            "agent_id": t_info["agent_id"],
            "traj_idx": ti,
            "local_frame": local_frame,
            "channels": channels,
            "floor_weight": float(tr["floor_weight"][offset]),
            "explore_factor": float(tr["explore_factor"][offset]),
            "is_terminated": {
                ch: bool(tr[f"is_terminated.{ch}"][ti])
                for ch in tr["channel_names"]
            },
            "importance": float(tr["importance"][ti]),
        })
    return results
```

## 界面布局

```
┌──────────────────────────────────────────────────────────┐
│  Episode [0 ▼]  seed=4138  200 frames                    │
│  [◄━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━►]  50 / 200    │
│  [▶ play]  speed: [1x ▼]                                  │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐  Episode Frame 50                       │
│  │              │  obs.robot_a (96-dim):                   │
│  │   image      │    [0.130, 0.274, 0.084, ...]            │
│  │  frame 50    │  obs.robot_b (96-dim):                   │
│  │              │    [-0.853, 0.311, -0.112, ...]         │
│  │  (or empty   │  actions.robot_a (21-dim):              │
│  │   if no      │    [0.078, 0.329, 0.470, ...]           │
│  │   render)    │  actions.robot_b (21-dim):              │
│  │              │    [...]                                │
│  └──────────────┘  explore_factor: -0.631                │
│                                                            │
│  Observer Outputs (frame 50):                              │
│    standing_balance_a.h_torso: 0.091                      │
│    standing_balance_a.stage: 1.0                          │
│    height_phi_a.height: 0.091  uprightness: -0.006        │
│    foot_state_a.left_contact: False  right_contact: False │
│    standing_balance_b.h_torso: 0.110  ...                  │
│                                                            │
├──────────────────────────────────────────────────────────┤
│  Trajectory 0 — robot_a    [t_start=0, local=50, len=200] │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ channel        reward      actor_weight  terminated  │ │
│  │ r_potential   +0.000001    1.0000       False        │ │
│  │ r_fall        +0.000000    0.0000       False        │ │
│  │ r_left_foot   +0.050000    0.0000       False        │ │
│  │ r_right_foot  +0.050000    0.0000       False        │ │
│  │ floor_weight: 0.0000  explore_factor: -0.631         │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
├──────────────────────────────────────────────────────────┤
│  Trajectory 1 — robot_b    [t_start=0, local=50, len=200] │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ channel        reward      actor_weight  terminated  │ │
│  │ r_potential   +0.000001    1.0000       False        │ │
│  │ r_fall        +0.000027    0.0000       False        │ │
│  │ r_left_foot   +0.050000    0.0000       False        │ │
│  │ r_right_foot  +0.050000    0.0000       False        │ │
│  │ floor_weight: 0.0000  explore_factor: -0.631         │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 布局说明

**顶部**：Episode 选择器 + 进度条 + 播放控制

**中部**：图片 + Episode 数据
- 左侧：当前帧的 PNG 图片（如果 render 过；没有则显示占位符）
- 右侧：当前帧的 episode 级数据
  - obs.robot_a / obs.robot_b（96 维向量，可折叠）
  - actions.robot_a / actions.robot_b（21 维向量，可折叠）
  - explore_factor
  - observer outputs（按 observer 分组，显示关键值）

**底部**：Trajectory 数据（有几条显示几条）
- 每个 trajectory 一个区块，标题显示 agent_id + t_start + local_frame + length
- 如果当前 episode 帧不在某 trajectory 覆盖范围内，该 trajectory 显示"不覆盖此帧"
- 表格展示当前帧的 per-channel reward / actor_weight / terminated
- 底部显示 floor_weight / explore_factor / importance

## 交互

| 操作 | 行为 |
|------|------|
| 拖动进度条 | 图片、obs、所有 trajectory 数据同步更新到当前帧 |
| 点击 ▶ | 自动播放，每秒 N 帧（可调速度） |
| 选择 episode | 加载该 episode 的数据，进度条归零 |
| 点击 obs/action 向量 | 展开/折叠（默认折叠，显示前 10 个值） |
| 点击 observer output | 展开/折叠（默认只显示关键 observer） |

## 关键 Observer 字段（默认显示）

不是所有 observer output 都同等重要。默认显示这些，其余折叠：

| Observer | 字段 | 含义 |
|----------|------|------|
| standing_balance_a | h_torso | robot_a 躯干高度 |
| standing_balance_a | stage | 训练阶段（1=standup, 2=balance） |
| height_phi_a | height | robot_a 高度 |
| height_phi_a | uprightness | robot_a 直立度 |
| foot_state_a | left_foot_contact | robot_a 左脚是否着地 |
| foot_state_a | right_foot_contact | robot_a 右脚是否着地 |
| standing_balance_b | (同上) | robot_b 对应字段 |

## 图片加载

如果 `<dump_dir>/record/` 存在（用户执行过 `render`），从那里加载：

```
<dump_dir>/record/episode_00000/step_00050.png  # 注意 off-by-one: step_N+1 = frame N
```

如果不存在，图片区域显示提示："未渲染图片，运行 `debug.py render <dump_dir> --episode N` 生成"。

## 技术方案

| 层 | 选择 | 理由 |
|----|------|------|
| 后端 | Python HTTP server（标准库 `http.server`） | 无需额外依赖，读 NPZ 返回 JSON |
| 前端 | 单页 HTML + vanilla JS + CSS | 无框架，无构建步骤 |
| 图表 | 不需要（本场景是数值表格，不是趋势图） | 保持简单 |
| 启动 | `debug.py viewer <dump_dir>` | 一条命令 |

### 后端 API

```
GET /api/manifest
  → {
      n_episodes, channel_names, observer_keys, has_images,
      traj_map: { episodes: [{list_pos, seed, num_frames, trajectories: [...]}] }
    }

GET /api/episode/<list_pos>/frame/<frame>
  → {
      obs_a: [...], obs_b: [...],
      act_a: [...], act_b: [...],
      explore_factor_a: float, explore_factor_b: float,
      observer_outputs: { key: value, ... },
      image_url: "/api/image/<list_pos>/<frame>" or null,
      trajectories: [
        {
          agent_id: "robot_a",
          traj_idx: 0,
          t_start: 0,
          local_frame: 50,
          length: 200,
          covered: true,
          channels: { r_potential: {reward, actor_weight}, ... },
          is_terminated: { r_potential: false, ... },
          floor_weight, explore_factor, importance
        },
        ...
      ]
    }

GET /api/image/<list_pos>/<frame>
  → PNG file or 404
```

### 前端交互流

```
用户选择 episode 0
  → GET /api/manifest → 知道有 512 episodes, 4 channels, traj_map
  → GET /api/episode/0/frame/0 → 渲染初始帧

用户拖动进度条到 frame 50
  → GET /api/episode/0/frame/50 → 更新所有数据

用户点击 ▶
  → 每秒 10 次 GET /api/episode/0/frame/N+1
```

## 前置工作

### 1. dump capture 增加 traj_map.json

在 `dump_capture.py` 中，`_make_frame_ids` 生成 `frame_id` 数组后，从中解析 `traj_map.json`：

```python
def _build_traj_map(episodes, trajectories, frame_ids):
    """从 frame_id 数组解析 episode → trajectory 映射。"""
    # ... (see 映射构建算法 above)
```

### 2. 处理匹配失败的情况

如果 `_make_frame_ids` 对某条 trajectory 匹配失败（obs 内容不在任何 episode 中），`frame_id` 为 `flat:{i}`。`traj_map.json` 中该 trajectory 不加入任何 episode 的映射，viewer 显示"溯源不可用"。

## 不做的事

- 不做趋势图（那是其他场景的事）
- 不做 GAE / combine / update 数据展示（那是其他场景的事）
- 不做多 episode 对比
- 不做 episode 列表搜索/过滤（512 个 episode 用下拉框够了）
