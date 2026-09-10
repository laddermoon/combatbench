# Debug Viewer — 总体前端入口设计

## 定位

一个 dump 目录 = 一个 update 的完整截面。Viewer 是这个截面的浏览器。

四个场景分为两类：
- **全局视角**：Scene 4（Timeline），独立入口，看整个 update 的训练过程
- **Drill-down 链**：Scene 1 → Scene 2 → Scene 3，从 episode 列表逐层进入，看具体数据

## 导航结构

```
┌─────────────────────────────────────────────────────────────┐
│                     顶层入口页                                │
│                                                               │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │  训练时间线      │    │  Episode 列表                   │  │
│  │  (Scene 4)      │    │  (所有 episode 的概览)           │  │
│  │                 │    │                                   │  │
│  │  独立入口        │    │  选中一个 episode →              │  │
│  │  全局视角        │    │    进入 Scene 1                  │  │
│  └─────────────────┘    └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    │ 用户选中一个 episode
                                    ▼
                    ┌──────────────────────────────┐
                    │  Scene 1: Episode → Trajectory │
                    │                                │
                    │  展示 episode 的逐帧数据        │
                    │  展示关联的所有 trajectory      │
                    │                                │
                    │  选中一条 trajectory →          │
                    │    进入 Scene 2                │
                    └──────────────────────────────┘
                                    │
                                    │ 用户选中一条 trajectory
                                    ▼
                    ┌──────────────────────────────┐
                    │  Scene 2: Trajectory → Target   │
                    │                                │
                    │  value / advantage / return     │
                    │  combined_adv / combine chain   │
                    │  趋势图 + 当前帧表格            │
                    │                                │
                    │  ┌──────┐ ┌──────┐             │
                    │  │Target│ │ Epoch│  ← tab 切换  │
                    │  │(默认) │ │      │             │
                    │  └──────┘ └──────┘             │
                    └──────────────────────────────┘
                                    │
                                    │ 切到 Epoch tab
                                    ▼
                    ┌──────────────────────────────┐
                    │  Scene 3: Trajectory × Epoch   │
                    │                                │
                    │  ratio / clip / new_value      │
                    │  epoch 对比图                   │
                    │  actor / critic dynamics        │
                    │                                │
                    │  ← 返回 Target tab              │
                    └──────────────────────────────┘
```

### 层级关系

```
顶层入口
├── 训练时间线 (Scene 4)        ← 独立，全局视角
└── Episode 列表
    └── Episode 详情 (Scene 1)
        └── Trajectory 详情 (Scene 2 + Scene 3)
            ├── Target tab (Scene 2)   ← 训练目标，epoch-invariant
            └── Epoch tab (Scene 3)   ← 训练动态，epoch-variant
```

Scene 2 和 Scene 3 是同一层级的两个 tab——它们看的是**同一条 trajectory**，只是一个看"训练目标"（不变），一个看"训练过程"（随 epoch 变）。

## 用户心智模型

```
顶层：这个 update 有哪些 episode？训练过程整体如何？
  ↓ 选中一个 episode
Scene 1：这个 episode 的某一帧，环境看到了什么？产生了什么 reward？分给了哪些 trajectory？
  ↓ 选中一条 trajectory
Scene 2：这条 trajectory 的每一帧，critic 估了什么？actor 用了什么信号？
  ↓ 切到 Epoch tab
Scene 3：这条 trajectory 在训练中怎么变？ratio 怎么偏离？哪帧被 clip？

（独立）Scene 4：整个 update 的训练时间线，KL 怎么走，何时 early stop
```

## 顶层入口页

```
┌────────────────────────────────────────────────────────────────────────┐
│  CombatBench Debug Viewer          dump: u00008  update: 8             │
│  experiment: standup_step_v3  steps: 204800  trajs: 1024  eps: 1024   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────┐  ┌────────────────────────────────┐ │
│  │ 训练时间线                    │  │ Episode 列表                   │ │
│  │                               │  │                                 │ │
│  │ 4 epochs × 50 batches        │  │ 1024 episodes                   │ │
│  │ early stop: epoch 2, mb 15   │  │                                 │ │
│  │                               │  │ ┌─────┬──────┬──────┬────────┐  │ │
│  │  [KL 演化缩略图]              │  │ │ pos │ seed │ len  │ trajs  │  │ │
│  │   ╱─────╲                    │  │ ├─────┼──────┼──────┼────────┤  │ │
│  │  ╱       ╲──× early stop     │  │ │  0  │ 42   │ 200  │   2    │  │ │
│  │ 0    50   100  150  200      │  │ │  1  │ 43   │ 200  │   2    │  │ │
│  │                               │  │ │  2  │ 44   │ 200  │   2    │  │ │
│  │ [进入时间线 →]                │  │ │ ...                        │  │ │
│  └──────────────────────────────┘  │ └─────┴──────┴──────┴────────┘  │ │
│                                     │                                 │ │
│                                     │ 点击 episode 进入详情 →        │ │
│                                     └────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### 两个入口

1. **训练时间线**：缩略图 + 关键信息（epochs / early stop），点击进入 Scene 4
2. **Episode 列表**：所有 episode 的概览（pos / seed / length / trajectory 数），点击进入 Scene 1

## Scene 2 + Scene 3 的 Tab 组织

Scene 2 和 Scene 3 共享同一条选中的 trajectory，用 tab 切换：

```
┌──────────────────────────────────────────────────────────────────────┐
│  ← 返回 Episode 42                                                   │
│  Trajectory 7  ep0042:robot_a  len=200  frame 50/200                 │
│                                                                        │
│  ┌──────────────┐ ┌──────────────┐                                   │
│  │  Target      │ │  Epoch       │  ← tab                            │
│  │  (Scene 2)   │ │  (Scene 3)   │                                    │
│  └──────────────┘ └──────────────┘                                   │
│                                                                        │
│  [当前 tab 的内容]                                                     │
│  ...                                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

- **Target tab**（Scene 2）：value / advantage / return / combined_adv，趋势图 + 表格
- **Epoch tab**（Scene 3）：ratio / clip / new_value，epoch 对比图 + 表格

Tab 切换时保持 trajectory 和 frame 选择不变。

## 面包屑导航

每个层级页面顶部有面包屑，支持返回上级：

```
顶层入口 > Episode 42 > Trajectory 7 (Target)
顶层入口 > Episode 42 > Trajectory 7 (Epoch)
顶层入口 > 训练时间线
```

## 共享状态

| 状态 | 顶层 | Scene 1 | Scene 2 | Scene 3 | Scene 4 |
|------|------|---------|---------|---------|---------|
| selected_episode | 列表选中 | ✅ 主角 | 继承 | 继承 | — |
| selected_trajectory | — | 列表选中 | ✅ 主角 | 继承 | — |
| selected_frame | — | ✅ 主角 | ✅ 主角 | ✅ 主角 | — |
| selected_epoch | — | — | — | ✅ 主角 | — |
| selected_step | — | — | — | — | ✅ 主角 |

**传递规则**：
- Episode 列表 → Scene 1：传递 `selected_episode`
- Scene 1 → Scene 2：传递 `selected_trajectory` + `selected_frame`
- Scene 2 ↔ Scene 3：共享 `selected_trajectory` + `selected_frame`
- Scene 4 独立，不参与 drill-down 链

## 界面结构

```
┌────────────────────────────────────────────────────────────────────────┐
│  Header: dump 信息 + 面包屑                                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │              当前页面内容（占满剩余空间）                          │ │
│  │              (顶层 / Scene 1 / Scene 2+3 / Scene 4)               │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ 上下文栏（底部，始终可见）                                         │ │
│  │ Episode: 42  Trajectory: 7  Frame: 50/200  Epoch: 1  Step: 35/200 │ │
│  │ frame_id: ep0042:robot_a:50                                        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

## 启动

```bash
PYTHONPATH=. python3 baseline/framework/ppo/debug.py viewer <dump_dir>
```

启动后默认打开**顶层入口页**——用户先看到全局概览（episode 列表 + 时间线入口），再选择 drill down。

## 技术方案

### 单页应用

一个 HTML 文件，页面切换用 display none/block，不重新加载。

```javascript
// 页面路由（前端，无 URL 刷新）
pages = {
  'home':     { show: showHome },      // 顶层入口
  'episode':  { show: showEpisode },   // Scene 1
  'traj':     { show: showTraj },      // Scene 2+3
  'timeline': { show: showTimeline },  // Scene 4
}

function navigate(page, params) {
  // 保存当前页面状态
  // 切换页面
  // 新页面读取 params 定位
  // 更新面包屑 + 上下文栏
}
```

### 后端

Python `http.server`，路由：

```
GET /                                    → index.html
GET /api/manifest                        → dump 元信息
GET /api/traj_map                        → trajectory → episode 映射
GET /api/episode_list                    → episode 概览列表

Scene 1:
GET /api/episode/<list_pos>/frame/<f>
GET /api/image/<list_pos>/<f>

Scene 2 + 3:
GET /api/trajectory/<idx>/overview
GET /api/trajectory/<idx>/frame/<f>
GET /api/trajectory/<idx>/epoch/<e>/overview
GET /api/trajectory/<idx>/epoch/<e>/frame/<f>
GET /api/trajectory/<idx>/epoch_compare

Scene 4:
GET /api/timeline/overview
GET /api/timeline/step/<s>
```

### 前端状态管理

```javascript
const state = {
  // 导航
  currentPage: 'home',
  breadcrumb: [],

  // 共享选择
  selectedEpisode: null,
  selectedTrajectory: null,
  selectedFrame: null,
  selectedEpoch: null,
  selectedStep: null,

  // Scene 2+3 的当前 tab
  trajTab: 'target',  // 'target' | 'epoch'

  // 缓存
  manifest: null,
  trajMap: null,
  episodeList: null,
};
```

### 文件组织

```
dumpkit/
├── viewer/
│   ├── server.py          # HTTP server + API
│   ├── index.html         # 单页入口
│   ├── viewer.css         # 样式
│   ├── viewer.js          # 主逻辑 + 状态管理 + 路由
│   ├── home.js            # 顶层入口页
│   ├── scene1.js          # Episode → Trajectory
│   ├── scene2.js          # Trajectory → Target
│   ├── scene3.js          # Trajectory × Epoch
│   └── scene4.js          # Update Timeline
├── DESIGN_viewer_overview.md  # 本文件
├── DESIGN_viewer_scene1.md
├── DESIGN_viewer_scene2.md
├── DESIGN_viewer_scene3.md
├── DESIGN_viewer_scene4.md
└── DATA_FLOW.md
```

## 导航示例

### 示例 1：从 Episode 列表 → 训练目标 → 训练动态

```
顶层入口：Episode 列表中点击 episode 42
  → navigate('episode', {episode: 42})
  → Scene 1：展示 episode 42 的逐帧数据 + 关联的 trajectory
  → 用户选中 trajectory 7
  → navigate('traj', {trajectory: 7, frame: 50})
  → Scene 2 (Target tab)：展示 value/adv/return 趋势 + frame 50 表格
  → 用户切到 Epoch tab
  → Scene 3：展示 ratio/clip/new_value 演化，保持 frame 50
```

### 示例 2：从 Timeline 发现问题

```
顶层入口：点击"训练时间线"
  → navigate('timeline')
  → Scene 4：看到 KL 在 step 115 爆了
  → 用户回到顶层，选一个 episode
  → 进入 Scene 1 → Scene 2 → Scene 3
  → 在 Scene 3 中选 epoch 2，看 ratio 偏离
```

### 示例 3：面包屑返回

```
当前在 Scene 3 (Trajectory 7, Epoch tab)
  → 点击面包屑 "Episode 42"
  → 返回 Scene 1，保持 episode 42 选中
  → 点击面包屑 "顶层入口"
  → 返回顶层入口页
```

## 不做的事

- 不做多 dump 对比（每个 viewer 实例只看一个 dump）
- 不做实时训练监控（viewer 是离线工具，看已捕获的 dump）
- 不做参数修改 / 重新训练（只读）
- 不做前端构建系统（vanilla JS + HTML + CSS）
