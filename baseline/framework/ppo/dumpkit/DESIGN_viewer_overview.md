# Debug Viewer — 总体前端入口设计

## 定位

一个 dump 目录 = 一个 update 的完整截面。Viewer 是这个截面的浏览器。

四个场景是四个观察视角，共享同一个 dump，共享部分上下文（选中的 trajectory / episode / frame）。

## 用户心智模型

用户打开一个 dump 后，心智路径是：

```
"这个 update 发生了什么？"
    ↓
Scene 4: 训练过程时间线（全局概览）
    "KL 怎么走？early stop 在哪？"
    ↓
"某条 trajectory 的训练目标是什么？"
    ↓
Scene 2: Trajectory → 训练目标（value/adv/return）
    "critic 估了什么？combine 后 actor 用什么？"
    ↓
"这条 trajectory 在训练中怎么变？"
    ↓
Scene 3: Trajectory × Epoch（ratio/clip/new_value 演化）
    "ratio 怎么偏离？哪帧被 clip？"
    ↓
"回到源头，这个 episode 的某一帧环境看到了什么？"
    ↓
Scene 1: Episode → Trajectory（obs/action/reward/observer）
    "reward 从哪来？分给了哪些 channel？"
```

不是线性流程——用户会在场景间跳转，带着一个问题从一个视角切换到另一个视角。

## 共享状态

四个场景之间共享的选择上下文：

| 状态 | Scene 1 | Scene 2 | Scene 3 | Scene 4 |
|------|---------|---------|---------|---------|
| selected_episode | ✅ 主角 | — | — | — |
| selected_trajectory | 关联显示 | ✅ 主角 | ✅ 主角 | — |
| selected_frame | ✅ 主角 | ✅ 主角 | ✅ 主角 | — |
| selected_epoch | — | — | ✅ 主角 | 关联高亮 |
| selected_step | — | — | — | ✅ 主角 |

**跨场景导航**：
- Scene 1 选中一条 trajectory → 切到 Scene 2/3 时自动选中同一条
- Scene 2 选中 frame 50 → 切到 Scene 3 时自动定位 frame 50
- Scene 4 选中 step 115（early stop）→ 切到 Scene 3 时自动选中对应 epoch

## 界面结构

```
┌────────────────────────────────────────────────────────────────────────┐
│  CombatBench Debug Viewer          dump: u00008  update: 8             │
│  experiment: standup_step_v3  steps: 204800  trajs: 1024  eps: 1024   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                           │
│  │ Scene1 │ │ Scene2 │ │ Scene3 │ │ Scene4 │  ← 顶部 tab 切换          │
│  │Episode │ │Traj→   │ │Traj×   │ │Update  │                             │
│  │→Traj   │ │Target  │ │Epoch   │ │Timeline│                             │
│  └────────┘ └────────┘ └────────┘ └────────┘                           │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │              当前场景的内容（占满剩余空间）                        │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  │                                                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ 共享上下文栏（底部，始终可见）                                    │ │
│  │                                                                    │ │
│  │ Episode: 42  Trajectory: 7  Frame: 50/200  Epoch: 1  Step: 35/200 │ │
│  │ frame_id: ep0042:robot_a:50                                        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### 顶部 Header

- dump 基本信息：update 编号、实验名、总步数、trajectory 数、episode 数
- 从 `manifest.json` 加载

### 顶部 Tab 栏

四个 tab，每个对应一个场景。tab 名简洁：
- **Episode → Traj**（Scene 1）
- **Traj → Target**（Scene 2）
- **Traj × Epoch**（Scene 3）
- **Timeline**（Scene 4）

Tab 切换时保留共享状态，不重新加载。

### 中间内容区

当前场景的完整界面（各场景自己的布局见各自设计文档）。

### 底部上下文栏

始终可见，显示当前共享状态：
- Episode / Trajectory / Frame / Epoch / Step
- frame_id（如果有）

让用户随时知道"我在看什么"。

## 启动

```bash
PYTHONPATH=. python3 baseline/framework/ppo/debug.py viewer <dump_dir>
```

启动后默认打开 Scene 4（Timeline），因为它是全局概览——用户先看训练过程整体健康度，再 drill down 到具体 trajectory。

## 技术方案

### 单页应用

一个 HTML 文件，四个场景用 tab 切换（display none/block），不重新加载页面。

```
index.html
├── <div id="scene1" style="display:none"> ... </div>
├── <div id="scene2" style="display:none"> ... </div>
├── <div id="scene3" style="display:none"> ... </div>
├── <div id="scene4" style="display:block"> ... </div>  ← 默认
└── <div id="context-bar"> ... </div>  ← 底部共享栏
```

### 后端

Python `http.server`，路由：

```
GET /                                    → index.html
GET /api/manifest                        → dump 元信息
GET /api/traj_map                        → trajectory → episode 映射

Scene 1:
GET /api/episode/<list_pos>/frame/<f>
GET /api/image/<list_pos>/<f>

Scene 2:
GET /api/trajectory/<idx>/overview
GET /api/trajectory/<idx>/frame/<f>

Scene 3:
GET /api/trajectory/<idx>/epoch/<e>/overview
GET /api/trajectory/<idx>/epoch/<e>/frame/<f>
GET /api/trajectory/<idx>/epoch_compare

Scene 4:
GET /api/timeline/overview
GET /api/timeline/step/<s>
```

### 前端状态管理

vanilla JS，一个全局 state 对象：

```javascript
const state = {
  // 共享状态
  selectedEpisode: null,
  selectedTrajectory: null,
  selectedFrame: null,
  selectedEpoch: null,
  selectedStep: null,

  // 当前场景
  currentScene: 'scene4',

  // 缓存
  manifest: null,
  trajMap: null,
};
```

场景切换时：
1. 保存当前场景的内部状态
2. 切换 tab
3. 新场景读取共享 state，定位到对应位置
4. 更新底部上下文栏

### 文件组织

```
dumpkit/
├── viewer/
│   ├── server.py          # HTTP server + API
│   ├── index.html         # 单页入口
│   ├── viewer.css         # 样式
│   ├── viewer.js          # 主逻辑 + 状态管理
│   ├── scene1.js          # Scene 1 逻辑
│   ├── scene2.js          # Scene 2 逻辑
│   ├── scene3.js          # Scene 3 逻辑
│   └── scene4.js          # Scene 4 逻辑
├── DESIGN_viewer_overview.md  # 本文件
├── DESIGN_viewer_scene1.md
├── DESIGN_viewer_scene2.md
├── DESIGN_viewer_scene3.md
├── DESIGN_viewer_scene4.md
└── DATA_FLOW.md
```

## 跨场景导航示例

### 示例 1：从 Timeline 发现问题 → drill down

```
Scene 4: 看到 KL 在 step 115 爆了（early stop）
  → 点击 step 115
  → 切到 Scene 3
  → 自动选中 epoch 2（step 115 = epoch 2 mb 15）
  → 看 ratio 在 epoch 2 的分布，哪些帧偏离最大
  → 选中偏离最大的帧
  → 切到 Scene 2
  → 看这帧的 value/adv/return，理解为什么 policy 想往这个方向移
```

### 示例 2：从 Episode 发现问题 → 训练目标

```
Scene 1: 选中 episode 42，看到 frame 50 的 reward 异常
  → 选中 frame 50 关联的 trajectory
  → 切到 Scene 2
  → 自动选中同一条 trajectory，定位 frame 50
  → 看 value/adv/return，理解 critic 怎么评估这帧
  → 切到 Scene 3
  → 看 epoch 0 的 ratio，确认 policy 初始评估
```

### 示例 3：从 Trajectory 训练目标 → 训练过程

```
Scene 2: 选中 trajectory 7，看到 combined_adv 在 frame 100 翻转
  → 切到 Scene 3
  → 自动选中 trajectory 7，定位 frame 100
  → 看 ratio 在各 epoch 的演化，理解 policy 怎么响应这个信号
  → 切到 Scene 4
  → 看 timeline，理解这个 update 的整体训练节奏
```

## 不做的事

- 不做多 dump 对比（每个 viewer 实例只看一个 dump）
- 不做实时训练监控（viewer 是离线工具，看已捕获的 dump）
- 不做参数修改 / 重新训练（只读）
- 不做前端构建系统（vanilla JS + HTML + CSS）
