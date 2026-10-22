# CoreClip 工具使用说明

> 数据格式契约见 `CORECLIP_SPEC.md`；本文档是**命令行工具**的使用手册。

所有 clip 操作经单一入口（必须从 `things/combatbench/` 运行）：

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 -m mocap.humanoid21.clip <command> [args]
```

## 命令一览

| 命令 | 用途 |
|---|---|
| `info` | 快速查看 clip 摘要（帧数、时长、meta） |
| `check` | **统一检查**：basic / velocity / actuator / ref_action |
| `play` | 运动学回放渲染 MP4（状态覆写，非物理执行） |
| `fix` | root-lift 穿地自动修复 |
| `rederive-vel` | 从姿态轨迹重算速度场 |
| `rederive-action` | 逐帧 CEM 求解最优近似 ref_action |
| `reclip` | 沿连续姿态路径优化动作序列，生成可执行 CoreClip |
| `convert` | 既有 retarget RefClip npz → CoreClip |

---

## `info` —— 摘要

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip info \
    --clip mocap/humanoid21/clip/data/jab_l_full.npz
```

输出 clip_id、T、时长、各字段范围、meta（含来源溯源）。

## `check` —— 统一检查

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip check --clip <x.npz> \
    [--only basic,velocity,actuator,ref_action] \
    [--robot robot_a] \
    [--json report.json]
```

- 默认跑全部四节；`--only` 逗号分隔选子集
- `--json` 导出结构化报告（给前端/AI 消费），字段：`sections.<name>.{status, metrics, findings[]}`
- **退出码**：全部非 FAIL → 0；任一节 FAIL → 1（可用于 CI 门禁）

### 四节语义与判读

| 节 | 检查内容 | 性质 |
|---|---|---|
| `basic` | 格式校验；脚部穿地；全身 geom 穿地；同机器人自碰撞；非脚 geom 触地；躯干倾倒；root 高度 | 几何硬检查 |
| `velocity` | 记录速度 vs 位置差分反推速度；加速度接缝（逐帧独立编辑）；首尾帧速度 | 一致性 |
| `actuator` | `mj_rne` 逆动力学需求 vs `gear×ctrlrange` 容量；反推 PD 指令是否越界 [-1,1] | **筛查，非证明** |
| `ref_action` | 记录动作序列顺向物理回放，对比目标状态（跟踪误差/发散/跌倒） | **唯一真可执行性测试** |

### 状态级别

`PASS` 通过 / `WARN` 可疑（值得看但不一定错）/ `FAIL` 明确问题（附修复建议命令）/ `SKIP` 无输入可检（如 clip 无 `ref_action`）。

### 判读要点（重要）

- **velocity 残差大 ≠ 数据错**。20Hz clip 若来自 120Hz 源下采样，
  记录速度携带高频瞬时信息，中心差分反推不回来是正常的（混叠）。
  真正可疑的是"记录速度与位置趋势**反向**"（报 FAIL）和
  **加速度接缝**（提示逐帧独立修改，如手工编辑或未平滑的 IK 修复）。
- **actuator FAIL ≠ 不可执行**。RNE 忽略接触支撑，支撑腿的读数是
  上界；它回答的是"如果不靠接触帮忙，需多大扭矩"。
- **ref_action 才是裁决**。即使它 WARN（跟踪误差偏大）也比前三节
  更能说明问题——它真的跑过物理。回放的 ref_action 对转换 clip 是
  "PD 目标建议"，发散不代表动作不可能，只代表这条指令序列不行。
- **basic 的 self-collision WARN**：模型只排除 waist↔thigh，其余
  自接触都会算——手臂折叠贴近躯干报小穿透常见，看深度是否 >5mm。

### 典型判读流程

```bash
# 1. 先只看几何+速度（不需要仿真的轻量检查）
PYTHONPATH=. python3 -m mocap.humanoid21.clip check \
    --clip x.npz --only basic,velocity

# 2. 若 basic 报穿地 → 修复 → 复检
PYTHONPATH=. python3 -m mocap.humanoid21.clip fix --clip x.npz
PYTHONPATH=. python3 -m mocap.humanoid21.clip check \
    --clip x_groundfix.npz --only basic

# 3. 全量检查（含物理筛查与回放）
PYTHONPATH=. python3 -m mocap.humanoid21.clip check --clip x_groundfix.npz
```

## `play` —— 运动学回放 → MP4

```bash
PYTHONPATH=. MUJOCO_GL=egl python3 -m mocap.humanoid21.clip play \
    --clip x.npz --out x.mp4 [--fps 20 --slow 2 --burn-frame]
```

- `--slow 2` 慢放一倍；`--burn-frame` 在画面上烧录帧号/时间
- **不是物理验证**：逐帧 `set_core_state` 覆写，只展示运动外观
- 需要 EGL 渲染环境（`MUJOCO_GL=egl`）

## `fix` —— 穿地自动修复

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip fix --clip x.npz \
    [--out x_groundfix.npz] [--mode lowest|constant] [--margin 0.002]
```

- `lowest`（默认）：逐帧抬 root_z 使最低脚点 = margin
- `constant`：整条 clip 统一抬最大穿透量
- **绝不原地覆盖**：默认输出 `<name>_groundfix.npz`；`--out` 与输入
  相同路径会被拒绝（退出码 2）
- 只改 `root_pos.z`；meta 记录 `fix_ground_penetration` 溯源
- **局限**：治"沉"不治"浮"——源数据双脚不共面时另一只脚仍虚浮；
  `root_vel_local.z` 不更新（修完后可用 `rederive-vel` 补速度一致性）

## `rederive-vel` —— 从姿态重算速度

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip rederive-vel \
    --clip x.npz [--out x_vel.npz] [--zero-endpoints]
```

- 姿态为第一性：内部帧用中心差分从姿态轨迹重算
  `joint_vel_norm` / `root_vel_local` / `root_angular_vel_local`，
  保证记录速度与存储姿态严格一致（`check` velocity 节归零）
- **首尾帧默认保留记录值**（携带 clip 衔接语义）；
  `--zero-endpoints` 强制清零（"从静止到静止"片段）
- 绝不原地覆盖；默认输出 `<name>_vel.npz`
- **注意**：refclip 转换 clip 的记录速度来自 120Hz 源，比 20Hz
  差分更精确——对未编辑过的转换 clip 做重导出是降级而非修复；
  适用场景是姿态被修改后（groundfix、手工编辑）

## `rederive-action` —— 最优近似 ref_action 求解

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip rederive-action \
    --clip x.npz [--out x_act.npz] \
    [--weights jp=1.0,jv=0.05,rp=1.0,rq=0.5,rv=0.2,ra=0.1,reg=0.02] \
    [--tol-solved 0.05 --tol-approx 0.2] [--n-samples 64 --n-iters 4]
```

- **ref_action 语义不变**：只是参考，不保证到达下一状态
- 逐帧从精确状态 k 出发，CEM 采样 action，仿真 25 步，选使
  加权距离 `D` 最小的指令
- `D` 分量：关节位姿(主)、关节速度、根位置/朝向/速度、
  动作正则项（等效解中选低耗力的）
- 输出 meta `ref_action_solver` 记录逐帧 D 和评级：
  `solved`(D<0.05) / `approx`(<0.2) / `unsolved`——
  **unsolved 帧说明该转移在此动力学下不可达，是数据事实**
- 逐帧 teleport 求解，不保证序列级跟踪（由 `check ref_action`
  回放节验证）；实测 jab：CEM 解的回放误差 0.17 < 转换值 0.26
- 绝不原地覆盖；约 15s / 22 帧 clip

## `reclip` —— 沿目标路径生成可执行 clip

新版默认方法是 `path-sweep`。它把源 Clip 的姿态构建成连续
`TargetPath q=γ(s)`，再用后缀 rollout 优化动作序列；路径只提供
姿态几何，时间进度由优化器决定。输出的每个状态都是仿真落点：

```text
sim(output[k], output.ref_action[k]) == output[k+1]
```

### 新接口

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip reclip \
    --clip source.npz \
    --objective track|pass|reach-stop \
    --path endpoints|all \
    [--start-frame 0 --end-frame -1] \
    [--budget-frames 12 | --time-budget 0.60 | --budget-search 8,10,12] \
    [--ds-max 0.2 --s-ext 0.10 --post-frames 0] \
    [--seeds 0,1 --sweeps 3 --n-samples 32 --n-iters 2] \
    --out result.npz
```

目标语义：

| objective | 含义 | 备注 |
|---|---|---|
| `track` | 保留源姿态时间线，同步经过每个路径 waypoint | 必须 `--path all`；中间速度不是优化目标 |
| `pass` | 沿路径尽快推进；首个 `s>=1` 控制帧为通过事件 | 默认截断到事件帧；不要求目标稳定 |
| `reach-stop` | 到达 `s=1`，并在事件帧优化/报告零速度误差 | 不做保持验证，不设固定容差 |
| `reach-state` | 语义保留：到达末姿态并匹配源末速度 | **当前未实现，CLI 返回 2** |

路径模式：

- `endpoints`：只使用片段首尾姿态，适合短单向过渡，不适合 `track`。
- `all`：使用片段内全部源姿态 waypoint；保留动作形态和全身协调顺序，
  适合 `track`，也可用于 `pass` / `reach-stop`。

`--ds-max` 是每个控制帧允许的局部进度搜索宽度，不是路径容差；
`--s-ext` 是有界外延，允许 `s` 少量超过 1（默认 `s_max=1.1`）。
`pass` 按 20Hz 控制帧检测通过事件，不做物理子步检测。

### 兼容旧调用

- `--via guide.npz`：在新默认 `path-sweep` 下映射为
  `track + all`，并使用 `guide.ref_action` 热启动。
- `--start s.npz --end e.npz`：映射为 `pass + endpoints`；
  `--T` 仍表示导轨帧数，对应动作预算 `T-1`。
- `--method sweep|greedy|mpc` 保留旧的固定帧号导轨实现；
  `--method path-sweep` 是新默认。

### 求解与输出

- `path-sweep` 对某一帧的候选 action 不只看下一步，而是执行完整
  action 后缀后由 objective evaluator 评分，因此会选择有利于后续
  推进的动作。
- `--budget-search 8,10,12` 会逐个尝试离散动作预算；上一最好动作
  序列会重采样为下一预算的热启动。结果记录的是该预算集合、该
  seed/采样预算下找到的最好结果，**不声称全局最速**。
- `--seeds` 可传多个 CEM seed；无 `--budget-search` 时使用 `--seed`。
- 输出前会从初始完整 CoreState 重新回放选中动作序列并录制；
  `meta["reclip"]["replay_error"]` 包含关节位姿/速度、根位置/朝向/
  线速度/角速度的全部回放误差。
- `meta["reclip"]` 记录 objective、path knots、solver 参数、事件帧、
  progress、逐帧路径误差、terminal pose/velocity error、status。
- 不预设固定姿态/路径/速度容差；优化结果直接报告偏差。
- `--out` 不得等于任何输入路径（`--clip` / `--start` / `--via` /
  `--end`），否则退出码 2。

示例：

```bash
# 保留整条源姿态路径，尽快通过末端姿态
PYTHONPATH=. python3 -m mocap.humanoid21.clip reclip \
    --clip /tmp/reclip_work/hdm05_3146_88_2895_127_ext.npz \
    --objective pass --path all --budget-frames 12 \
    --ds-max 0.4 --seeds 0,1 --out /tmp/jab_pass_all.npz

# 老 --via 写法：track + all
PYTHONPATH=. python3 -m mocap.humanoid21.clip reclip \
    --start guide.npz --via guide.npz --out guide_track.npz
```

已实测的 extend 回归中，`pass + endpoints` 与 `pass + all`
均可生成 `ref_action` 回放 PASS 的输出。若源初始姿态本身穿地，
输出的 `basic` 检查仍会继承该问题；`fix` 会改变 root_pos 并破坏
"ref_action 精确回放"语义，不能把它当作 reclip 的后处理。

## `convert` —— retarget clip 转换

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip convert \
    --ref-id cmu_144_20_seg01 \
    --dir baseline/humanoid21/hierarchy/assets/clips/full \
    --out x.npz [--start 0 --end 13] [--no-ref-action]
```

---

## 血缘（lineage）

每个派生 clip 的 `meta["lineage"]` 是按序追加的溯源记录：

```json
"lineage": [
  {"op": "convert", "parent": "cmu_144_20_seg01", ...},
  {"op": "fix_ground", "parent": "...", "params": {...}, "ts": "..."},
  {"op": "rederive_vel", "parent": "...", "params": {...}, "ts": "..."}
]
```

`convert` / `slice` / `fix` / `rederive-vel` / `record` 都会自动追加。
`meta["source"]` 保留最近一次的直接来源；`lineage` 是完整链。

## 已知边界

- 修复命令：`fix`（root-lift 穿地）、`rederive-vel`（速度重算）。
  其余修复项（IK 姿态、时序平滑）未实现。
- `basic` 检查运行仿真状态设置但不做物理步进，无需 EGL；
  `play` 需要 EGL。
- 检查只读，不修改输入 clip。
