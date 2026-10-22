# Reclip 实现计划

> 状态：实施计划；对应已评审通过的 `RECLIP_DESIGN.md`。
>
> 本文档把已接受设计拆成可执行工程任务。尚未开始实现。

## 0. 实现目标与不变量

实现后的 `reclip` 应以一个源 CoreClip 为输入，输出一个真实仿真生成的 CoreClip：

- 输出状态来自仿真，不是插值结果；
- 输出 `ref_action` 是实际执行过的动作；
- 最后一版输出必须从初始状态完整回放一次并重新录制；
- 中间速度不是优化目标；
- `track` 保留源时间线，但只匹配同步姿态；
- `pass` 沿路径尽快前进，目标姿态可以是不稳定状态，不做通过后验证；
- `reach-state` 只保留接口语义，暂不实现；
- `reach-stop` 以 `s=1` 为完成主轴，报告末端零速度误差，不做保持验证；
- `s` 允许少量超过 `1`，用于 `pass` 的路径外延；
- 不预设固定路径容差；只记录路径偏差，偏离多少由优化器决定；
- 输出 meta 记录目标、路径、进度、事件、优化参数、验证指标和 lineage；
- 不承诺数学全局最速，只报告给定求解器与预算下找到的结果。

## 1. 总体架构

```text
源 CoreClip
   │
   ├─ start_state = source.state(start_frame)
   ├─ source ref_action（可选热启动）
   └─ pose waypoints
        │
        ▼
TargetPath（连续姿态路径，无时间线）
        │
        ▼
ObjectiveEvaluator（track / pass / reach-stop）
        │
        ▼
PathSweepOptimizer（动作序列 + 后缀 rollout 评分）
        │
        ▼
final replay + CoreClipRecorder/录制器
        │
        ▼
输出 CoreClip + meta/report
```

建议新增/重构的文件：

| 文件 | 作用 |
|---|---|
| `target_path.py` | 连续姿态路径、路径参数化、采样、外延、投影 |
| `reclip_objectives.py` | 目标定义、rollout 评分、事件判定、指标汇总 |
| `reclip_engine.py` | 通用动作 rollout、路径版 sweep、最终完整回放录制 |
| `reclip.py` | 对外 API；保留旧接口兼容层 |
| `cli.py` | 新参数入口和旧参数映射 |
| `test_reclip_path.py` | 新路径/目标/优化器测试；现有测试继续回归 |

可以先不按文件拆这么细，但逻辑边界必须清楚：**路径、目标、求解器、录制**四件事不要混在一起。

## 2. TargetPath 实现

### 2.1 数据结构

建议定义：

```python
@dataclass
class PoseSample:
    s: float
    root_pos: np.ndarray       # (3,)
    root_rot: np.ndarray       # (4,), wxyz
    joint_pos_norm: np.ndarray # (21,)

@dataclass
class PathProjection:
    s: float
    pose: PoseSample
    errors: dict               # jp / rp / rq / total

class TargetPath:
    mode: str                  # "endpoints" | "all"
    source_clip_id: str
    s_end: float               # 固定为 1.0
    s_max: float               # 1.0 + s_ext
    root_pos_mode: str         # "absolute" | "free"
    pose_weights: dict         # resolved jp / rp / rq weights
```

核心 API：

```python
TargetPath.from_clip(
    clip: CoreClip,
    mode: str,
    start_frame: int = 0,
    end_frame: int = -1,
    s_ext: float = 0.1,
    pose_weights: dict | None = None,
    root_pos_mode: str | None = None,   # absolute | free
) -> TargetPath

path.sample(s: float) -> PoseSample
path.project(state: dict, s_hint: float, ds_max: float) -> PathProjection
path.frame_to_s(frame_index: int) -> float
```

`project` 只使用姿态字段：`root_pos`、`root_rot`、`joint_pos_norm`；`ds_max` 是进度搜索窗口，不是路径容差。

### 2.2 路径参数化

`endpoints`：

- 两个约束点：`start_frame`、`end_frame`；
- `s=0` 对应起点，`s=1` 对应末点；
- `s ∈ [0, 1+s_ext]`。

`all`：

- 所有源姿态作为有序 waypoint；
- waypoint 的 `s` 按相邻姿态的加权距离累计并归一化；
- `frame_to_s(k)` 保存源帧到路径参数的映射，供 `track` 使用；
- 这样密集采样不表示“应该更慢”，密集只表示路径上有更多约束点。

建议姿态距离用于路径参数化：

```text
pose_error = w_jp * jp_err + w_rp * rp_err + w_rq * rq_err
```

第一版可以复用现有 `frame_distance` 的姿态分量权重，但单独抽出 `pose_distance`，不要沿用速度分量。

`root_pos_mode` 决定 `rp` 是否进入上述总分：

- `absolute`：默认，`rp` 参与路径参数化、投影和 terminal pose error；
- `free`：`rp=0`，只匹配 `root_rot + joint_pos_norm`；`root_pos` 仍保留在路径和输出状态中，`components.rp` 作为诊断保留。

`free` 仅允许 `pass` / `reach-stop`；`track` 必须保持 `absolute`。`pose_weights` 应按“默认权重 + override 字段”合并，不能把未提供的 `jp/rq` 当成 0。

### 2.3 插值方式

第一版实现目标是“连续且少过冲”，不要求高阶平滑速度：

- `root_pos`、`joint_pos_norm`：分段单调三次 Hermite（PCHIP 风格）或受控 Catmull-Rom；
- `root_rot`：分段 slerp；第一版不强制旋转导数连续；
- `s > 1`：沿末端局部方向做有界外延；
- `s < 0`：第一版 clamp 到 `0`，不支持反向外延。

注意：插值路径只是目标路径，不要求每个路径点在动力学下可达；实际偏差由优化器和报告体现。

### 2.4 投影策略

对仿真状态 `q_k`：

```python
proj = path.project(q_k, s_hint=s_prev, ds_max=ds_max)
```

要求：

- 在 `[s_prev, min(s_prev + ds_max, s_max)]` 内搜索；
- `s_k >= s_{k-1}`，保证统一进度单调；
- 返回投影姿态和分量误差；
- 不做固定“走廊”判定，路径误差只进评分和报告。

`track` 不需要自由投影；它直接按源帧号取 `frame_to_s(k)`。

## 3. ObjectiveEvaluator 实现

建议新增目标配置：

```python
@dataclass
class ReclipObjectiveConfig:
    kind: str                  # "track" | "pass" | "reach-stop"
    source_frames: int         # track 使用
    time_budget_frames: int
    ds_max: float
    progress_weight: float
    path_weight: float
    reg_weight: float
    terminal_pose_weight: float
    terminal_vel_weight: float
    fall_weight: float = 0.0
```

目标 evaluator：

```python
class ObjectiveEvaluator:
    def evaluate_rollout(
        self,
        states: list[dict],
        actions: np.ndarray,
        path: TargetPath,
        start_s: float,
    ) -> RolloutEvaluation
```

返回：

```python
@dataclass
class RolloutEvaluation:
    score: float
    status: str                # completed / budget_exhausted
    event_frame: int | None
    progress: np.ndarray
    pose_errors: list[dict]
    terminal_errors: dict
    metrics: dict
```

### 3.1 `track`

- 输入时间预算 = 源片段帧数；
- 第 `k` 个输出帧的目标路径点为 `frame_to_s(k)`；
- 评分：同步姿态误差 + 动作正则 + 可选失稳惩罚；
- 不使用中间速度误差；
- 报告每帧姿态误差、平均/最大姿态误差。

兼容现有 `via` reclip 时应近似复现当前 sweep 行为，但去掉速度目标。

### 3.2 `pass`

- 每个落地状态做局部路径投影；
- 第一次 `s >= 1` 的控制帧记为 `event_frame`；
- 事件之后的轨迹不参与完成判定；
- 输出默认截断到 `event_frame`，可提供 `post_frames` 选项，默认为 `0`；
- 不要求目标姿态稳定，不做通过后 rollout 验证；
- 若预算内没有到达 `s=1`，记 `budget_exhausted`，并报告最大进度和路径偏差。

评分建议采用清晰的字典序/分层指标，而不是把所有指标混成一个含义不明的数：

```text
优先级 1：最早 event_frame（未到终点时退化为最大 s_final）
优先级 2：较小路径偏差
优先级 3：较小动作正则
```

如果现有 sweep 需要标量，可以把上述层级编码成 `score`，但 meta 中必须保存原始分量。

### 3.3 `reach-stop`

- 第一次 `s >= 1` 视为路径终点到达；
- 在事件帧报告：
  - `pose_error(q, q_end)`；
  - `vel_error(v, 0)`；
- 不做保持验证；
- 不把“未来是否会继续稳定”作为目标；
- 可先按 `pass` 的代码路径实现，再增加末端零速度误差。

### 3.4 `reach-state`

- 保留 CLI/API 枚举和文档语义；
- 当前实现中明确报 `NotImplementedError` 或 CLI `exit 2`；
- 不把它混入第一阶段开发，避免把软性末速度匹配和硬完成条件搅在一起。

## 4. 优化器实现

### 4.1 通用 rollout

从现有 `_roll_from` 抽象出：

```python
roll_actions(
    sim,
    robot: str,
    other_state: dict,
    start_state: dict,
    actions: np.ndarray,
    evaluator: ObjectiveEvaluator,
    path: TargetPath,
    start_s: float,
) -> RolloutEvaluation
```

该函数负责：

- `set_core_state` 恢复完整仿真状态；
- 逐 50ms 执行动作；
- 收集实际落地状态；
- 调 evaluator 得到路径进度、路径误差、事件和评分。

### 4.2 路径版 sweep

现有 `reclip_sweep` 的有效性来自“候选动作按完整后缀结果评分”。新版本保留该结构，但把固定帧号导轨评分替换为 `ObjectiveEvaluator`：

```python
reclip_path_sweep(
    start: dict,
    path: TargetPath,
    objective: ReclipObjectiveConfig,
    init_actions: np.ndarray,
    budget_frames: int,
    sweeps: int = 3,
    n_samples: int = 32,
    n_iters: int = 2,
    sigma0: float = 0.15,
    seed: int = 0,
) -> ReclipSolveResult
```

每帧候选动作的评价方式：

1. 从 `states[k]` 恢复仿真；
2. 替换 `actions[k]`；
3. 执行当前动作后缀；
4. 评估整段路径进度/事件/误差；
5. 选最好候选并更新后续实际落点。

这满足“每个 action 尽量沿路径走得远，同时考虑后续后果”的设计。

### 4.3 初始化

建议按目标选择：

- `track + all`：优先源 `ref_action`；没有则用路径姿态 PD target；
- `pass + endpoints`：路径姿态 PD target + 多 seed；
- `pass + all`：源 `ref_action` 优先，其次路径 PD target；
- 时间预算搜索：用上一次成功/最好结果做热启动。

### 4.4 最终完整回放

动作序列确定后，必须重新执行：

```python
sim.set_core_state(start_state)
for a in actions:
    sim.set_action(a)
    sim.physical_step × 25
    record actual CoreState
```

输出 Clip 使用这一次完整回放的状态和动作。随后计算回放验证指标：

- 关节姿态误差；
- 关节速度误差；
- 根位置误差；
- 根朝向误差；
- 根线速度误差；
- 根角速度误差。

由于回放就是生成过程，误差应在数值精度附近；报告仍要保留，防止以后修改生成路径。

## 5. CLI 设计

新接口建议：

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip reclip \
    --clip source.npz \
    --objective track|pass|reach-stop|reach-state \
    --path endpoints|all \
    [--start-frame 0 --end-frame -1] \
    [--time-budget 0.60 | --budget-frames 12] \
    [--s-ext 0.10 --ds-max 0.20] \
    [--method path-sweep --sweeps 3 --n-samples 32 --n-iters 2] \
    [--seed 0 --post-frames 0] \
    --out result.npz
```

默认建议：

- `track` 默认 `--path all`，禁止 `--path endpoints`；
- `pass` 默认 `--path endpoints`，允许 `--path all`；
- `reach-stop` 默认 `--path endpoints`；
- `reach-state` 目前报“语义保留，未实现”。

旧接口兼容：

- `--via guide.npz`：映射为 `track + all`；
- `--start/--end + --T`：映射为 `pass + endpoints` 或 legacy guide 模式；
- 旧 `greedy`/`mpc` 可保留为 legacy 方法，不作为新默认；
- 新默认方法为 `path-sweep`。

必须补的输出保护：

```text
resolve(out) == resolve(input_clip) -> exit 2
```

## 6. Meta 与报告

输出 `meta["reclip"]` 建议包含：

```jsonc
{
  "objective": "pass",
  "path": {
    "mode": "endpoints",
    "source_clip_id": "...",
    "start_frame": 0,
    "end_frame": -1,
    "s_end": 1.0,
    "s_ext": 0.1,
    "s_max": 1.1
  },
  "solver": {
    "method": "path_sweep",
    "budget_frames": 12,
    "sweeps": 3,
    "n_samples": 32,
    "n_iters": 2,
    "sigma0": 0.15,
    "seed": 0
  },
  "event": {"type": "pass", "frame": 9, "time": 0.45},
  "status": "completed",
  "metrics": {
    "s_final": 1.02,
    "path_error_mean": ...,
    "path_error_max": ...,
    "terminal_pose_error": ...,
    "terminal_vel_error": ...,
    "replay_error": {...}
  },
  "note": "fastest/earliest result found under this solver and budget; not a global minimum-time proof"
}
```

`source`：

```jsonc
{"type": "reclip", "parent": "<source_clip_id>"}
```

`lineage` 追加 `op="reclip"`，params 记录 objective/path/solver。

## 7. 分阶段实现

### 阶段 0：保护与兼容准备

任务：

- 给现有 `reclip` 增加输入/输出同路径保护；
- 整理现有 `greedy`、`mpc`、`sweep` 为 legacy method；
- 确认 `check_ref_action` 能报告完整状态误差，必要时扩展指标。

验收：

- 现有测试 `test_reclip_executable_by_construction` 仍通过；
- `--out` 等于输入路径时返回 exit 2；
- 当前 extend reclip 流程不退化。

### 阶段 1：`TargetPath`

任务：

- 实现 `PoseSample`、`PathProjection`、`TargetPath`；
- 实现 `endpoints` 和 `all`；
- 实现姿态分量误差；
- 实现 `sample`、`project`、`frame_to_s`；
- 实现 `s_ext` 外延。

测试：

- endpoints `sample(0)` 等于首帧姿态，`sample(1)` 等于末帧姿态；
- `all` 对每个源 waypoint `sample(frame_to_s(k))` 等于源姿态；
- `project` 单调且只在局部窗口内前进；
- `s > 1` 有界并返回有限姿态；
- 四元数插值始终是单位四元数。

### 阶段 2：`ObjectiveEvaluator`

任务：

- 实现 `RolloutEvaluation`；
- 实现 `track`、`pass`、`reach-stop` 评分和指标；
- `reach-state` 明确报未实现；
- 加入动作正则和可选失稳惩罚。

测试：

- `track` 不使用速度分量；
- `pass` 第一帧 `s>=1` 记录事件；
- `pass` 不要求目标稳定性；
- `reach-stop` 报告零速度误差但无保持验证；
- 没有到达 `s=1` 时 status 为 `budget_exhausted`。

### 阶段 3：路径版 sweep + `pass + endpoints`

任务：

- 抽象 `roll_actions`；
- 实现 `reclip_path_sweep`；
- 实现最终完整回放录制；
- 先接入 `pass + endpoints`；
- 支持多 seed / 初值；
- 输出默认截断到事件帧。

测试：

- 小型短过渡 clip：输出可执行，完整回放误差接近 0；
- `s_final >= 1` 或如实报告 `budget_exhausted`；
- meta 记录事件、进度和路径误差；
- 同 seed 输出可复现。

验收建议：

- 在 `hdm05_3146_88_2895_127` 这类单向 jab 上跑 `pass + endpoints`；
- 渲染视频，确认运动是从初始姿态推进到打击姿态，而不是静态或提前退化。

### 阶段 4：`track + all` 回归

任务：

- `via` 映射到新 `track + all`；
- 使用源 `ref_action` 热启动；
- 保持 20Hz 时间线；
- 对比旧 sweep 输出。

测试：

- 输出 T 与源片段一致；
- 回放完整状态误差接近 0；
- 同步姿态误差被记录；
- 中间速度不作为目标，但输出速度是仿真真实值。

验收建议：

- 对四条 extend clip 分别跑旧 `via` 和新 `track + all`；
- 比较 final path/pose error 和视频；
- 新版本不应明显差于旧版本。

### 阶段 5：`pass + all` 与时间预算搜索

任务：

- 在 `pass` 中启用 `all` 路径；
- 保留源姿态顺序和动作形态；
- 增加多预算运行接口；
- 使用上一次最好动作序列热启动；
- 多 seed 取最早完成结果。

验收：

- 同一 clip 的 `endpoints` 与 `all` 结果可比较；
- `all` 保留更多原动作形态，`endpoints` 更自由；
- 报告最短“搜索得到”的通过帧，不宣称全局最速。

### 阶段 6：`reach-stop`

任务：

- 复用 `pass` 的进度推进；
- 到达 `s=1` 时报告末端姿态误差和零速度误差；
- 不做保持验证；
- 支持提前制动权重。

验收：

- 对简单单向过渡能生成可执行 clip；
- meta 明确显示终端零速度误差；
- 没有稳定验证字段。

### 阶段 7：CLI、文档与真实数据回归

任务：

- 新 CLI 参数落地；
- 旧参数映射和报错信息；
- `USAGE.md` 更新；
- `CORECLIP_SPEC.md` 补充 reclip meta 字段；
- extend 四条 clip 全量回归；
- 输出 guide/reclip 对比视频；
- 运行 `check` 的 `basic`、`ref_action` 节。

验收：

- 四条 extend clip 可生成可执行输出；
- `ref_action` 回放通过；
- 文档中的命令可直接复制运行；
- 不覆盖源文件。

## 8. 测试清单

新增测试建议：

```text
test_target_path_endpoints
test_target_path_all_waypoints
test_target_path_projection_monotonic
test_target_path_extension_bounded
test_pose_error_components
test_objective_track_ignores_mid_velocity
test_objective_pass_event_frame
test_objective_reach_stop_reports_zero_vel
test_path_sweep_executable_output
test_reclip_final_full_replay
test_reclip_cli_new_args
test_reclip_rejects_same_input_output
test_reclip_metadata_and_lineage
```

回归测试：

```text
pytest mocap/humanoid21/clip/test_core_clip.py
pytest mocap/humanoid21/clip/test_reclip_path.py
```

真实数据 smoke：

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.clip reclip \
    --clip /tmp/reclip_work/hdm05_3146_88_2895_127_ext.npz \
    --objective pass --path endpoints \
    --time-budget 0.60 \
    --out /tmp/reclip_work/jab_path_pass.npz
```

## 9. 主要风险与处理

| 风险 | 表现 | 处理 |
|---|---|---|
| 路径投影跳进度 | 路径自交或姿态重复时直接跳到末端 | 局部 `ds_max` 搜索 + 单调进度；报告异常跳变 |
| 插值过冲 | 关节/根位置走出不合理范围 | 标量维度用 PCHIP/受控插值；路径采样检查越界 |
| 自由进度导致局部最优 | 优化器停在路径中段 | 多 seed、源 action 热启动、路径 PD 初值、多 sweep |
| “最快”误导 | 结果只是当前预算/求解器下较早 | meta/文档明确“非全局最速” |
| 目标姿态不可达/不稳定 | `s=1` 但姿态误差大 | 报告姿态误差；不要求稳定，不假报成功 |
| 搜索成本高 | 多 budget/seed 慢 | 先短 clip、阶段化预算、复用热启动、可后台并行 |
| 旧接口回归 | `via` 用户依赖原行为 | `track + all` 回归对比；保留 legacy 方法 |
| replay 验证只查关节 | 掩盖根/速度不一致 | 最终回放检查全部 CoreState 字段 |

## 10. 第一版完成定义（DoD）

第一版实现完成后，应至少满足：

1. `track + all` 可替代当前 `via` reclip；
2. `pass + endpoints` 能在短单向 clip 上生成可执行输出；
3. `pass` 以控制帧判断 `s >= 1`，输出可截断到事件帧；
4. `reach-stop` 有基础实现并报告末端零速度误差；
5. `reach-state` 有明确“未实现”错误；
6. 最终输出经过完整回放录制和全字段验证；
7. meta 完整记录 objective/path/solver/event/metrics/lineage；
8. 所有新测试和现有测试通过；
9. 四条 extend clip 至少完成 `track + all` 回归，且至少一条完成 `pass + endpoints` 实测视频。
