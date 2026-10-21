# CoreClip — Humanoid21 专用 Clip 格式契约

> 类型：契约 —— Motion Workbench 第一阶段
>
> 范围：`mocap/humanoid21/clip/` 产出的所有 CoreClip 文件。

## 1. 定位

CoreClip 是**绑定 Humanoid21 仿真器**的动作记录格式：

- 每帧间隔固定 **50ms（20Hz 控制率）**，不存时间戳数组。
- 每帧内容 = 仿真器 `get_core_state()[robot_id]` 的全部六个字段，
  可直接喂给 `set_core_state`（运动学回放），无需任何转换。
- 可选的逐帧参考动作 `ref_action`（归一化 PD 目标指令）。

**与 retarget npz 的关系**：`clips/<id>.npz`（`mocap.humanoid21.tracking`）
是重定向产物——世界系 root 速度、弧度关节、120Hz。CoreClip 使用
**机体系速度 + 归一化关节 + 20Hz**，约定完全不同，不可混用；
`convert.refclip_to_coreclip` 负责单向转换。

## 2. 文件结构

单个 `.npz` 文件 = 数组 + `meta_json`。默认库根
`/data1/dev/combatbench_mocap/core_clips/`（大文件不入 git）。

| 字段 | 形状 | 约定 |
|---|---|---|
| `root_pos` | (T,3) f32 | 世界系，米 |
| `root_rot` | (T,4) f32 | wxyz 单位四元数 |
| `root_vel_local` | (T,3) f32 | **机体系**线速度，m/s |
| `root_angular_vel_local` | (T,3) f32 | **机体系**角速度（qvel[3:6] 直通），rad/s |
| `joint_pos_norm` | (T,21) f32 | `CONTROLLED_JOINTS` 顺序，norm=(q−mid)/half |
| `joint_vel_norm` | (T,21) f32 | norm=qv/half（无量纲/s） |
| `ref_action` | (T,21) f32，可选 | 归一化 PD 目标指令，∈[−1,1] |
| `meta_json` | 标量 str | UTF-8 JSON，见 §3 |

归一化基准（mid/half）= Humanoid21 关节限位
（`Humanoid21Meta` / `robot_spec.json`，与 `RefClip` 同源），
meta 中只记 `joint_space="humanoid21"` 标识。

## 3. meta_json

```jsonc
{
  "spec_version": 1,                    // 必填
  "clip_id": "jab_r_core",              // 必填
  "robot": "humanoid21",                // 固定
  "joint_space": "humanoid21",          // 固定
  "control_dt": 0.05,                   // 固定
  "phy_dt": 0.002,                      // 固定
  "model": {"file": "battle_circular_v2.xml", "sha256": "..."},
  "source": {                           // 必填，溯源
    "type": "refclip | policy_rollout | slice | edit | generated",
    "ref_clip_id": "cmu_...",           // type=refclip 时
    "ctrl_range": [0, 61],
    "agent_id": "robot_a"               // rollout 时
  },
  "ref_action_semantics": "pd_target_norm",  // 有 ref_action 时必填
  "notes": "free text"
}
```

**`ref_action` 因果语义**：`ref_action[k]` 是在状态 k 发出的指令，
目标是产生状态 k+1。录制自策略 rollout 时是当时实际发出的
action；转换自 RefClip 时是该帧参考关节角的归一化 PD 目标。

## 4. 约定对照表（防混淆）

| 项 | retarget npz | CoreClip |
|---|---|---|
| 帧率 | 120Hz | **20Hz（50ms）** |
| root 线速度 | 世界系 | **机体系** |
| root 角速度 | 世界系 | **机体系** |
| 关节量 | 弧度 / 弧度每秒 | **归一化** |
| 关节顺序 | joint_names | `CONTROLLED_JOINTS`（固定，不存） |
| 用途 | 重定向中间产物 | 仿真器直接回放/训练 |

## 5. 校验规则

`CoreClip.validate()`：硬错误 raise `ValueError`，软问题返回 warning。

- 硬错误：字段缺失、形状不为 (T, 指定维数)、四元数非单位（dev>1e-3）、
  NaN/inf、`ref_action` 越出 [−1,1]、meta 缺 `spec_version`/`clip_id`。
- 软 warning：`joint_pos_norm` 超 [−1,1] 少量（>0.05 容差）、
  有 `ref_action` 但缺 `ref_action_semantics`。

## 6. 回放语义（重要）

`play` 是**运动学回放**：逐帧 `set_core_state` 覆写状态后渲染。
它展示记录下来的运动外观，**不代表该运动可被机器人通过
动作指令执行**。可执行性验证见 `check` 的 `ref_action` 回放节。

## 7. 统一 CLI

所有 clip 操作经单一入口（`clip/cli.py` + `clip/__main__.py`）：

```
PYTHONPATH=. python3 -m mocap.humanoid21.clip <cmd>
  check    --clip x.npz [--only basic,velocity,actuator,ref_action]
           [--robot robot_a] [--json out.json]
  play     --clip x.npz --out x.mp4 [--fps 20 --slow 2 --burn-frame]
  fix      --clip x.npz [--mode lowest|constant --margin .002]
  convert  --ref-id <id> --dir <clips_dir> --out x.npz [--start --end]
  info     --clip x.npz
```

`check` 四节（`--only` 可选子集），每节输出
`PASS/WARN/FAIL/SKIP` + findings + metrics + 修复建议命令：

| 节 | 内容 |
|---|---|
| `basic` | 格式校验；脚部穿地（胶囊最低点 + solver 穿透）；全身 geom 穿地；同机器人自碰撞；非脚 geom 触地；躯干倾倒 |
| `velocity` | 记录速度 vs 20Hz 位置差分反推速度（关节 norm、root 线/角速度）；**加速度接缝**（逐帧独立编辑检测）；首尾帧速度信息性报告。注：120Hz 源下采样 clip 的记录速度携带真实高频信息，残差属混叠而非错误——硬错误仅"记录速度与位置趋势反向" |
| `actuator` | `mj_rne` 逆动力学需求筛查（忽略接触，支撑腿为上界）；逐关节力矩利用率 vs `gear×ctrlrange`；反推 PD 指令是否越界 [-1,1]。**仅筛查，非可执行性证明** |
| `ref_action` | 独立节：把记录的 action 序列顺向灌入物理回放（逐 50ms），对比下一帧目标状态；报告跟踪误差、发散帧、跌倒帧。无 `ref_action` 时报 SKIP。**这是唯一的真可执行性测试** |

## 8. 工具 API

- `CoreClip.load/save`、`clip.state(k)`（CoreState dict）、
  `clip.state_at(t)`（lerp + slerp 连续采样）、`clip.slice(k0,k1)`、
  `CoreClip.from_states(list_of_core_states)`。
- `convert.refclip_to_coreclip(ref, start, end)` —— 现有 mocap → CoreClip。
- `CoreClipRecorder.record_sim(sim, action)` —— 运行中录制。
- `checks.run_checks(clip)` —— 程序化检查入口（返回 `SectionResult` dict）。
- `fix.fix_ground_penetration(clip)` —— root-lift 穿透修复，
  修复结果带 `source.type="fix_ground_penetration"` 溯源；仅抬
  root_z，不改关节/速度，脚部仍可能虚浮（源数据腿不一致时需
  IK 修复）。
