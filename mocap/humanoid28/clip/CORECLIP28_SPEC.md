# CoreClip28 格式契约

Humanoid28（MimicKit DeepMimic 系 humanoid，28 hinge dof，pelvis 根）
的 sim 原生 clip 格式。结构与 `mocap/humanoid21/clip/CORECLIP_SPEC.md`
同构——本文只记 h28 的差异点。

## 1. 与 humanoid21 的契约差异

| 项 | humanoid21 | humanoid28 |
|---|---|---|
| CoreState 关节字段 | `joint_pos_norm` / `joint_vel_norm`（归一化） | **`joint_pos` / `joint_vel`（原始弧度 / rad·s⁻¹）** |
| DOF | 21 | 28 |
| root body | torso | pelvis |
| ref_action 语义 | `pd_target_norm` ∈[-1,1] | **`pd_target_rad`（弧度 PD 目标）** |
| 语义关键点 | 7（`foot_left` 风格） | 7（side-first：`pelvis/torso/head/right_foot/left_foot/right_hand/left_hand`） |

共同点：6 个 CoreState 字段与 `sim.get_core_state()` 逐键对应；
20Hz `control_dt`；meta_json / lineage / rsi_bad_frames / labels /
split 语义一致；`keypoint_pos` 为可选 sidecar（世界系，FK 自状态）。

## 2. npz 数组

| 字段 | 形状 | 约定 |
|---|---|---|
| `root_pos` | (T,3) f32 | 世界系，米（pelvis） |
| `root_rot` | (T,4) f32 | wxyz 单位四元数 |
| `root_vel_local` | (T,3) f32 | 机体系线速度 m/s |
| `root_angular_vel_local` | (T,3) f32 | 机体系角速度 rad/s |
| `joint_pos` | (T,28) f32 | **原始弧度**，CONTROLLED_JOINTS 序 |
| `joint_vel` | (T,28) f32 | **原始 rad/s** |
| `ref_action` | (T,28) f32，可选 | 弧度 PD 目标（= joint_pos） |
| `keypoint_pos` | (T,7,3) f32，可选 | 语义关键点世界系位置 |
| `meta_json` | str | clip_id/labels/split/issues/质量/谱系 |

注意 h28 的 MimicKit 扩展 action bounds 允许 PD 目标**超出**机械限位
——`ref_action` 合法域是 action bounds 而非 jnt_range。

## 3. RSI 用法

与 h21 相同：每帧是完整物理状态，`sim.set_core_state({robot_id:
clip.state(k)})` 直接落位；采样时跳过 `meta.rsi_bad_frames`
（>0.6 rad 单帧 dof 跳变 + 首尾各 2 帧端点）。

## 4. 生成管线

```
AMASS smplx npz
  → gmr/build_core.py     (GMR/mink IK + 肘部直接映射 bias; 30fps PHC)
  → phc/combatbench_<bundle>_v1.pkl
  → clip phc2core          (重采样 20Hz + 速度差分 + FK keypoints)
  → data/<bundle>/*.npz
```

h28 特有：无腱约束（mimickit 形态无 tendon）；地面校准以
**足底面**为准（foot body 系原点下 0.05 m），非 body 原点。
