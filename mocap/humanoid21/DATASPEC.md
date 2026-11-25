# humanoid21 动作资产数据契约 (DATASPEC)

> 类型：契约
>
> 范围：本模块产出的所有机器人参考运动片段。
>
> **spec_version: 2**（2026-11）。canonical 格式定为 **PHC 系 pkl**
> （PHC/PBHC/OmniH2O/ASAP/HumanoidVerse 生态的消费标准），由
> **GMR**（`mocap/gmr/`）重定向 AMASS smplx 数据产出。
> v1 的 npz 契约（120Hz、wxyz、`joint_angles`/`keypoints` 分字段）降为
> legacy，仅约束存量 `cmu_*` 库，见附录 A。

## 1. 存储布局

```
<library_root>/                        # 大体积工件默认 /data1/dev/combatbench_mocap/
├── phc/
│   ├── combatbench_core_v1.pkl        # bundle = {clip_id: clip_dict}，共享基准必选集
│   ├── combatbench_extended_v1.pkl    # 扩展池
│   └── manifest.json                  # bundle 级元数据（入库 git add -f）
└── clips/                             # legacy v1 npz（存量，不再新增）
```

规则：pkl 与原始大文件不入 git；`manifest.json`、本契约、审计报告入库。
manifest 记录每条 clip 的 `src_sha256` 与 bundle 的 `sha256`，保证可追溯。

## 2. clip_dict 字段

所有数组 float32，时间维 T 为首维。**坐标系一律 MuJoCo 世界系**（z-up、
米、弧度）；**四元数为 scipy/PHC 约定的 xyzw**——注意与 MuJoCo 原生
qpos 的 wxyz 相反，导出层统一转换，下游不得再换序。

### 必填（PHC stock 字段，外部消费者直接读）

| 字段 | 形状 | 说明 |
|---|---|---|
| `fps` | int | **固定 30**；源帧率记于 meta.src_fps |
| `root_trans_offset` | (T,3) | 根（torso freejoint）世界位置，米 |
| `root_rot` | (T,4) | 根四元数 **xyzw**（= qpos[3:7] 的 wxyz 换序） |
| `dof` | (T,21) | 关节角，弧度，顺序 = manifest `dof_names`（CONTROLLED_JOINTS 序） |
| `pose_aa` | (T,22,3) | 轴角表示的全身姿态：`[:,0]`=根 rotvec，`[:,1+i]`=`dof[:,i]·dof_axis[i]`。消费者免 MJCF 即可 FK 全部 link 位姿 |
| `smpl_joints` | (T,J,3) | 人侧关节世界位置（SMPL-X FK，J=22 body joints）；名称在 manifest `smpl_joint_names` |

### 扩展（本项目自用；stock consumer 忽略未知键）

| 字段 | 形状 | 说明 |
|---|---|---|
| `dof_names` | (21,) str | 冗余存一份使 clip 自解释 |
| `keypoint_pos` | (T,16,3) | 全部非根 body 原点世界位置（导出时 FK，便利+一致性件）；名在 manifest `keypoint_names` |
| `foot_contacts` | (T,2) 可选 | [左,右] 0/1；估计方法记 meta.contact_method |
| `meta` | dict | 见下 |

```python
meta = {
    "src_npz": "ACCAD/Male2MartialArtsPunches_c3d/G1 - jab cross_poses.npz",
    "src_tree": "smplx_from_smplh",        # 或 smplx_gender_specific
    "src_sha256": "...",                    # 源 npz 哈希
    "src_fps": 120.0,
    "labels": ["strike_punch"],             # 分类见 DATASET.md §6
    "split": "train",                       # train | val，subject 级划分
    "dataset": "core",                      # core | extended | null
    "parent_clip": null,                    # 派生片段填父 clip_id
    "retarget_method": "gmr_mink_ik",
    "retarget_params": {"ik_config": "smplx_to_humanoid21.json",
                        "tgt_fps": 30, "burn_in": 10,
                        "scale_table_flat": 0.9},
    "quality_level": "L1",
    "issues": ["foot_penetration"],         # 显式记录，不得静默修补
}
```

### 不存的字段

- **速度**（`dof_vel`/`root_vel`/`root_ang_vel`）：PHC stock 不存，消费者
  按 fps 有限差分（约定 `np.gradient(t, 1/fps)`）；manifest 统一注明。
  存了反而是第二种真源。
- **action/ctrl**：运动学资产不含控制量；带 action 的记录属于
  `clip/CORECLIP_SPEC.md` 管辖区。

## 3. pose_aa 构造约定

```
pose_aa[:, 0]   = Rotation.from_quat(root_rot).as_rotvec()          # (T,3)
pose_aa[:, 1+i] = dof[:, i:i+1] * dof_axis[i]                       # i=0..20
```

`dof_axis` (21,3) 为各 hinge 轴在父系下的单位向量，从
`humanoid21_mocap.xml` 抽取，存于 manifest。这是 PBHC 的标准构造
（`dof_axis * dof[:,:,None]`），PHC 训练管线对它 FK 即得全体 link
位姿。

## 4. clip_id 规范

AMASS 来源：`<subset_lower>_<相对目录扁平化>_<文件slug>`——
`SSM/20160330_03333/walking_01_stageii.npz` → `ssm_03333_walking_01`；
`ACCAD/Male2MartialArtsPunches_c3d/G1 - jab cross_poses.npz` →
`accad_m2ma_punches_g1_jab_cross`。目录缩写表与 slug 规则由导出
工具实现并在 manifest 登记。

**派生类型（`meta.parent_clip` + clip_id 后缀）**：

| 后缀 | 含义 | 附加字段 |
|---|---|---|
| `_seg<k>` | 父片段帧区间切片 | `frame_range: [s,e)`（30fps 帧号） |
| `_m` | 矢状面镜像（左右对调） | 无（语义固定） |

镜像在**重定向目标层**完成（人侧关键点做 x→-x 反射 + l/r 对调后重跑
IK），不是对导出数据的逐通道符号变换——骨骼非严格对称时几何上更安全。

## 5. manifest.json 结构

```jsonc
{
  "spec_version": 2,
  "quat_order": "xyzw",
  "velocity_method": "finite_difference_np_gradient",
  "robot": {
    "model_xml": "mocap/humanoid21/gmr/humanoid21_mocap.xml",
    "model_sha256": "...",
    "dof_names": ["abdomen_z","abdomen_y","abdomen_x","hip_x_right", "..."],
    "dof_axis": [[0,0,1], "..."],          // (21,3)
    "keypoint_names": ["torso","head","waist_lower","pelvis", "..."], // 16
    "smpl_joint_names": ["pelvis","left_hip", "..."]                   // 22
  },
  "bundles": {
    "combatbench_core_v1.pkl":     {"n_clips": 145, "sha256": "..."},
    "combatbench_extended_v1.pkl": {"n_clips": 0,   "sha256": "..."}
  },
  "clips": {
    "accad_m2ma_punches_g1_jab_cross": {
      "frames": 58, "duration_s": 1.93, "split": "train",
      "labels": ["strike_punch"], "dataset": "core",
      "src_sha256": "...", "quality_level": "L1", "issues": []
    }
  }
}
```

## 6. 验收等级

| 等级 | 含义 | 判据 |
|---|---|---|
| L0 | 已导出 | 字段齐全、T 一致、全有限值、fps=30 |
| L1 | 运动学检查通过 | 限位/肌腱/速度尖峰/帧跳变/穿透检查通过或显式记 issue |
| L2 | 物理验证通过 | 在当前模型+控制器下跟踪成功（不逐帧覆写），记录跟踪误差 |
| L3 | 先验可用 | 在先验/跟踪训练中实际使用且诊断正常 |
| rejected | 不可用 | `reject_reason` 必填；不得静默修补混入 |

等级只升不降的例外：模型或契约变更后须重验。
core 集准入一票否决：`invalid_frames`、`frame_jump`、
`root_orientation_jump`；容许但记录：`foot_penetration`、
`high_residual_*`、`joint_velocity_spike`。

## 7. 操作规则

- 缺失帧/坏帧：`foot_contacts`/`keypoint_pos` 允许导出工具标
  `valid_mask`，缺失不得填零伪装。
- 标签多标签组合；训练/验证按 **subject** 划分（DATASET.md §7），
  镜像派生不算独立片段、随父 clip 同侧。
- 每条 clip 的 issues 显式登记在 `meta.issues` 与 manifest 两处；
  不得以修剪方式"洗白"。
- bundle 重导出（改 ik_config/模型）必须 bump bundle 版本号
  （`_v2`），旧 bundle 保留供回归。

## 8. 相关格式

- **CoreClip**（`clip/CORECLIP_SPEC.md`）：绑定仿真器的 20Hz 动作
  记录格式（机体系速度、归一化关节、可选 ref_action），与本契约的
  重定向资产约定不同、不可混用。
- **GMR 中间产物**（`retarget_out/*.pkl` 的 `{fps,root_pos,root_rot,
  dof_pos}`）：工具内部格式，不是交付格式；经导出工具转本契约。
- **ProtoMotions MotionLib**（`.motion`）：外部生态消费格式，需要时
  从本契约转换生成，不做真源。
- **DATASET.md**：core/extended 集的范围、标签体系、划分规则。

## 附录 A：legacy v1（npz 契约）

存量 `cmu_*` 库（`/data1/dev/combatbench_mocap/clips/*.npz`，1264 条）
仍按 v1 契约解释：`timestamps/root_pos/root_quat(wxyz!)/root_linvel/
root_angvel/joint_angles(21)/joint_velocities/keypoints(15)/foot_contacts/
valid_mask`，120Hz，字段表见 git 历史中的 DATASPEC v1。
**四元数序差异是最大迁移坑**（v1=wxyz，v2=xyzw）；新旧数据混用前
必须显式换序。v1 库不再新增条目，新方法一律以 PHC bundle 为准。
