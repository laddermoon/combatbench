# humanoid21 动作资产数据契约 (DATASPEC)

> 类型：契约 —— M0 交付物之一
>
> 范围：本模块产出的所有源运动片段与机器人参考运动片段。

## 1. 存储布局

```
<library_root>/                       # 大体积工件默认在 /data1/dev/combatbench_mocap/
├── index.json                        # manifest：全库索引（本目录内若入库则 git add -f）
└── clips/
    └── <clip_id>.npz                 # 每片段一个文件
```

规则：npz 与原始大文件不入 git；`index.json`、本契约、审计报告入库（`git add -f`）。
每条 clip 的 index 记录其 npz 的 sha256，保证可追溯。

## 2. clip_id 规范

`<src>_<subject>_<trial>[_seg<k>][_m]`，例如 `cmu_13_17`、`cmu_13_17_seg0`、`cmu_14_01_m`（镜像）。
切分/镜像派生片段通过 `identity.parent_clip` 关联，必须在 index 中登记派生参数。

**派生类型（`identity.derivation`）**：

| 值 | 含义 | 附加字段 |
|---|---|---|
| `segment` | 父片段的帧区间切片 | `frame_range: [s, e)`（120Hz 帧号） |
| `mirror_sagittal_x` | 矢状面镜像（左右对调，前进方向不变） | 无（镜像语义固定） |

派生片段不重复写 `_src` npz；`npz_sha256` 记录派生产物本身。镜像在重定向
目标层完成（源骨架世界位姿做 x→-x 反射 + l/r 语义对调后重新 IK），不是对
导出数据的逐通道符号变换——骨骼非严格对称时几何上更安全。

## 3. npz 字段（统一运动表示）

所有数组 float32（除标记外），时间维 T 均为首维。**坐标系一律 MuJoCo 世界系**：
z 向上、米、弧度、四元数 wxyz——转换在导入时完成，不在下游做。

| 字段 | 形状 | 说明 |
|---|---|---|
| `timestamps` | (T,) | 秒，单调递增；源帧率与重采样见 index |
| `root_pos` | (T,3) | 根部位置（世界系，米） |
| `root_quat` | (T,4) | 根部朝向 wxyz |
| `root_linvel` | (T,3) | 根部线速度（世界系，m/s）；由位置差分得到时须在 index 注明方法 |
| `root_angvel` | (T,3) | 根部角速度（世界系，rad/s） |
| `joint_angles` | (T,21) | 弧度，顺序 = `Humanoid21Meta.CONTROLLED_JOINTS`（源骨架片段此字段可为对应映射骨架 DOF，由 index `joint_space` 区分） |
| `joint_velocities` | (T,21) | 弧度/秒；差分推导须注明方法 |
| `keypoints` | (T,K,3) | 关键点世界坐标；K 与名称在 index `keypoint_names` |
| `foot_contacts` | (T,2) | [左,右] 0/1 或置信度；估计方法必须在 index `quality.contact_method` 记录 |
| `valid_mask` | (T,) uint8 | 1=有效帧；缺失帧不得填零伪装 |

## 4. index.json 结构

```jsonc
{
  "spec_version": 1,
  "clips": {
    "cmu_13_17": {
      "identity": {
        "source": "cmu", "subject": "13", "trial": 17,
        "parent_clip": null,           // 派生片段填父 clip_id
        "source_file": "baseline/humanoid21/mocap/data/raw/13_17.amc",
        "source_sha256": "...",
        "npz_sha256": "...",
        "version": 1
      },
      "time": {
        "src_fps": 120.0, "fps": 120.0,   // fps = 重采样后
        "frames": 4840, "duration_s": 40.33
      },
      "frames_convention": "mujoco_world_z_up",
      "joint_space": "humanoid21_v2",     // 或 "cmu_asf"（纯源数据）
      "classification": {
        "labels": ["boxing"],            // 多标签数组，非互斥
        "phase_segments": [],            // [{start,end,phase:"prepare|execute|recover"}]
        "label_source": "cmu_official",  // cmu_official | auto_feature | manual
        "official_description": "boxing",
        "confidence": "high"
      },
      "quality": {
        "level": "L1",                   // 见 §5
        "kinematic_checks": {"limit_violations": 0, "max_overshoot_deg": 0.0},
        "physics": null,                 // L2 时填 {method, tracking_err, ...}
        "issues": [],                    // ["foot_penetration", ...]
        "rejected": false, "reject_reason": null
      },
      "adaptation": {
        "model": "battle_circular_v2.xml", "model_sha256": "...",
        "retarget_method": null, "retarget_params": null
      }
    }
  }
}
```

## 5. 验收等级

| 等级 | 含义 | 判据 |
|---|---|---|
| L0 | 已导入 | 解析成功、字段齐全、时间单调 |
| L1 | 运动学检查通过 | 限位/肌腱约束/速度异常/缺帧检查通过或显式标注 issue |
| L2 | 物理验证通过 | 在当前模型+控制器下跟踪成功（不逐帧覆写状态），记录跟踪误差 |
| L3 | 先验可用 | 在先验/跟踪训练中实际使用且诊断正常 |
| rejected | 不可用 | `reject_reason` 必填；不得静默修补混入 |

等级只升不降的例外：模型或契约变更后须重验。

## 6. 操作规则（对应大纲 §5.3）

- 速度、接触可由原始数据推导，**方法必须记录**；缺失填 `valid_mask=0`，不填零。
- 标签支持组合（如 `["walk","punch"]`）；阶段标签粗分 prepare/execute/recover。
- 训练/验证划分按**源 clip 或 subject** 进行，禁止按帧切；镜像增强不计独立片段。
- 若片段含控制 action（暂不应有），须在 `adaptation.controller` 记录控制器、
  频率、模型版本；action 与参考关节角不得混用同一字段。
