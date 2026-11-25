# humanoid21 共享运动数据集（DATASET）

> 类型：契约 —— 定义"所有基线方法复现共同消费的数据集范围"。
> 字段格式见 `DATASPEC.md`（spec_version 2，PHC 系 pkl）；本文只管
> **哪些 clip 进集、怎么命名、怎么打标签、怎么划分**。

## 1. 设计目标

所有 baseline 复现（DeepMimic 系 tracking、AMP/ASE/CALM latent、
MaskedMimic/HOVER masked-goal、GPC/NCP 离散化、FAIR/NCP/SLMP/RoboStriker
格斗流水线）吃**同一份参考库**。方法间可比性的公约数 = 相同的
clip 集合 + 相同的字段契约 + 相同的质量标记。

## 2. 方法需求矩阵（格式的充分性论证）

每帧世界系机器人状态 + 人侧参考是全部方法的公共超集：

| 方法族 | 需要的字段 | 本契约覆盖 |
|---|---|---|
| DeepMimic/PHC/FAIR/NCP/GPC/BeyondMimic tracking | q/root pos·quat、link 位姿、相位 | ✓ dof/root/pose_aa 全 FK 可解 |
| AMP/ASE/CALM/SLMP 判别器 Φ(s,s′) | 根速度+关节速度（差分）+末端局部位置 | ✓ pose_aa FK + 差分约定 |
| MaskedMimic/HOVER masked-goal | 任意子集的未来轨迹 → 全 body 位姿 + 关节 | ✓ pose_aa/keypoint_pos |
| RSI/GSI 初始化 | 任意有效帧状态 | ✓ |
| SMP 扩散 | H 帧连续片段 → 滑窗 | ✓ |
| PFSP/自博弈 | 无新增数据需求（消费先验） | — |
| **人侧对照**（残差评估、human-feature 基线） | SMPL-X 关节位置 | ✓ `smpl_joints` |

结论：**canonical 表示 = PHC 系 pkl dict（DATASPEC v2）**——
PHC/PBHC/OmniH2O/ASAP 生态直接消费，不引入第二种真源；
ProtoMotions MotionLib 等外部格式一律由转换工具按需生成。

## 3. 集合分层

| tier | bundle 文件 | 含义 | 用途 |
|---|---|---|---|
| `core` | `combatbench_core_v1.pkl` | **必选集**，所有基线必须在其上训练/报告 | 方法间可比 |
| `extended` | `combatbench_extended_v1.pkl` | 可选池，方法可自由使用 | 扩展覆盖；报告须与 core 分开标注 |
| （不入 bundle） | — | 试验/调试用 | — |

`meta.dataset` 与 manifest 同时登记 tier（DATASPEC §2/§5）。

## 4. core 集构成（v1，已按磁盘实盘点）

原则：**格斗为主、步态够用、规模可控**。全部来自 AMASS→GMR 管线
（`retarget_method="gmr_mink_ik"`），30fps。**权威清单 = `gmr/build_core.py`
的 `clip_list()`**（下表为其结果的分类汇总；校准/空白 take
`*Calibration*`/`*Subject Cal*`/`*blanks*` 不入集）。

| 类别 | 来源（`amass/` 下相对路径前缀） | 数量 |
|---|---|---|
| `strike_punch` | `ACCAD/Male2MartialArtsPunches_c3d/`（jab/cross/hook/uppercut 全套 18）+ SSM `03333/punching`、`50032`·`50033/punch_kick`（3）+ CMU `13_17`·`13_18`（boxing）、`144_13/14/20/21`（punch sequences）、`135_09`（lunge punch）、`111_19`（punch kick，8） | **29** |
| `strike_kick` | `ACCAD/Male2MartialArtsKicks_c3d/`（front/side/back/spinning/axe/push，23 − 1 cal = 22）+ SSM `50032`·`50033/dribble_kick`、`50032/special_move`（3） | **25** |
| `stance_guard` | `ACCAD/Male2MartialArtsStances_c3d/`（17 − 1 blanks = 16）+ `ACCAD/Male2MartialArtsExtended_c3d/`（4） | **20** |
| `footwork` | `ACCAD/MartialArtsWalksTurns_c3d/`（23；其中按文件名打 `block`/`dodge` 标签者计入防御覆盖，见 §6） | **23** |
| `locomotion` | `ACCAD/Male2Walking_c3d/`（23）+ `Male2Running_c3d/`（23） | **46** |
| `general` | `ACCAD/Male2General_c3d/`（13 − 1 cal = 12） | **12** |
| `defense_recovery` | CMU 补丁集（§4a，17 条已定点） | **17** |

**合计 172 条**。**刻意单一来源族（ACCAD Male2 + 少量 SSM/CMU）**：
同一受试者的武术全集，风格一致、覆盖攻防全要素，规模小到每条
都可逐条人工复核（延续 M1"不宣称无问题、显式记 issue"的做法）。

### §4a. defense_recovery 补丁（CMU，17 条定点）

格斗里"防-倒-起"闭环的补充类别（user 定，非重设计）：

| 子类 | clip_id | 来源描述 |
|---|---|---|
| `block`（格挡/拨挡，4） | `cmu_144_07` | Left_Blocks（M5/M6 审计已验） |
|  | `cmu_144_26` | Right_Blocks |
|  | `cmu_135_05` | 低格挡（M5 审计列） |
|  | `cmu_135_10` | 刀手格挡（M5 审计列） |
| `dodge`（闪避/下潜，5） | `cmu_77_09` | duck to avoid flying object |
|  | `cmu_139_09` | Ducking |
|  | `cmu_76_03` | avoid attacker |
|  | `cmu_127_29` | Run Duck Underneath |
|  | `cmu_128_04` | Run Duck Underneath |
| `getup`（起身，6） | `cmu_140_01` | Get Up Face Down（趴） |
|  | `cmu_140_04` | Get Up Laying on Side（侧） |
|  | `cmu_140_08` | Get Up Laying on Back（仰） |
|  | `cmu_139_18` | Get Up Laying on Back（仰 alt） |
|  | `cmu_111_06` | Get up from floor（审计已验） |
|  | `cmu_113_08` | Lay down and get up（审计已验） |
| `fall`（真摔，2） | `cmu_90_16` | fall on face |
|  | `cmu_90_18` | RugPullFall |

缺文件已替换：`140_03`→`140_04`、`90_17`→`90_18`、`134_01`→`128_04`。
全部走 GMR 管线 L1 准入（`has_ground_contact`/`recovery` 自动标）。

### extended 池建议（不进 core）

- 其余 ACCAD（Female1 全套做 subject-level 留出验证）
- `WEIZMANN/*`（2227 步态变体，locomotion 扩充）
- `KIT/*`（4232 日常动作）、`BioMotionLab_NTroje`（BMLmovi/rub 体育）
- `EyesJapan`、`TotalCapture`、`HDM05`（多样性背景数据）
- `CMU` 其余 subject（与旧 1264 条 ASF/AMC 库同源，可做新旧管线对照）

### 旧库（`cmu_*`，120Hz ASF/AMC 管线）的关系

- **保持有效**，v1 条目继续在 index 中；不自动进共享集。
- 新 pipeline 的 `amass_cmu_*`（同 subject/trial 的 SMPL 化版本）与其
  并存，供管线间对照；`cmu_13_17` 两条线的版本都做 L1/L2 交叉验证。
- 所有新方法的 baseline 一律以 core 集为准，旧库只做回归参照。

## 5. clip_id 命名（AMASS 来源）

`<subset_lower>_<相对目录扁平化>_<文件slug>`，规则：

- 目录层级压平，`Male2MartialArtsPunches_c3d` → `m2ma_punches`
  （`_c3d`/subject 编号目录按缩写表处理，见 build 工具映射）；
- 文件名去 `_poses`/`_stageii` 后缀，非字母数字 → `_`，小写；
- 例：`SSM/20160330_03333/walking_01_stageii.npz` → `ssm_03333_walking_01`；
  `ACCAD/Male2MartialArtsPunches_c3d/G1 - jab cross_poses.npz` →
  `accad_m2ma_punches_g1_jab_cross`；
- 镜像/切分派生沿用 DATASPEC §2 机制（`_m`/`_seg<k>`）。

## 6. 标签体系（`classification.labels`，多标签）

动作类别（必选其一以上）：
`strike_punch` `strike_kick` `strike_combo` `block` `stance_guard`
`footwork` `walk` `run` `turn` `crouch` `jump` `dance` `getup` `idle`
`dodge` `fall` `general_misc`

属性标签（自动派生，可叠加）：
`fast`（峰值关节角速度 > 阈值，建库时算）、`mirrored`（派生）、
`has_ground_contact`（含蹲/坐/地面动作）、`recovery`（倒地爬起段）。

`label_source`：`subset_official`（ACCAD/SSM 目录语义）|
`auto_feature` | `manual`。core 集标签以 subset_official 为准，
`manual` 复核后升级 confidence。

## 7. 划分规则

- 训练/验证按 **subject 级**划分（DATASPEC §6）：core 的验证集 =
  ACCAD Female1 全系（与 Male2 不同受试者）+ CMU 13 留 2 个 trial。
- 镜像派生不算独立片段，随父 clip 同侧分配。
- 各方法可在 core 内再选子集做 skill 集（如只挑 strike_*），
  但**报告必须注明使用的 clip_id 列表**。

## 8. core 集准入质量门

进 core 必须过 L1（kinematic_check），且：
- **一票否决**：`invalid_frames`、`frame_jump`、`root_orientation_jump`
- **容许但记录**：`foot_penetration`（形态学缺口）、`high_residual_*`
  （臂/踝残差）、`joint_velocity_spike`（快拳真实瞬态）
- 镜像/切分派生继承父 clip 的 issues 后再独立复检

## 9. 建库流水线

```
amass smplx npz ──GMR retarget──> qpos ──export──> PHC clip_dict
                                                  (split root/dof /
                                                   wxyz→xyzw /
                                                   pose_aa = [root_aa | dof·axis] /
                                                   FK keypoint_pos+smpl_joints /
                                                   foot_contacts /
                                                   ground-calib: 支撑脚z中位→
                                                   root/keypoint z 同步抬升
                                                   (meta.ground_calib_dz) /
                                                   meta+issues)
                          │
                          └──── kinematic check（pkl 版）──> L1 标记、
                              写 manifest + meta.issues
                                        │
                          bundle 打包──> combatbench_{core,extended}_vN.pkl
```

- 导出工具：`gmr/export_phc.py`（单条 GMR qpos → PHC clip_dict → bundle+manifest）；
  批量建库 `gmr/build_core.py`（本文 §4 的权威 clip 清单）；
  校验 `gmr/check_phc.py`；审片 `gmr/render_phc.py`
- 库根沿用 `/data1/dev/combatbench_mocap/`，v2 产物在 `phc/` 子目录
- RSI 友好性：GMR burn-in 已消引导瞬态；仍有问题的片段在 meta 标注
- 旧 `cmu_*` npz 库（v1 契约，wxyz/120Hz）只作回归参照，不进 v2 bundle
