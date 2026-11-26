# CombatBench Core — CoreClip 数据使用说明

本目录是 **Core 数据集（172 条）的 sim 原生执行格式**库。每条 `.npz` 是一个
`CoreClip`：20Hz 的完整物理状态序列，**任何一帧都可以直接 `set_core_state`
落进真实仿真环境**，用作回放、RSI 起始帧或 PD 跟踪目标。

上游 canonical 格式是 PHC bundle
（`mocap/humanoid21/phc/combatbench_core_v1.pkl`，30fps、世界系位置量、
无速度）；本目录是它的单向派生产物（`meta.lineage` 记录转换链）。

## 目录结构

```
data/core/
├── <clip_id>.npz     # 172 条 CoreClip
└── videos/           # 真 env 回放视频（battle 模型 + 跟拍相机）
```

`clip_id` 命名约定：`<来源>_<动作集>_<序号>_<描述>`，如
`accad_m2ma_punches_e1_jab`、`cmu_140_01`（起身）、`ssm_03333_walking_01`。

## npz 内容

| 字段 | shape | 说明 |
|---|---|---|
| `root_pos` | (T,3) | 根位置（世界系） |
| `root_rot` | (T,4) | 根四元数 wxyz |
| `root_vel_local` | (T,3) | 根线速度（机体系） |
| `root_angular_vel_local` | (T,3) | 根角速度（机体系） |
| `joint_pos_norm` | (T,21) | `(q-mid)/half`，`CONTROLLED_JOINTS` 序 |
| `joint_vel_norm` | (T,21) | `qd/half` |
| `ref_action` | (T,21) | PD 目标，`meta.ref_action_semantics="pd_target_norm"` |
| `keypoint_pos` | (T,7,3) | 7 个语义关键点的世界系位置，顺序 = `meta.keypoint_names`（torso/head/pelvis/foot_r/l/hand_r/l） |
| `meta_json` | str | JSON：clip_id/labels/split/issues/质量/`rsi_bad_frames`/谱系 |

帧率 20Hz（`control_dt=0.05`）。速度在重采样网格上由姿态推导，
`v·dt = Δq` 在存储帧上严格成立（误差 ~1e-6）——每帧都是物理自洽状态。

## 用法一：真 env 回放

```bash
# 单条 / 整目录渲染视频（真 battle 模型，对手机器人幽灵化）
python -m mocap.humanoid21.clip play-env --clip data/core/<clip>.npz --out /tmp/x.mp4
python -m mocap.humanoid21.clip play-env --dir data/core --outdir data/core/videos
```

代码中逐帧落位（运动学回放 = 每帧写 qpos/qvel）：

```python
from mocap.humanoid21.clip import CoreClip
from envs.humanoid21.simulator import Humanoid21Simulator

clip = CoreClip.load('data/core/accad_m2ma_punches_e1_jab.npz')
sim = ...                                # 加载 battle_circular_v2.xml
for k in range(clip.T):
    sim.set_core_state({'robot_a': clip.state(k)})
    # sim.physical_step() 若想叠加物理演化
```

## 用法二：RSI 起始帧采样

```python
import numpy as np, json
from mocap.humanoid21.clip import CoreClip

clip = CoreClip.load(path)
bad = set(clip.meta.get('rsi_bad_frames', []))      # >0.6rad 跳变帧 + 首尾各2
valid = [k for k in range(clip.T) if k not in bad]
k = np.random.choice(valid)
sim.set_core_state({'robot_a': clip.state(k)})       # 完整状态（含速度）一次落位
# → 从这个姿态起 policy rollout
```

**一定要跳过 `rsi_bad_frames`**：那些帧的姿态来自单帧跳变，速度场被差分
放大到非物理量级（峰值 ~16-20 rad/s），拿去做 seed 会产生假冲击。

## 用法三：关键点（r_kp / 摔倒判定）

`keypoint_pos` 是**参考侧**轨迹——世界系绝对位置，由状态 FK 生成，
与 env 侧 `get_derived_state().body_xpos` 同模型同口径：

```python
clip = CoreClip.load(path)
kp = clip.keypoint_pos[k]          # (7,3)，顺序见 meta['keypoint_names']
ref_head = clip.keypoint(k)['head']  # 单帧 dict 访问

# DeepMimic r_kp：参考关键点 vs 机器人实况关键点（root-relative 误差）
obs = sim.get_derived_state()['robot_a']['body_xpos']
live = obs[static['keypoint_body_names']['head']]
# 奖励侧自取相对量（减根/转机体系），数据存绝对位置不预设用法
```

摔倒判定同理：参考侧 `keypoint_pos` 的 head/pelvis 高度关系 +
实况 `body_xpos` 的 head_z < pelvis_z 判定。

## 用法四：质量过滤与标签

```python
m = clip.meta
m['phc_quality_level']   # 'L1' 干净 / 'L1_veto' 有未决病灶
m['phc_issues']          # ['frame_jump', 'root_orientation_jump', ...]
m['labels']              # ['strike_kick'] / ['walk'] / ['getup'] ...
m['split']               # 'train'（171 条）/ 'val'（1 条）
```

当前分布：81 L1 / 91 L1_veto。veto 不等于不可用——
`frame_jump` 多为高速动作撞阈值的疑似误报，`foot_penetration` 是
踝映射残差，`head_below_pelvis` 在 getup/fall 中属预期。训练侧建议
L1 全用、L1_veto 按 issue 类型白名单选用。

## 质量门 issue 语义速查

| issue | 含义 | 建议 |
|---|---|---|
| `frame_jump` | 单帧 >20° 关节跳变 | 采样用 rsi_bad_frames 屏蔽 |
| `foot_penetration` | 摆动脚 <z=-2cm | 踝映射残差，物理回放可吸收 |
| `tendon_violations` | 腘绳肌约束越界 | **已清零**（IK 硬约束） |
| `root_orientation_jump` | 根姿态跳变 | 高速旋转动作，慎用对应帧 |
| `head_below_pelvis` | 头低于髋 | getup/fall 预期行为 |

## 重新生成

```bash
# 从 PHC bundle 全量重转（幂等覆盖本目录）
python -m mocap.humanoid21.clip phc2core --bundle core
```

PHC → CoreClip 转换定义见 `clip/convert.py: phc_to_coreclip`，
格式契约见 `clip/CORECLIP_SPEC.md`，上游数据集定义见 `DATASET.md`。

## 注意

- `MUJOCO_GL`：本机 EGL 坏了，`clip` 包已在导入前 pin `osmesa`；
  换有 EGL 的机器时环境变量显式设置可覆盖
- `clip play`（旧 broadcast 相机）在本机不可用，请用 `play-env`
- 旧 `clip/data/heavy_*`、`reclip_bc/` 产物属于 refclip 历史管线，
  与本库无关
