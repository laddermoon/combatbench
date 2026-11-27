# Humanoid28 Core 数据集（CoreClip 格式）

MimicKit DeepMimic 系 humanoid（28 hinge DOF，pelvis 根）的 172 条
CoreClip。与 humanoid21 Core 同源同清单（`gmr/build_core.py`
`clip_list()`），经 GMR IK 重定向到 h28 形态。

**关键差异（vs humanoid21）：`joint_pos`/`joint_vel` 是原始
弧度/rad·s⁻¹，不是归一化值**——MimicKit 系机器人不归一化。

## 文件

- `*.npz` — 每条 clip 一个文件，20Hz，`set_core_state` 可直接喂
- `videos/` — 真 env（battle_humanoid28.xml）运动学回放视频
- `../phc/combatbench_core_v1.pkl` — canonical PHC bundle（30fps
  运动学位形，转换源）

## npz 字段

| 字段 | 形状 | 说明 |
|---|---|---|
| `root_pos` | (T,3) | pelvis 世界系位置 |
| `root_rot` | (T,4) | wxyz 四元数 |
| `root_vel_local` / `root_angular_vel_local` | (T,3) | 机体系速度 |
| `joint_pos` / `joint_vel` | (T,28) | **弧度 / rad/s**，dof 序见 meta |
| `ref_action` | (T,28) | 弧度 PD 目标（= joint_pos） |
| `keypoint_pos` | (T,7,3) | 语义关键点世界系位置 |
| `meta_json` | str | clip_id/labels/split/issues/谱系 |

`meta.keypoint_names` = `pelvis/torso/head/right_foot/left_foot/
right_hand/left_hand`（side-first 命名，与 h21 不同序）。

## 用法

```python
from mocap.humanoid28.clip import CoreClip
clip = CoreClip.load('accad_m2ma_punches_e1_jab_left.npz')
sim.set_core_state({'robot_a': clip.state(k)})   # 任一帧 -> RSI seed
clip.keypoint(k)   # {name: (3,)} 语义关键点
```

**RSI**：每帧都是物理自洽状态（速度由 20Hz 网格差分得到）。
采样前屏蔽 `meta.rsi_bad_frames`（>0.6rad 单帧跳变 + 首尾各2帧）。

## 重新生成

```bash
mocap/gmr/.venv/bin/python mocap/humanoid28/gmr/build_core.py   # PHC
python -m mocap.humanoid28.clip phc2core --bundle core          # npz
python -m mocap.humanoid28.clip play-env --dir data/core \
       --outdir data/core/videos                                # 视频
```

本机 EGL 坏——`clip` 包内已 pin `MUJOCO_GL=osmesa`，不要在导入前覆盖。

## 注意

- 无腱约束：h28 形态没有 tendon（h21 的 hamstring 限位不存在）
- h28 action bounds 允许 PD 目标超出机械限位——`ref_action` 合法域
  是 action bounds，不是 jnt_range
- 地面校准以**足底面**为准（foot body 系原点下 0.05m）
