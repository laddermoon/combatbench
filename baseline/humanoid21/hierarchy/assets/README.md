# 打击参考动作资产

技能训练中四种打击技能的参考 clip（mocap 数据）与源骨架渲染视频。

## 技能 ↔ clip 对应

| 技能 | clip_id | 数据库来源 |
|---|---|---|
| jab_l 左刺拳 | `hdm05_3146_88_2895_127` | HDM05（subject 3146） |
| jab_r 右刺拳 | `hdm05_3146_88_2895_127_m` | 同上，`_m` = 镜像版 |
| heavy_l 左重拳 | `cmu_144_20_m_seg01` | CMU mocap 144-20 的打击段切割，`m_` = 镜像版 |
| heavy_r 右重拳 | `cmu_144_20_seg01` | CMU mocap 144-20 的打击段切割 |

`_m` 镜像变体由同一数据经左右对称变换生成，用于对侧技能。
**原始侧**：jab 源数据是左刺拳（jab_r 为镜像）；heavy 源数据是
右弓步冲拳（heavy_l 为镜像）。

**镜像实现细节**：`_m` 的 `_src.npz` 是"几何镜像 + 180° 偏航，
不换左右关节标号"——镜像后 `lwrist` 标号仍在原左手轨迹上
（=镜像体的解剖右手）。重定向 `.npz` 的标号正确换边（`cmu_m`
的 `hand_left` 承载原右手轨迹）。渲染 facing 箭头时 `_m` clip
需交换肩线两端（`render_src_clip.py` 已处理），否则箭头指向背后。

## 三种切法（full / extend / retract）

原始 clip 是完整打击段（发力准备→出拳伸展→收回），按落地帧
`clip_land_frame`（手部前向最大伸展帧，npz 源帧下 cmu=39、hdm05=63）
切成三档，落地帧同时归入 extend 末帧与 retract 首帧：

| 目录 | 内容 | 帧数（cmu / hdm05） |
|---|---|---|
| `full/` | 完整段（0 → T-1） | 78 / 127 |
| `extend/` | 只打出（0 → land，含） | 40 / 64 |
| `retract/` | 只收回（land → T-1，含） | 39 / 64 |

每档内文件名不变（drop-in 可替换，如 `RefClip(cid, clips/<variant>/)`）。

## 文件

```
clips/<variant>/<clip_id>.npz       重定向后的 21-DOF 机器人参考（训练
                                    实际使用）：joint_angles(T,21),
                                    keypoints(T,15,3), root_pos/quat,
                                    foot_contacts 等
clips/<variant>/<clip_id>_src.npz   重定向前的源人体骨架（31 关键点，
                                    ASF 命名），用于渲染人形视频与人工
                                    核对动作内容
videos/<variant>/<skill>.mp4        _src.npz 的三视图骨架视频（矢状/
                                    冠状/俯视，双手红点高亮，30fps）
```

## 使用说明

训练中由 `strike_track_observer.py` 经 `_CLIP_MAP` 按 `(eff_side, is_near)`
解析到 clip_id，`RefClip(clip_id, clips_dir)` 加载 `.npz`；clip 第 0 帧
是发力准备姿态（launch gate 判据帧），`clip_land_frame` 给出落地帧。
权威 clip 目录是 `/data1/dev/combatbench_mocap/clips`，本目录是四个
打击段的副本备份。

当前训练是 extension-only：进度指针只评到落地帧，等价于 `extend/`
档的语义；`retract/` 档为将来"打完收回"评价预留数据。

视频复现命令：

```bash
python3 render_src_clip.py --clips <clip_id> --outdir <out> --stride 4 --fps 30
# 渲染脚本: /data1/dev/s3_eval_validation/render_src_clip.py
# 注意其 CLIPS 常量指向要渲染的 variant 目录
```
