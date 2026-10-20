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

**段内容**：每个 clip 是完整打击段——发力准备 → 出拳伸展 → **收回**，
收回段数据/视频都在。训练是 extension-only：单调进度指针只评分到
落地帧 `clip_land_frame`（手部前向最大伸展帧，jab k=11/22 ctrl 帧、
heavy k=6/13 ctrl 帧），落地帧之后的收回段存在于数据与视频中但
不参与奖励——这意味着 policy 收回动作是自由发挥的，不作为评价
对象。

## 文件

```
clips/<clip_id>.npz       重定向后的 21-DOF 机器人参考（训练实际使用）：
                          joint_angles(T,21), keypoints(T,15,3),
                          root_pos/quat, foot_contacts 等
clips/<clip_id>_src.npz   重定向前的源人体骨架（31 关键点，ASF 命名），
                          用于渲染人形视频与人工核对动作内容
videos/<skill>.mp4        _src.npz 的三视图骨架视频（矢状/冠状/俯视，
                          双手红点高亮，30fps）
```

## 使用说明

训练中由 `strike_track_observer.py` 经 `_CLIP_MAP` 按 `(eff_side, is_near)`
解析到 clip_id，`RefClip(clip_id, clips_dir)` 加载 `.npz`；clip 第 0 帧
是发力准备姿态（launch gate 判据帧），`clip_land_frame` 给出落地帧
（手部前向最大伸展帧）。权威 clip 目录是
`/data1/dev/combatbench_mocap/clips`，本目录是四个打击段的副本备份。

视频复现命令：

```bash
python3 render_src_clip.py --clips <clip_id> --outdir <out> --stride 4 --fps 30
# 渲染脚本: /data1/dev/s3_eval_validation/render_src_clip.py
```
