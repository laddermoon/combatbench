# 参考动作视频渲染指南

打击技能模仿的 mocap clip 的各层可视化方法。资产库根：`/data1/dev/combatbench_mocap/`。

## 数据层级

```
clips/<clip_id>_src.npz   人类源骨架：31 关键点 3D 轨迹（120Hz，mocap 本体）
clips/<clip_id>.npz       重定向后：humanoid21 关节角 + root + keypoints
```

当前打击技能的 clip 绑定（`resolve_clip`）：

| 技能 | clip_id |
|---|---|
| jab_l / jab_r | `hdm05_3146_88_2895_127` / `hdm05_3146_88_2895_127_m` |
| heavy_l / heavy_r | `cmu_144_20_m_seg01` / `cmu_144_20_seg01` |

镜像约定：`*_m` 或 `*_m_<seg>` 后缀 = 父 clip 的左右镜像；`*_m_src.npz`
存的是**镜像前**源骨架，渲染时需自行镜像（`compare_render` 已处理）。

## 渲染工具

### 1. 人体骨架 vs 机器人对照（首选）

```bash
cd /data1/mono/things/combatbench
MUJOCO_GL=egl PYTHONPATH=. python3 -m mocap.humanoid21.compare_render \
    --clips cmu_144_20_seg01 cmu_144_20_m_seg01 hdm05_3146_88_2895_127 \
    --outdir /data1/dev/combatbench_mocap/compare_strikes --every 3 --fps 40
```

输出：三联画面——源骨架矢状面（面向对齐，拳头朝右伸）+ 顶视 +
机器人 `mj_forward` 运动学回放（**无物理**，纯姿态回放）。
需要 `MUJOCO_GL=egl`（headless 渲染）。

### 2. 纯人体骨架动画

```bash
python3 /data1/dev/s3_eval_validation/render_src_clip.py \
    --clips cmu_144_20_seg01 hdm05_3146_88_2895_127 \
    --outdir /data1/dev/combatbench_mocap/preview_src --stride 3 --fps 40
```

直接从 `_src.npz` 渲骨架（不依赖原始 AMC，hdm05 clip 也适用）。
若只想看源动作形态用这个最轻。

### 3. 原始 AMC 骨架预览（仅 CMU clip）

```bash
PYTHONPATH=. python3 -m mocap.humanoid21.preview \
    --clips 144_20 --outdir /data1/dev/combatbench_mocap/preview
```

需要 `/data1/dev/combatbench_mocap/raw/<subject>/<clip>.amc` 存在，
渲的是**整条原始 clip**（非 seg 段），三联视图同 compare_render。

### 4. 物理回放（PD 开环灌入真实仿真）

```bash
MUJOCO_GL=egl PYTHONPATH=. python3 -m mocap.humanoid21.track_replay \
    --clips cmu_144_20_seg01 hdm05_3146_88_2895_127 \
    --outdir /data1/dev/combatbench_mocap/track --render
```

物理仿真 + PD 跟踪参考轨迹——验证重定向动作在动力学约束下
实际能跟到什么程度，同时输出 JSON 指标（saturation、误差等）。

### 5. 学到的策略 rollout 视频

`render_skill_videos.py`（本目录）——对指定 checkpoint 渲各技能
动作视频，输出到 `videos/<timestamp>/`。

## 当前已渲视频

| 内容 | 路径 |
|---|---|
| 骨架+机器人对照（4 打击源 clip） | `/data1/dev/combatbench_mocap/compare_strikes/` |
| 纯人体骨架（源段） | `/data1/dev/combatbench_mocap/preview_src/` |
| 物理回放 | `/data1/dev/combatbench_mocap/track/*_track.mp4` |
| u9860 策略（P2 收敛） | `videos/20261019_052455/` |
| u8000 策略（定形态版） | `videos/20261018_141445/` |

## 源动作形态备注（肘角诊断结论）

肘部三点夹角（上臂-前臂，180°=完全伸直）在源骨架最大伸展帧：

- **jab**（`hdm05_..._127`）：源 179.6° —— 直拳，机器人同样打直
- **heavy**（`cmu_144_20_seg01`）：源 157.4° —— **曲肘重拳**（摆拳
  式发力），ref/机器人忠实复现。若需直臂重拳，要换 clip 或改
  landing 目标，不是训练缺陷。
