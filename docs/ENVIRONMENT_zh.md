# 仿真环境

> 类型：契约
> English: ENVIRONMENT.md

该仿真环境支持具有 21 自由度 (DoF) 控制的人形机器人。当前场地模型为
`envs/humanoid21/battle_circular_v2.xml`（`Laddermoon_Arena_Circular`）。

仿真环境主要包括：

1. **机器人所处环境：**
   一个完全封闭的圆形竞技场，由地面、24 块斜面平板拼接的圆形围墙
   （`wall_00`–`wall_23`）和天花板围合而成。
   - **场地高度：** 6.10 米。
   - **场地半径：** 约 3.44 米（直径约 6.88 米，规模上相当于 AIBA
     业余拳击 6.10 米 / 20 英尺见方的正式拳台）。
   - **材质贴图：** 墙面交替使用六张贴图
     （`envs/humanoid21/textures/wall_0.png`–`wall_5.png`），地面用
     `floor_circular.png`，天花板用 `ceiling.png`，均在同目录下。
   - **物理选项：** `timestep = 0.002 s`（500 Hz）、`condim = 3`、
     `impratio = 10`。

2. **两个机器人：**
   两个 21 自由度人形机器人，源自官方
   [MuJoCo humanoid model](https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml)，
   直接内嵌在场地 XML 中定义，带有颜色与几何定制。
   - `robot_a` 为红色，`robot_b` 为蓝色。
   - 两机器人隔场地中心线面对面站立，间距为 `initial_distance`
     （默认 2.0 米）。
   - 初始状态均为完全竖直站立。

3. **灯光设置：**
   使用 MuJoCo headlight（`ambient = diffuse = 0.4`）加渐变天空盒，
   没有独立的离散光源。

4. **固定摄像机（共 9 台）：**
   - 四台对角相机位于 `(±2.22, ±2.22, 4.0)`（`cam_diag_0`–`cam_diag_3`），
     朝向场地中心。
   - 四台正方位相机位于 `(±3.14, 0, 3.0)` / `(0, ±3.14, 3.0)`
     （`cam_east`/`south`/`west`/`north`），朝向场地中心。
   - 一台顶视相机（`ceilingcamera`）位于 `(0, 0, 6.0)`，垂直向下俯拍。
   - 此外每个机器人自带 `back`/`side` 追踪相机（`mode="trackcom"`）
     和 `egocentric` 第一人称相机。
   - `get_broadcastview_image()` 的默认广播视角是自动取景的动态相机，
     不属于以上固定相机。
