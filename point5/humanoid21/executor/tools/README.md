# tools/ — 无状态物理先验工具集

每个模块都是纯函数/无状态类。P5 约定：5 个关键点局部坐标 `(5,3)`
——头、左手、右手、左脚、右脚，重力对齐局部系（x 前、z 上）。

| 文件 | 功能 | 输入 → 输出 |
|---|---|---|
| `com_solver.py` | P5 → 推荐质心 | `p5 (5,3)` → `com (3,)` |
| `pose_solver.py` | P5 → 关节姿态+动作 | `p5 (5,3)` → `PoseResult{qpos(21,), action(21,), satisfied, energy}` |
| `inertia_solver.py` | P5 → 绕质心转动惯量 | `p5 (5,3)` → `I (3,3) kg·m²` |
| `capability_solver.py` | P5 → 质心能力预期（LIP） | `p5 (5,3)` → `Capability`；`margin(d)`/`a_max(d)`/`v_max(d)` |
| `landmark_viz.py` | 锚点序列 → MP4 走查动画 | `landmarks, out_mp4` → 视频文件 |
| `cop_solver.py` | P5 脚点 → CoP 可行域 | `p5 (5,3)` → `cop_ranges [poly|None ×2]`；`foot_cop_range(foot_xy)` |
| `obs_decoder.py` | obs → 局部 P5（FK 重建） | `obs (96,)` → `(p5 (5,3), R_local (3,3))` |
