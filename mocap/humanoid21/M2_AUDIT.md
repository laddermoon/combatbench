# M2 审计：物理跟踪验证（进行中）

> 状态：**W1–W2 完成（open-loop 基线）**；W3（跟踪策略）待实施
> 模型：`envs/humanoid21/battle_circular_v2.xml`；控制器：固定 KP/KD 归一化位置 PD（CONTROLSPEC）
> 数据：`/data1/dev/combatbench_mocap/track/`（每 clip 一个 `*_track.npz` + `*_track.json` + `summary_openloop.json`）

## M2 目标（对齐 README §9）

验证 retargeted 参考运动在**真实物理 + 真实控制链**下能否被执行：
`mj_forward` 摆位回放（compare_render）只验证运动学，不算物理成功。
M2 要求 state-non-overwriting：t=0 初始化一次，之后只靠控制输入。

## W1 跟踪回放器 `track_replay.py`

- 从 `clips/cmu_*.npz` 读参考（120Hz）：root pos/quat/vel、21 关节角/速度、keypoints。
- 水平平移轨迹使 xy-bbox 居中于场地（偏移量写入 metrics 溯源）。
- t=0 `set_core_state` 一次性写入根 pos/quat/局部速度 + 归一化关节 pos/vel。
- 之后每 25 个物理步（20Hz）用参考关节角 → `action=(q−mid)/half` 喂 `set_action`；
  **全程不再写状态**。
- 记录每控制帧：actual/ref 关节角、根 pos/quat、6 个 keypoint body（torso/head/foot/hand）、
  ctrl 饱和度；摔倒判据 torso_z<0.55 或 head_z<0.45（摔倒即停，不覆写）。
- robot_b 在 `initial_distance=6.0` 处 standing，未与 A 接触。

## W2 open-loop 基线结果（17/17 全部失败）

| 指标 | 范围 |
|---|---|
| 存活时间（摔倒前） | **0.45–2.0 s**（07_01 走 1.5s；13_17 拳 0.8s；最长 14_01 2.0s） |
| 关节跟踪 RMSE（存活窗口内） | 4.3–11.7° |
| 根位置漂移（max） | 0.8–2.2 m（参考继续前进，机器人原地滞后→侧倒） |

对照实验：robot_a 直接 hold `standing` 初始姿态 action，**1.5s 内同样倒地**
（torso_z 1.28→0.24）。

### 结论

1. **失败主因是缺闭环平衡，不是关节刚度或参考质量问题**：
   关节 RMSE 在倒地前仅 ~10°，PD 在忠实执行目标；但 humanoid 站立本就是不稳定的
   倒立摆，固定 KP/KD 位置 PD 无根级反馈，站都站不住 → 参考 motion 的腿部支撑
   序列在物理上无法复现。
2. PD 控制律为 `KP*(target−q) − KD*qvel`，**无目标速度前馈项**，
   对运动目标存在 ~`KD/KP`·qvel 的恒定滞后（13_17 存活期 rot_err 达 60–100°，
   部分来自滞后放大失衡）。
3. ctrl 饱和比例 ~10–17%，拳击片段最高（0.171）但不是主要失稳源。
4. open-loop 回放达标线为物理不可行 → **L2 达标必须走闭环跟踪策略**
   （DeepMimic 式：模仿奖励 + 早期终止 + 相位变量），这也正是 motion-prior
   的目标形态，历史 standing/fight 策略也证明了这点。

## 下一步（W3 计划）

- 新建跟踪环境：`motion_tracking` 插件 + observer，DeepMimic 式奖励
  （关节角/关节速度/keypoint 位置/根高度与朝向），fell→early termination；
  相位 φ 驱动参考帧采样（参考下采样到 20Hz 控制率，npz 120Hz → 每 6 帧取 1）。
- 先在 `cmu_07_01`（2.6s 走）PPO smoke；再拳击子集。
- L2 判据（草案，待 W3 后定稿）：训练完成的跟踪策略 rollouts 中
  episode 存活 ≥ 90% clip 时长、平均 keypoint 误差 ≤ ~0.2m、关节 RMSE ≤ ~15°，
  并生成对照视频供人工复核。
- 已知注定失败的片段类型：13_01–03 起身类（源坐在凳子上，环境中无凳子接触，
  开环与闭环前 ~1s 均缺支撑面）——保留数据但在 L2 评估时单独标注。

## 已复用教训（M1 迁移）

- 视觉验收标准必须源自同一 FK 约定生成的参考；`track_replay` 直接消费
  已验证 npz，不重新解释 mocap 数据。
- 区分验证层级：`tracking_openloop` 字段显式标注方法串，不混淆 L1/L2。
