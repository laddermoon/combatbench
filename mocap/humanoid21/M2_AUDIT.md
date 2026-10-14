# M2 审计：物理跟踪验证（进行中）

> 状态：**W1–W3 进行中**。open-loop 基线完成（全部失败）；闭环跟踪策略
> 已验证管线可行 —— `cmu_07_01`（走）训练收敛；`cmu_13_17`（拳）训练中。
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

## W3 闭环跟踪策略（DeepMimic 式）

新组件（`mocap/humanoid21/tracking/`）：

- `ref_data.RefClip`：npz → 20Hz 控制帧视图（每 6 帧取 1），含归一化关节
  目标、根 pos/quat/vel、keypoints。
- `simulator.MotionTrackingSimulator`：
  - reset 时 RSI（随机相位，seed 驱动；`options.start_frame` 可固定起点），
    一次性写入根 pos/quat/线速度/机体系角速度 + 关节 pos/vel；
  - `set_action` 递增相位；obs 扩展 46 维（phase + ref q_norm +
    ref qv_norm + ref 根速度机体系）→ 142 维，仅被跟踪机器人获得
    （robot_b 保持 96 维给 standing policy）；
  - `get_derived_state()['motion_track']` 暴露当前参考帧给奖励/终止。
- `plugins.MotionTrackGuardPlugin`：torso_z<0.55 或头低于骨盆 →
  `track_fall`；相位走完 → `track_clip_end`。
- `reward.MotionTrackRewarder`：DeepMimic 加权项
  0.45·pose + 0.10·vel + 0.35·kp(torso/head/双手/双足) + 0.10·root高度。
- 蓝图 `baseline/humanoid21/blueprints/track_ref_env.yaml`；
  实验 `baseline/experiments_ppo/exp_track_walk.py`
  （`--set clip_id=...` 切片段；robot_b = StandingCombatPolicy 场景机）。
- 评估 `track_eval.py`：rsi='first' 从帧 0 确定性整段回放 + 视频。

### 结果

**cmu_07_01（走, 2.6s, 53 控制帧）** — 训练 run `train_track_walk_ppo_20261014_132001`：

| update | random-RSI eval (16 eps) | 备注 |
|---|---|---|
| ~500 | survive≈1.0, complete≈0.9, fall≈0.06, kp_err≈0.30m | 初步稳定 |
| ~1150 | complete=1.000, fall=0, kp_err≈0.13–0.15m, q_rmse≈11° | 收敛 |

- 整段确定性回放（帧0起）：`track_clip_end` 完成，q_rmse≈13°，
  kp_err 0.43m（高于随机相位评估的 0.13 —— 误差集中在首尾过渡与
  系统性 ~7cm 躯干下沉）；u830 曾结尾下蹲至 torso_z 0.60，
  u1156 已全程走完。
- 广播视频抽帧目测：迈步、换脚、手臂摆动均正确。
- **L2 候选**：待视觉复核后在 index 标记。

**cmu_13_17（拳, 40s, 807 控制帧）** — run `track_box_13_17` 训练中；
u310 时 fall_rate 仍 ~0.9（拳击动态强 + 片段长，属预期慢收敛）。

## L2 判据（定稿）

一个 clip 记为 L2 需要：
1. 存在训练收敛的跟踪策略：random-RSI eval `complete_rate ≥ 0.9`、
   `fall_rate ≤ 0.1`；
2. `rsi='first'` 整段确定性回放达到 `track_clip_end` 且不摔倒；
3. 逐帧指标：q_rmse ≤ ~15°、kp_err_mean ≤ ~0.45m（形态学差距内）；
4. 生成 `track_eval` 视频供人工复核确认运动语义。

已知注定失败类型：13_01–03 起身类（源坐凳子，环境无凳子）——保留数据，
L2 评估单独标注 `requires_prop`。

## 已复用教训（M1 迁移）

- 视觉验收标准必须源自同一 FK 约定生成的参考；`track_replay` 直接消费
  已验证 npz，不重新解释 mocap 数据。
- 区分验证层级：`tracking_openloop` 字段显式标注方法串，不混淆 L1/L2。
