# CONTEXT.md — point5/

## 一句话

5 点（头/双手/双脚）**业务动作意图语言**的核心库（v2：躯干朝向局部系）：
人/战术模型写稀疏关键帧 clip，本库做分段线性插值 + 自适应采样，
输出可采样的参考信号。

## 心智模型

- 业务语言 ≠ 机器人状态 ≠ 规划器。建筑设计 → 结构设计 → 施工 的"建筑设计"层。
- **Simulator-Unaware**：本包（`humanoid21/` 除外）禁止 import mujoco /
  envs.* / baseline.*。提取、评分、PPO 插件全是目录外的 adapter；
  `humanoid21/` 是唯一允许碰 sim 代码的子包。
- v2 局部系（DESIGN.md §3）：原点 = 双踝 XY 中点，X = 躯干前向，Z = 上
  （地面 z=0）。P5 状态只有 `local_pos[5×3]`，无绝对位置/朝向/yaw 字段。
- Clip = keyframes[]；第 k 帧带 `transfer{dx,dy,dyaw,dur}` 记录
  `F_k = F_{k-1} ∘ T`（transfer 记在到达帧上，第一帧没有）。
  世界放置 `f1=(origin_xy, yaw_deg)` 是消费端的选择，不是姿态属性。
- 实测与参考共用 `Point5State`，下游对称比较。

## 关键约定（改代码前必读）

- 词表固定 5 点；每帧恰好覆盖 5 点，无 free/mask/先验。
- 插值纯线性；`sample_clip` 按世界位移自适应细分（每帧 ≤ max_step），
  出界/非法 fail-loud（`TaskValidationError`/`SampleOutOfRange`）。
- 脚点 = 踝关节中心（`LANDMARKS` +0.06z），站姿实测 z≈0.087 而非 0；
  `is_planted` 语义 = "接近地面"标注。
- `transfer.dur` 是 schema 必需字段，但 P5 不承诺执行时序语义
  （stand_punch 直接填 0.1 占位）。
- 每帧 `meta["specs"]` 存 `{w,tol}` 评分注解给下游读，P5 不消费。
- 校验只做结构层（形状/有限性/transfer 归属），不做可达性判断——
  可达性归 humanoid21/mechanics。

## 入口

- `P5Clip.from_dict()/to_dict()` —— JSON-safe serde（`type=p5clip.v2`）
- `sample_clip(clip, f1, max_step)` → `List[FrameSample]`，`.world_pos()` 取世界系
- `keyframe_world(clip, f1)` / `keyframe_frames(clip, f1)` —— 关键帧世界位姿
- `Point5State(pos, vel=None, metadata)` —— 实测/参考共用
- `humanoid21.plan_punch(initial_p5, target, hand, cfg)` → `PunchPlan`
  （`task` 为 `P5Clip` 或 None：out_of_sector/out_of_reach）
- `humanoid21.plan_stand_punch(target, hand, cfg)` —— 站姿前手拳 4 帧
  （coil→swing→land→strike，最近迭代热点）
- `viz.py`：`PYTHONPATH=. python3 -m point5.viz --clip c.json --out c.mp4 [--html]`

## 测试

`cd things/combatbench && PYTHONPATH=. python3 -m pytest point5/`（57 passed）

## 下游怎么用（本目录不负责）

- `humanoid21/adapter.point5_from_view(view, robot_id)`：唯一数据切面——
  `get_derived_state()['robot_x']` 的 view dict → 实测 `Point5State`（局部系）。
  为 observer 插件内嵌设计，不依赖 sim 对象。
- 评分/观测/PPO 插件等 harness 仍在目录外（baseline/、envs/）。
