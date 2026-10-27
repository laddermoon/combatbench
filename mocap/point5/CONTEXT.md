# CONTEXT.md — mocap/point5/

## 一句话

5 点（头/双手/双脚）**业务动作意图语言**的核心库：人/战术模型写稀疏关键帧任务，
本库把 anchor 解析成世界点并做**纯线性分段插值**，输出可采样的参考信号。

## 心智模型

- 业务语言 ≠ 机器人状态 ≠ 规划器。建筑设计 → 结构设计 → 施工 的"建筑设计"层。
- **Simulator-Unaware**：绝不 import mujoco / envs.* / baseline.* / centroidal_dynamics。
  提取、评分、PPO 插件全是目录外的 adapter。
- 实测与参考共用 `Point5State`，下游对称比较。

## 关键约定（改代码前必读）

- 词表固定 5 点；每个 keyframe **必须恰好覆盖 5 点**（没有 free/mask/先验）。
- 任务 = `keyframes[K] + durations[K-1]`，段时长+段内归一化 s；
  采样 `sample(seg,s)` / `sample_time(t)`，**出界报错不钳制**。
- v1 无速度：spec 无 velocity、插值无切向量、参考 `vel` 恒 None；
  静止段靠"相邻 kf 写相同 target"表达（没有 hold 字段）。
- anchor 全静态，build 时经 `Resolver` 一次解析；`initial` 仅供解析/诊断，
  kf0 的 target 才是参考起点。无动态锚点（不看对手）。
- 脚 z≈0 即触地（`is_planted`）；`tolerance`/`weight` 是给下游读的标注，
  合成器不消费。
- 校验只做结构层（点集/时长/容差），fail-loud。

## 入口

- `build_reference(task, resolver)` → `Point5Reference`
- `StaticStartResolver(initial, root_pos, root_rot)` —— adapter 构造它
- serde：`task.to_dict()/from_dict()`（JSON-safe）

## 测试

`PYTHONPATH=<combatbench> python3 -m pytest mocap/point5/`

## 下游怎么用（本目录不负责）

adapter 层（`envs/humanoid21` / `baseline`）把 `sim.data.xpos[keypoint_body_ids]`
喂成实测 `Point5State`，把本库参考变成 reward/obs。v2 候选：动态 anchor、
先验填充、评分模块、编辑运算——见 DESIGN.md §8。
