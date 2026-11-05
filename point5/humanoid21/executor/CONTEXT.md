# CONTEXT.md — point5/humanoid21/executor/

## 一句话

P5 指令的**无状态伺服后端**：`Executor.gen_action(Context) -> Action`。
Executor 是策略后端——只能看到 policy 的 96 维 observation，拿不到
sim 内部结构；只持有自己的内部仿真器（wsim），不维护任何执行进度。
调用方（`e2e_policy.py`）负责 clip 进度推进 + 每步把剩余 clip 首帧
重锚定到当前观测。

## 流水线（已实现）

```
gen_action(ctx)
  0  _verify_anchor    clip kf0.pos 必须==obs 解出的当前 P5（atol 1e-6）
  1  算法1 CentroidalPlanner.plan(obs, clip)
       ObsDecoder(obs→局部P5) → extract_landmark(起点)
       → plan_footsteps(支撑规划：捕获点矫正/回位/leapfrog)
       → gen_landmarks(帧驱动锚点序列)
  2  算法2 ActionComputer.compute(ctx, landmarks, wsim)
       → (21,) 归一化 PD 目标，只执行 u0
```

## 算法2 形态（当前）

- 注入 `obs_to_core_state` 重建 x̂0 到 wsim；**模型与真实 sim 零发散**
- 候选 = **[k(5), d(5)]** 5 关节组跟踪增益+残差偏移；rollout 内
  **闭环**：`u = clip(a_ref + k(a_ref−q) + d)`（反馈在模型内）
- 先验工具集在 `tools/`（各工具说明见其 README.md）：com/pose/
  inertia/capability/cop/start solver + landmark_viz
- `a_ref[i]` = 锚点 i 的 clip 帧 P5 → `tools.pose_solver.solve().action`
  （按 pos 内容缓存）
- rollout 内 ref 按锚点序列走查：**reach-only（reach_eps=0.06）**——
  时间兜底变体让 ref 追着做不到的锚点走，视频全倒，已废弃
- score：per-step 平衡项(tilt²·6/com_z 下限·12/vz/捕获点出域) +
  schedule-aware 末端误差 + tgt 推进奖励；tilt abort 机械存在但
  阈值 3.0rad 实际禁用
- CEM：pop32×iter4，注入 hold/mean/纯ref 保底，种子=ctx.meta.seed

## 调用方职责（e2e_policy.py）

- 进度门控翻页：P5 err<0.30 且过半 dur → 翻页；兜底 max(3×dur,1.2s)
- clip 耗尽 → 收势站姿 clip（hold target = stand_p5，拳击收回站架）
- meta 传 last_action / seed

## 契约（contracts.py）

- `Context` = {observation, clip, meta}；clip 是切好的**剩余指令**
- `LandMark` = {com(3,), feet{force_ratio,cop,foot_pos}, vel|None,
  frame}；`Footstep` = {foot,target,kind,for_frame|before_frame}
- `Action` = (21,) PD 目标；`PlanError`/`AnchorError` 为失败语义

## 已证结论（勿重踩）

- 姿态开环 PD 必倒；**反馈式 rollout 候选才能抗扰**（开环动作带里
  没有"主动扶正"方向）
- CEM 必须给 null/mean 候选打分，否则漂进比 hold 更烂的区域
- 平衡项要 per-step 累加；放 `min()` 里会被倒塌瞬间刷分
- com_z 目标必须配单向下限：下蹲在 tilt 项里免费，会棘轮式坐化
- 时间烧 clip（dur 预算）会让意图先于身体过期——必须进度门控
- obs[45] 是 root_z(≈1.28) 不是 com_z(≈0.87)，下限要打在 com 上
- 迈步轨迹不是静态姿态能表达的——v1 迈步只能到 ~+0.10/0.35
- **系统完全确定**（CEM 定种子+确定性 sim）：同代码=同轨迹=同视频，
  v12 MD5==v4。任何评分改动都会重排那条唯一存活的轨迹——"改进"
  必须 A/B 到视频长度（40 步 diag 预测不了 120 步存活）

## 验证抓手

- `diag_e2e.py [stand|punch] [steps]`：手动 sim loop + com_z/tilt/
  fl_x/reach 日志，FALLEN 自动停
- `test_e2e.py`：pytest 断言（stand 60 步直立；punch 存活+伸展+脚动）
- RoundRunner 视频：`e2e_env.yaml` + `e2e_policy.yaml` → mp4；
  输出 /data1/dev/combatbench_executor/e2e/*.mp4
- WORKLOG.md：每轮尝试+结果记录
