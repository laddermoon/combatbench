# P6 验证报告：CentroidalPlan 对接接口与同预算对比框架

状态：完成  
范围：`planning.py`、`test_planning.py`

## 接口边界

`CentroidalPlan` 是一次 motion request 的质心层输入包，包含：

- `initial_state`：经手工求和 + MuJoCo subtree 交叉验证的 `CentroidalState`；
- `initial_loads`：规范化后的接触与 applied-load `LoadSnapshot`；
- `domain`：第一版保守适用域 `DomainReport`；
- `target_summary`：`TargetPath` 模式、路径节点、`root_pos_mode`；
- `objective_summary`：Reclip objective 类型与预算；
- `budget_frames`；
- `model_fingerprint`。

`CentroidalPlan` 明确不是全身 action plan，`metadata["is_full_body_executable"] == False`。

`PlanContext` 把一次对比所需的共享输入固定下来：

- simulator；
- `robot_id`；
- 初始 `CoreState`；
- 对手 `CoreState`；
- `TargetPath`；
- `ReclipObjectiveConfig`；
- `budget_frames`；
- `CentroidalPlan`；
- `CentroidalModelSpec`。

## Reclip 对接

`direct_reclip_solver(**solver_kwargs)` 生成一个 `SolverMethod`，内部调用现有 `reclip_path_sweep()`：

```python
reclip_path_sweep(
    ctx.start_state,
    ctx.path,
    ctx.objective,
    budget_frames=ctx.budget_frames,
    robot=ctx.robot_id,
    sim=ctx.sim,
    **solver_kwargs,
)
```

因此比较框架不会修改 Reclip 的 objective、path、budget 或 replay semantics。

## 同预算对比

`run_same_budget_comparison(ctx, methods)` 接收：

```python
{
    "direct": direct_reclip_solver(...),
    "centroidal_guided": future_solver,
}
```

每个 method 收到同一个 `PlanContext`，运行前会把 simulator 重置到同一个 robot/other CoreState。输出包含：

- `status` / `completed`；
- `solver_score`；
- `event_frame`；
- `budget_frames`；
- `control_frames_executed`；
- `executed_physics_steps`；
- `wall_time_s`；
- `fall_count`；
- `path_progress_final`；
- `path_error_max`；
- `terminal_pose_error`；
- `terminal_vel_error`；
- `action_abs_mean`；
- `action_saturation_frac`；
- `replay_error`；
- `clip_T`；
- centroidal domain validity/issues/margins。

solver exception 会被记录为 `status="solver_exception"`，不会让整批比较中断；异常类型和消息保留。

## 尚不声称的能力

当前只提供对比框架和未来 centroidal-guided solver 的调用点。它还没有：

- 生成接触序列；
- 搜索落脚点；
- 生成全身 action；
- 替代 Reclip expert generation；
- 证明 centroidal feasible => full-body executable。

真实 centroidal-guided method 必须通过同一个 `SolverMethod -> ReclipSolveResult` 契约进入比较，并继续接受 replay validation。

## 测试覆盖

- `CentroidalPlan` 能收集 state/load/domain/target/objective；
- 多 solver 同预算输出被规范化；
- 每个 method 运行前恢复同一 CoreState；
- solver exception 被记录；
- `direct_reclip_solver` 工厂可构造。

当前结果：

```text
39 passed
```
