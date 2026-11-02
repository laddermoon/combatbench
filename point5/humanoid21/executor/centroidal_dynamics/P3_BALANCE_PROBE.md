# P3 验证报告：物理子步探针与动量收支

状态：完成  
范围：`balance.py`、`probe.py`、`test_balance.py`、`test_probe.py`

## 已实现接口

`PhysicsStepProbe.probe_step(robot_id)` 执行以下顺序：

1. 在刷新动力学前缓存 `xfrc_applied/qfrc_applied`；
2. 采样目标 robot 的接触和外力 `LoadSnapshot`；
3. 提取 `CentroidalState(t0)`；
4. 调用一次 `sim.physical_step()`；
5. 提取 `CentroidalState(t1)` 和 `LoadSnapshot(t1)`；
6. 计算 `pre/post/trapezoid` 三种线动量与角动量收支残差。

`probe_steps()` 支持连续多个物理子步，并允许先设置一次符合 simulator 契约的 action dict。

## 收支定义

线动量：

\[
\Delta l=l(t_1)-l(t_0),\qquad
I_l=mg\Delta t+\sum_i F_i\Delta t
\]

角动量绕全身 COM：

\[
\Delta L=L(t_1)-L(t_0),\qquad
I_L=\sum_i\left[(p_i-c)\times F_i+T_i\right]\Delta t
\]

重力是均匀体力，绕全身 COM 不产生额外力矩。

三种积分模式：

- `pre`：使用 `t0` 接触端点力；
- `post`：使用 `t1` 接触端点力；
- `trapezoid`：平均 `t0/t1` 接触端点力；
- `xfrc_applied` 始终使用 `t0` ledger，因为它在物理步内生效并被清除。

## 关键时序修正

MuJoCo 3.8 中 `mj_forward()` 会消费/清除 `xfrc_applied`。因此 `ExternalLoadNormalizer.snapshot()` 现在先复制 applied-load ledger，再刷新 contacts。`PhysicsStepProbe` 也先采样 snapshot，再做 centroidal-state refresh。

这是实现中发现的重要坑：若先调用 `CentroidalStateExtractor.state()`，其中的 `mj_forward()` 会使后续 `xfrc_applied` 读数变成 0。

## 当前残差解释

探针报告的是**端点接触力近似的收支残差**，不是 MuJoCo 内部约束冲量的精确审计。MuJoCo 的 contact solver、关节约束和积分路径可能使端点力积分不等于实际动量变化，因此残差必须保留在结果中，不得通过改坐标或静默补偿隐藏。

当前测试要求：

- 三种 residual 必须有限且结构完整；
- pre/post/trapezoid 都被记录；
- 不把 residual 伪造成 0。

后续若需要更高精度，应单独引入 solver impulse 级接口；不能用当前端点 force 声称精确平衡。

## 测试覆盖

- 重力和 `xfrc_applied` 的合成冲量；
- trapezoid 只平均端点接触力；
- `qfrc_applied` 不能静默进入空间外力收支；
- `t1 > t0` 时间契约；
- live MuJoCo substep 的 pre/post 状态、load、residual 记录；
- applied wrench 在 `physical_step()` 前被捕获；
- action dict 契约不被 probe 改变；
- 目标 robot 的非零 qfrc fail loud。

当前结果：

```text
26 passed
```
