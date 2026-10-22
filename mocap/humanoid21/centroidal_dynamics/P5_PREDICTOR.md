# P5 验证报告：短时离散质心预测器

状态：完成  
范围：`predict.py`、`test_predict.py`

## 模型

`DiscreteCentroidalPredictor` 使用固定 MuJoCo 物理步长 `dt` 的半隐式 Euler：

\[
F_{net}=mg+\sum_i F_i
\]

\[
\tau_{net}=\sum_i\left[(p_i-c_k)\times F_i+T_i\right]
\]

\[
l_{k+1}=l_k+dt\,F_{net}
\]

\[
L_{k+1}=L_k+dt\,\tau_{net}
\]

\[
c_{k+1}=c_k+dt\,l_{k+1}/m
\]

该模型显式预测 `m,c,l,L`，但不声称能生成全身 action。

## 输入契约

每一步输入是一个已规范化的 `LoadSnapshot`：

- `robot_id` 必须与 state 一致；
- `time` 必须与当前 state time 一致；
- `gravity` 必须与 model spec 一致；
- 不允许 `generalized_loads`；
- 默认只允许：
  - `contact_ground`
  - `xfrc_applied`
- `contact_wall` / `contact_opponent` / 其他 source 默认抛出 `PredictionDomainViolation`；
- 点接触不允许非零独立 `torque_world`；
- state mass 必须与当前 fingerprint 的 robot subtree mass 一致。

## 与真实 rollout 的比较

测试中使用 `PhysicsStepProbe` 记录真实物理子步，然后用同一个 `snapshot0` 做一步预测。

由于预测器使用与 `pre` 收支完全相同的端点力积分：

```text
predicted_lin - actual_lin = -pre_linear_residual
predicted_ang - actual_ang = -pre_angular_residual
```

测试锁定这一关系。当前 reset 后单步典型误差：

- 线动量误差约 `1e-5 kg*m/s`；
- 角动量误差约 `3e-6 kg*m²/s`。

这只是“端点力近似”的误差，不代表未来更长时间预测或接触切换预测精度。预测器的结果必须继续使用 residual/diagnostic 而不是伪装成精确动力学。

## 测试覆盖

- 重力 + `xfrc_applied` 的一步积分；
- 与真实 MuJoCo 子步的 pre-balance 误差一致性；
- 非 ground contact 默认拒绝；
- qfrc 输入拒绝；
- snapshot/state 时间不一致拒绝。

当前结果：

```text
35 passed
```
