# P0 审计报告：Humanoid21 物理契约与模型指纹

状态：完成初版审计与代码契约锁定  
范围：`contracts.py`、`model_spec.py`、对应单元测试

## 已确认的模型事实

- MuJoCo 版本：`3.8.0`
- 当前竞技场模型：`envs/humanoid21/battle_circular_v2.xml`
- 物理步长：`0.002 s`
- 重力：`[0, 0, -9.81] m/s²`
- 接触锥：`pyramidal`（MuJoCo `cone=0`）
- `impratio=10.0`
- 总自由度：`nv=54`；执行器数 `nu=42`
- 每个机器人：
  - 16 个 subtree bodies
  - 21 个受控 hinge joints / actuators
  - 总质量约 `40.844021 kg`
- 根 body：
  - `robot_a`: `torso_a`, `body_id=1`, root qpos/dof adr `0`
  - `robot_b`: `torso_b`, `body_id=17`, root qpos/dof adr `27`

## 已确认的 MuJoCo 接口事实

### 子树状态

- subtree COM 位于 `MjData.subtree_com`，不是 `MjModel` 字段。
- subtree mass 位于 `MjModel.body_subtreemass`。
- `mujoco.mj_subtreeVel(model, data)` 更新：
  - `data.subtree_linvel`：子树质心线速度
  - `data.subtree_angmom`：绕子树 COM 的角动量
- 因此线动量使用 `body_subtreemass * subtree_linvel`，角动量直接读取 `subtree_angmom`。

### 外作用字段

- `data.xfrc_applied` 是按 body 的 6D applied wrench：`[force, torque]`。
- 探针确认：`mj_forward()` 后，`xfrc_applied` 的贡献进入 `data.qfrc_smooth`，而不是 `data.qfrc_applied`。
- 使用 `mj_jacBodyCom` 构造的 COM Jacobian 时，探针得到：
  - `qfrc_smooth` 增量与 `J_com.T @ [F, T]` 完全一致（数值差异为 0）。
- 因此当前契约：
  - `xfrc_applied` 的力参考点是 body COM；
  - force/torque 均按 world-frame spatial wrench 处理；
  - `qfrc_applied` 是独立的广义力输入，不能与 `xfrc_applied` 混同。
- `Humanoid21Simulator.physical_step()` 先执行 `_apply_pd_control()` 和 `mj_step()`，随后清零：
  - `data.xfrc_applied`
  - `data.qfrc_applied`
- 因此 P2/P3 必须在调用 `physical_step()` 前采样 applied-load ledger。

### 接触方向

`contacts_vec.force_world` 沿用 MuJoCo 约定：

```text
force_world = geom1 对 geom2 施加的世界坐标力
```

因此对目标 body：

```text
body == geom2 body -> +force_world
body == geom1 body -> -force_world
```

`condim=3` 下接触是点力；不得虚构独立接触力矩。

## P0 已落地的契约

### `contracts.py`

定义：

- `CentroidalState`
  - `com_pos` 为 world-frame 全身 COM
  - `lin_mom` 为 world-frame 总线动量
  - `ang_mom` 为绕 `com_pos` 的 world-frame 角动量
- `ExternalLoad`
  - `force_world` 作用于 `point_world`
  - `torque_world` 是同参考点的自由力偶
  - `moment_about(p) = (point-p) × F + T`
- `GeneralizedLoad`
  - 表示尚未转换为空间外力/力矩的 qfrc 级输入
- `LoadSnapshot`、`BalanceStep`
- `DomainIssue`、`DomainReport`
- fail-loud 异常类型

### `model_spec.py`

实现：

- `build_model_spec(sim)`
- body / joint / actuator / geom / robot / model option spec
- 稳定 JSON payload
- `fingerprint = sha256(canonical JSON payload)`
- XML 文件 SHA-256 纳入模型身份
- `Humanoid21Meta.validate()` 前置校验

### 测试

当前 P0 测试覆盖：

- 模型步长、重力、执行器数量
- 双机器人根/body/joint/actuator 数量与质量
- fingerprint 稳定性
- ground 与 foot geom 的归属、condim、priority
- `CentroidalState` 类型校验
- `ExternalLoad.moment_about()` 力矩参考点
- `DomainReport` 非法组合拒绝

当前结果：

```text
6 passed
```

## 已知边界

- `GeneralizedLoad` 目前只记录 qfrc 输入，不尝试推断其空间外力等效形式；P2 中非零 qfrc 必须显式标记为 unsupported，除非有明确 body/Jacobian 映射。
- `xfrc_applied` 的探针已确认其 COM Jacobian 映射，但它仍是一个 MuJoCo applied-body-wrench 接口；质心层会将其作为独立外部力项记录，不把它和接触力混淆。
- `set_core_state()` 后 `mj_forward()` 只能刷新运动学/约束求解入口状态，不能恢复 stepper 内部 warm-start/contact 历史。
- 手工求和 `L` 必须使用 body inertial pose `ximat`，不能把 `xmat` 或 `xquat` 误当惯量姿态。

## 下一步（P1）

实现 `state.py` 的两条独立路径：

1. body 惯量手工求和：`m,c,l,L`
2. MuJoCo subtree：`subtree_com/subtree_linvel/subtree_angmom`

并加入：

- A/B subtree 隔离测试
- reset / 扰动姿态 / 修改 qvel 测试
- 手工与 subtree 路径一致性测试
- 根平移协变性和 yaw 旋转协变性测试
