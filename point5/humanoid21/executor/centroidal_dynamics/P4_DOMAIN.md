# P4 验证报告：适用域与接触能力检查

状态：完成  
范围：`domain.py`、`test_domain.py`

## 已实现接口

`CentroidalDomainChecker.check(state, snapshot)` 对 `CentroidalState + LoadSnapshot` 做保守适用域检查，返回 `DomainReport`。

`DomainReport` 包含：

- `valid`：是否满足第一版适用域；
- `issues`：warning/invalid 级别诊断；
- `margins`：可用于评估和排序的数值余量；
- `metadata`：本次检查使用的限制和接触计数。

## 第一版支持域

当前检查明确只支持：

- `Humanoid21` 已指纹化的模型；
- 平地 `ground` 接触；
- `condim=3` 点接触；
- 足部 geom 作为支撑接触；
- 目标 robot 无非零 `qfrc_applied`；
- root up-axis 倾斜、root 高度、COM 高度在保守阈值内；
- COM 水平投影位于实际足部接触点凸包内。

以下情况显式 invalid：

- 模型指纹不一致；
- 未知/未映射 contact；
- `robot-robot` 或 wall contact（第一阶段不把它当支撑）；
- 非足部支撑接触；
- `condim != 3`；
- contact normal 对目标 robot 不够朝上；
- 接触法向力为负；
- 切向力超过摩擦锥保守上界；
- 接触点少于 3 个或无法构成 support polygon；
- COM 投影落在 support polygon 外；
- 倾斜/高度越界；
- unsupported qfrc。

## 接触能力指标

每个 foot-ground contact 生成：

- 对目标 robot 的接触法向；
- 法向力 `normal_force`；
- 切向力 `tangent_force`；
- 摩擦系数 `mu`；
- 摩擦余量 `mu * normal_force - tangent_force`；
- 接触点相对全身 COM 的力矩幅值上界。

汇总 margins 包括：

- `total_contact_normal_force`
- `contact_friction_force_bound`
- `contact_moment_about_com_bound`
- `support_polygon_area`
- `support_polygon_margin`

这里 `contact_moment_about_com_bound` 是逐接触点力矩幅值和的保守上界，不是可任意施加的六维接触力矩。

## Reset 状态实测余量

在当前 `battle_circular_v2.xml` reset 姿态下，`robot_a`：

```text
valid                         True
total_contact_normal_force    ~356.55 N
contact_friction_force_bound  ~356.55 N
contact_moment_bound          ~50.45 N*m
support_polygon_area          ~0.0462 m^2
support_polygon_margin        ~0.0957 m
tilt_margin                   0.5 rad
root_z_margin                 ~0.582 m
com_z_margin                  ~0.450 m
```

## 测试覆盖

- reset 状态通过保守域；
- root pitch 超过阈值会 invalid；
- 合成摩擦锥违规会 invalid；
- 非零 qfrc load 会 invalid；
- support polygon signed margin 内外符号正确。

当前结果：

```text
31 passed
```
