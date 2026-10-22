# Humanoid21 质心动力学模型实施计划

> 类型：详细实施计划  
> 状态：P0–P2 已完成，P3–P6 待实施  
> 上级设计：`mocap/HUMANOID21_CENTROIDAL_DYNAMICS_META_DESIGN.md`  
> 目标目录：`mocap/humanoid21/centroidal_dynamics/`  
> 原则：先证明“提取的质心状态与动量收支是真的”，再做预测；先证明“预测是真的”，再谈主路径规划。

## 0. 实施目标与边界

### 0.1 目标

在 `mocap/humanoid21/centroidal_dynamics/` 内落地一个可独立测试的质心动力学层，为未来“低维主路径规划 + Reclip 全身细化”提供可信输入。

第一阶段工程目标：

1. 从 `Humanoid21Simulator` 的完整物理状态中提取单机器人的：
   - 总质量 `m`；
   - 全身质心 `c`；
   - 线动量 `l = m v_c`；
   - 绕全身质心的角动量 `L`。
2. 提取并规范化作用于该机器人的外部作用：
   - 接触点力；
   - `xfrc_applied` / `qfrc_applied` 等外部施加项；
   - 重力。
3. 对物理子步做动量积分收支：
   - `l(t+dt) - l(t)`；
   - `L(t+dt) - L(t)`；
   - 重力和外力积分。
4. 建立明确的模型适用域和错误分类，禁止静默忽略未知外力或未知接触。
5. 在上述核验可信后，再做保守的离散预测层。

### 0.2 非目标

本阶段不做：

- 接触序列搜索；
- 落脚规划；
- 全身 IK / 全身轨迹优化；
- 实际控制 action 综合；
- 多机器人交互规划；
- 跌倒恢复或腾空阶段规划；
- 修改 `Humanoid21Simulator` 的仿真行为；
- 用质心模型替代 `CoreClip` / `TargetPath` / Reclip。

### 0.3 第一版适用域

第一版只支持：

- 目标机器人：`robot_a` 或 `robot_b` 任一；
- 平地；
- 给定接触模式；
- 连续接触阶段内的状态提取与动量核验；
- 无对手碰撞、无未知外力、无非脚部身体接触用于支撑；
- 预测层先以“给定外部 wrench / 接触约束”的短时推进为目标。

超出范围的输入必须返回显式错误/诊断，而不是静默简化。

## 1. 已核对的工程事实

以下事实应在实现和测试中被固定为契约；后续若模型或模拟器变化，需要重新审计。

### 1.1 模拟器与时间尺度

- `Humanoid21Simulator` 使用 `Humanoid21Meta.DT` 作为 MuJoCo timestep。
- Reclip/CoreClip 的控制周期为 50 ms（25 个物理子步）。
- 质心收支核验应基于物理子步，不应只依赖 20 Hz clip 帧。

### 1.2 机器人归属与 body 表

`Humanoid21Meta.build_runtime_tables(model)` 已提供：

- `root_body_id` / `root_body_name`；
- `body_ids_sorted` / `body_names`；
- `body_masses` / `total_mass`；
- `body_to_robot`、`body_id_to_aff`；
- `geom_id_to_name`、`geom_id_to_aff`；
- `ground_geom_id`。

这些是静态 body/geom 归属的主来源，但实现仍应在 `model_spec` 中独立校验，不默认可信。

### 1.3 现有 per-body 数据

`get_derived_state([robot_id])` 已经能给出：

- `body_xipos`：body 惯性中心世界坐标；
- `body_linvel_world`；
- `body_angvel_world`；
- `body_xpos`；
- `body_xquat`。

实现可以使用这些公开路径做独立交叉验证，但内部性能路径不应依赖完整 per-robot observation view。

特别注意：`model.body_inertia` 的局部惯性坐标系不一定是 `xquat` 表示的 body frame；实现前必须核对 `body_ipos/body_iquat` 与 `xipos/xquat` 的关系。

### 1.4 接触数据与符号

`get_derived_state(['contacts'])` 的 `contacts_vec` 已提供：

- `geom1/geom2/body1/body2`；
- `aff1/aff2`；
- `position`；
- `normal`；
- `frame`；
- `force_mag`；
- `force_world`。

其中 `force_world` 是 `geom1` 对 `geom2` 施加的力。因此：

- 若目标机器人是 `geom2`：`force_on_robot = +force_world`；
- 若目标机器人是 `geom1`：`force_on_robot = -force_world`。

当前 `condim=3` 的接触是点力，没有独立接触力矩。足部多点接触对质心的总力矩只能来自接触点位置和力臂。

### 1.5 外部施加力

`apply_external_force()` 写入 `data.xfrc_applied`，而 `physical_step()` 在 `mj_step()` 后清零该字段。

因此：

- 外部施加项必须在调用 `physical_step()` 前采样；
- 或提供受控的单步执行 API，在 `mj_step` 前后明确记录；
- 不能在 `physical_step()` 结束后回头读已经被清零的 `xfrc_applied`。

`xfrc_applied` 的作用点/坐标系不能只凭字段名判断，必须用受控实验验证：纯力、纯力矩、非质心作用点三种情况都应测。

### 1.6 CoreState 不足以完整恢复求解器上下文

`set_core_state()` 会设置 `qpos/qvel` 并 `mj_forward()`，但不会完整恢复：

- contact solver warm start；
- actuator/ctrl 内部历史；
- 外部插件/episode 状态；
- 约束求解器的瞬时缓存。

质心模型应避免把“teleport 后的静态量”和“连续物理步的受力量”混为一谈。

## 2. 模块与目录设计

建议目录结构：

```text
mocap/humanoid21/centroidal_dynamics/
├── IMPLEMENTATION_PLAN.md        # 本计划
├── __init__.py                    # 稳定公开接口
├── contracts.py                   # 数据结构、坐标/单位/错误类型
├── model_spec.py                  # 物理参数与模型指纹提取
├── state.py                       # 质心状态提取
├── external_loads.py              # 接触 + xfrc/qfrc + gravity 规范化为作用于机器人的外部量
├── balance.py                     # 动量收支与残差计算
├── probe.py                       # 物理子步采样、时间对齐与 live-sim wrapper
├── domain.py                      # 适用域、接触模式、保守能力约束
├── predict.py                     # 后续降维预测器（只在收支可信后做）
├── report.py                      # 验收数据汇总，不修改仿真
├── diagnostics.py                 # 面向图表/调试的序列指标
├── test_contracts.py
├── test_model_spec.py
├── test_state.py
├── test_external_loads.py
├── test_balance.py
├── test_probe.py
├── test_domain.py
└── test_predict.py
```

后续若加入规划接口，再新增：

```text
objective.py      # 质心层目标集合投影
constraints.py    # 接触/动量/可达性约束
solver.py         # 主路径候选求解
```

这些后续文件不属于本次第一阶段。

## 3. 核心数据契约

### 3.1 `CentroidalState`

建议字段：

```python
CentroidalState(
    robot_id: str,
    time: float,
    mass: float,
    com_pos: np.ndarray,       # (3,), world, m
    lin_mom: np.ndarray,       # (3,), world, kg*m/s
    ang_mom: np.ndarray,       # (3,), about com, world, kg*m^2/s
)
```

派生但明确标记为 derived：

```python
com_vel = lin_mom / mass
```

禁止把 `root_pos`、`root_vel` 写成 com/com_vel 的别名。

### 3.2 `ExternalLoad`

统一把“作用于目标机器人的外部作用”规范化为：

```python
ExternalLoad(
    source_type: str,          # contact_ground/contact_wall/contact_opponent/self/xfrc/qfrc/other
    body_name: str | None,
    geom_pair: tuple | None,
    point_world: np.ndarray,   # 力作用点；无点作用时定义清楚
    force_world: np.ndarray,
    torque_world: np.ndarray,  # 关于 point_world 的 free couple
    affine_to_robot: int,      # env/self/opponent/unknown
)
```

设计原则：

- 同一机器人内部接触默认不计入整体外力；
- robot-vs-opponent、robot-vs-ground、robot-vs-wall 是外部作用；
- 自碰撞接触是内部作用，但应单独计数/报告；
- 未知来源不能静默忽略。

### 3.3 `CentroidalBalanceStep`

用于每个物理子步核验：

```python
CentroidalBalanceStep(
    robot_id: str,
    t0: float,
    t1: float,
    state0: CentroidalState,
    state1: CentroidalState,
    loads: list[ExternalLoad],
    gravity: np.ndarray,
    lin_impulse: np.ndarray,
    ang_impulse: np.ndarray,
    lin_residual: np.ndarray,
    ang_residual: np.ndarray,
    diagnostics: dict,
)
```

### 3.4 错误与诊断

定义明确错误/状态码，例如：

- `UnknownExternalSource`；
- `AmbiguousContactDirection`；
- `OutOfDomainContact`；
- `StateNotConsistent`；
- `MissingAppliedLoad`；
- `PredictionDomainViolation`。

所有错误都要带 robot/body/geom/time/context，不能只返回布尔值。

## 4. 分阶段实施计划

## P0：物理契约与模型指纹审计

### 目标

在不修改仿真行为的前提下，确认所有坐标、归属、惯量、控制、接触与外部作用字段的真实含义。

### 工作

1. 新建 `model_spec.py`：
   - 从 `Humanoid21Simulator`/`Humanoid21Meta`/compiled `MjModel` 提取机器人 body/joint/actuator 集合；
   - 校验 `body_names`、`body_ids_sorted`、`total_mass`；
   - 提取 timestep、gravity、solver/cone、contact-related option；
   - 提取每个 body 的 `body_mass/body_inertia/body_ipos/body_iquat`；
   - 提取 geom `condim/friction/priority/solref/solimp/margin/gap`；
   - 生成可哈希的 `model_fingerprint`，覆盖足以影响质心动力学的字段。
2. 建立 `contracts.py`：
   - 明确单位、坐标系、wrench 表示点、外力方向；
   - 定义 `CentroidalState` / `ExternalLoad` / `BalanceStep` / error types。
3. 写 `model_spec` 测试：
   - 独立加载模型与 `sim._meta` 对得上；
   - 机器人 A/B body 数、质量、joint/DoF 数一致；
   - 模型指纹重复加载稳定；
   - 修改 body 归属时必须报错，不静默继续。

### 产出

- `contracts.py`
- `model_spec.py`
- `test_model_spec.py`
- `reports/p0_model_spec.md`（如果生成报告）

### 停止条件

- 所有 robot body/geom/actuator 映射有唯一解释；
- `xfrc_applied` 坐标/作用点、`contacts_vec.force_world` 方向、body inertia frame 都有测试结论；
- 不确定项不能进入下一阶段。

## P1：质心状态提取

### 目标

从真实 `MjData` 状态得到 `m, c, l, L`，并用至少两条独立路径交叉验证。

### 工作

1. 新建 `state.py`，提供两个实现：
   - **手工刚体求和路径**：按 `body_ids_sorted` 使用质量、COM 位置、COM 线速度、局部惯量和角速度；
   - **MuJoCo 子树路径**：在目标机器人 root body 上调用 MuJoCo 子树质心/动量接口，作为独立 cross-check。
2. 统一返回 `CentroidalState`。
3. 核对 `xipos`、`cvel`、`body_inertia` 的坐标/参考点：
   - `cvel` 的角速度/线速度排列；
   - 线速度参考点是否为 body COM；
   - `body_inertia` 的局部惯性 frame 旋转方式。
4. 不允许 fallback：两个路径显著不一致时抛 `StateNotConsistent`。

### 测试

`test_state.py` 至少覆盖：

- 静态站立状态下，手工求和与子树接口一致；
- 改变 `robot_b`，不影响 `robot_a` 的 `m,c,l,L`；
- 平移机器人只改变 `com_pos`，不改变 `ang_mom`；
- 绕 z 轴整体旋转时，`lin_mom/ang_mom` 按向量旋转协变；
- 给 `joint_vel`/`root_vel` 后，`lin_mom = m * com_vel`；
- 返回数组是 copies，不暴露 MuJoCo buffer。

### 产出

- `state.py`
- `test_state.py`
- `reports/p1_state_extraction.md`

### 停止条件

手工路径和 MuJoCo 子树路径在独立验收集上达到冻结的数值门槛；所有坐标语义写进 `contracts.py` 和报告。

## P2：外部作用规范化

### 目标

把接触、外部施加项、重力统一成“作用于目标机器人的世界系 wrench”，并避免方向错误。

### 工作

1. 新建 `external_loads.py`。
2. 接触路径：
   - 消费 `contacts_vec` 或直接消费 `MjData.contact`；
   - 按 geom/body/aff 分类；
   - 将 force 归一化为“external actor 对目标机器人”的方向；
   - 保留作用点 `position`；
   - `condim=3` 下 `torque_world` 应为零，若模型将来出现 condim>3，必须先实现真实接触 wrench 再允许。
3. 分类至少包括：
   - `contact_ground`；
   - `contact_wall`；
   - `contact_opponent`；
   - `contact_self`；
   - `contact_unknown`。
4. `xfrc_applied` / `qfrc_applied`：
   - 在物理步前采样；
   - 用受控实验确认力/力矩坐标系与作用点；
   - 转成以世界系表示、关于目标机器人质心的等效外部作用。
5. 重力单独列出，不与接触/施加项混淆。
6. 未知 `qfrc` 来源不能直接归零；至少要区分 actuator、passive、applied、constraint 或明确声明某类未支持。

### 测试

`test_external_loads.py` 覆盖：

- 脚为 geom2 时方向取正；脚为 geom1 时取反；
- 地面接触合力方向合理；
- robot-vs-robot 接触对目标机器人方向正确；
- 自碰撞不产生外部 net load；
- 单点接触只产生力，不凭空产生力偶；
- 对目标 body 施加纯外力/纯外力矩，负载记录方向正确；
- 未知 geom/body/aff 触发明确错误。

### 产出

- `external_loads.py`
- `test_external_loads.py`
- `reports/p2_external_loads.md`

### 停止条件

同一组场景下，`contacts_vec` 提取结果和直接使用 `mj_contactForce` 的低层路径一致；外力项不再有时序或方向歧义。

## P3：物理子步探针与动量收支

### 目标

在真实物理子步粒度上验证：

```text
Δl ≈ ∫(mg + external forces) dt
ΔL ≈ ∫(moments about com + free couples) dt
```

### 工作

1. 新建 `probe.py`：
   - 在 `physical_step()` 前后采样 `CentroidalState`；
   - 在 `physical_step()` 前采样 `xfrc_applied`/`qfrc_applied`；
   - 在 `mj_step()` 后采样 contact；
   - 记录当前 action、contact、step index、physics time；
   - 不修改模拟器状态。
2. 新建 `balance.py`：
   - 计算线/角动量增量；
   - 计算外部力冲量与关于 COM 的力矩冲量；
   - 输出残差、归一化残差、分量明细；
   - 记录是“积分路径”还是“微分近似”。
3. 支持单步和多步窗口；多步必须逐子步累计，不应只在首尾用平均力。
4. 处理角动量参考点移动问题：
   - 如果用各步 `p_i - c(t)`，要说明这是关于随时间变化的质心参考点的积分近似；
   - 需要同时提供关于固定参考点的 cross-check，避免移动参考点带来的隐藏项。
5. 接触力采样时间点必须与状态时间点一致；若 MuJoCo 的接触力只可在 step 后读取，报告该离散化偏差并用多场景校准。

### 测试场景

- 无运动的静止支撑；
- 无接触自由下落/受控抛物线；
- 给定根速度/关节速度的短运动；
- `xfrc_applied` 纯外力；
- `xfrc_applied` 纯力矩；
- 单脚/双脚接触；
- 存在 opponent 接触时只计入其对目标机器人的作用；
- 故意漏掉一个接触源时应报错而不是默默残差变大。

### 产出

- `probe.py`
- `balance.py`
- `test_probe.py`
- `test_balance.py`
- `reports/p3_momentum_balance.md`

### 停止条件

- 残差来源可解释；
- 静态/自由运动/外力/接触场景均通过冻结阈值；
- 若存在系统性偏差，已定位到时间对齐、惯量、接触方向或外力遗漏。

## P4：模型适用域与保守能力约束

### 目标

在预测器之前，明确哪些主路径候选“不属于本模型可承诺的范围”。

### 工作

1. 新建 `domain.py`。
2. 接触能力约束：
   - 从实际 `contact.friction`、`condim`、geom 位置和接触点集合出发；
   - 法向力非负；
   - 按 MuJoCo pyramidal cone 的实际约束处理摩擦，不做夸大能力；
   - 多点接触的合成 wrench 要保留力臂关系。
3. 保守可达性约束：
   - 第一版只检查明显的腿长/接触区域/关节构型边界；
   - 不把粗略可达性当作全身可实现证明。
4. 状态适用域：
   - 接触模式必须给定；
   - 接触丢失、非脚部身体承重、墙体接触、opponent 接触分别明确是否支持；
   - 未知外力或缺失接触记录直接标为 invalid。
5. 给出结构化 `DomainReport`：
   - valid / warning / invalid；
   - 具体违例；
   - 能力余量。

### 测试

`test_domain.py` 覆盖：

- 法向拉力被拒绝；
- 超出摩擦锥被拒绝；
- 在不存在的支撑点施加 wrench 被拒绝；
- `condim=3` 下凭空接触 torque 被拒绝；
- 接触点集合与足部几何包络被区分记录；
- 未支持接触模式返回 `OutOfDomainContact`。

### 产出

- `domain.py`
- `test_domain.py`
- `reports/p4_domain.md`

### 停止条件

所有已知会扩大机器人能力或混淆测量/规划语义的输入都能被识别为 invalid/warning。

## P5：离散预测器

### 目标

在已验证的质心状态和外部作用基础上，提供短时预测。

### 工作

1. 新建 `predict.py`。
2. 第一版预测输入：
   - 当前 `CentroidalState`；
   - 给定接触模式；
   - 每段的外部 wrench 或接触约束假设；
   - timestep。
3. 推进公式：
   - `l_{k+1} = l_k + (m g + Σf_ext) dt`；
   - `c_{k+1} = c_k + (l_{k+1}/m) dt` 或明确选定的积分格式；
   - `L_{k+1} = L_k + Σ((p_j-c_k)×f_j + τ_j) dt`。
4. 预测输出必须带：
   - 预测轨迹；
   - 约束余量；
   - domain validity；
   - 模型版本/fingerprint。
5. 与真实回放对比时，不得把真实接触力拿去“解释”预测误差后宣称预测成功；预测路径只能使用规划时可知的约束/控制假设。

### 测试

`test_predict.py` 覆盖：

- 静止支撑的一步/多步；
- 自由下落；
- 已知外力；
- 同一初态不同接触模式；
- 接触模式不合法时拒绝预测；
- 时域变长时误差报告完整。

### 产出

- `predict.py`
- `test_predict.py`
- `reports/p5_prediction.md`

### 停止条件

在声明的接触模式和外力假设下，预测误差在冻结阈值内；超出适用域时不会返回伪成功。

## P6：与现有 Reclip/BC 工作流对接（后续）

### 目标

把质心层变成主路径规划的“骨架层”，但不直接改写现有 Reclip 语义。

### 工作

1. 定义 `CentroidalPlan`：
   - `CentroidalState` 时间序列；
   - contact mode / support region；
   - external wrench reference；
   - target-set projection；
   - domain margin；
   - model fingerprint。
2. 提供 `plan -> TargetPath/Reclip constraints` 的转换接口草案，但保持 CoreClip 仍是全身轨迹格式。
3. 将细化失败反馈分类传回主路径：
   - 接触不可行；
   - 关节不可达；
   - 力矩/PD 饱和；
   - 动力学失配；
   - 求解预算耗尽。
4. 用 pilot 对比：
   - direct Reclip；
   - centroidal-guided Reclip。
5. 比较指标必须包含：
   - 任务合格数；
   - 全身细化通过率；
   - solver wall time；
   - 物理 rollout 数；
   - fall/stability/terminal quality；
   - 失败分布。

### 停止条件

只有 P0–P5 验收通过后，P6 才允许用于 expert 数据生成。

## 5. 验收数据集与场景

### 5.1 校准集

用于调阈值和发现系统性错误，不作为最终验收：

- `standing` / `squat` / `stand_on_left_leg` 初始姿态；
- 短 PD 动作；
- `xfrc_applied` 已知力与力矩；
- 单脚/双脚接触；
- 无对手接触；
- 少量 warmup replay。

### 5.2 独立验收集

- frozen warmup policy 生成的初始状态；
- Reclip expert 中已验证可回放的轨迹；
- 失败的、歪斜的、接近失衡的轨迹；
- 明确的非法外力/非法支撑反例；
- 对手接触、自碰撞、墙体接触作为域外/边界样例。

### 5.3 验收分层

必须分别报告：

1. **提取正确性**：`m,c,l,L` 是否准。
2. **受力完整性**：外部作用是否全部入账、方向是否正确。
3. **收支闭合**：动量变化是否可由外部作用解释。
4. **适用域准确性**：非法/域外输入是否被拒绝。
5. **预测正确性**：规划时可用输入是否能预测短时质心演化。
6. **系统价值**：低维主路径是否真的提高合格率和效率。

不能跳级使用后面的结果证明前面的正确性。

## 6. 数值门槛制定流程

计划不预写未经验证的固定阈值。每个阶段按以下流程确定门槛：

1. 先用校准集估计：
   - MuJoCo 重复性；
   - 两条提取路径的数值差异；
   - 离散积分残差；
   - 接触采样时间对齐误差。
2. 用归一化尺度表示误差：
   - `m`；
   - `m*g`；
   - `m*sqrt(g*l_ref)`；
   - `m*l_ref*sqrt(g*l_ref)`。
3. 冻结绝对/归一化阈值、最大/分位数和预测时域。
4. 在独立验收集上运行，不允许回头按结果放宽阈值。
5. 阈值写进 `report.py` 的验收配置，而不是散落在测试中。

## 7. 工程规则

### 7.1 不改仿真语义

- 本模块只读 `model/data` 和公开接口；
- 不改变 `physical_step()` 的物理结果；
- 若为了精确记录需要在 `mj_step()` 前后插桩，必须通过 wrapper 或显式 hook 设计，不改变 simulator 默认路径；
- 不修改 XML、接触参数、PD 参数、观测契约。

### 7.2 不静默回退

- 缺 body、缺 joint、缺 contact 字段、未知 aff、未知外力来源，直接 raise；
- 不把缺失 wrench 当 0；
- 不把静态接触位置当实际受力；
- 不用 normalized joint velocity 代替物理速度；
- 不用 `feet_forces` 的体重归一化量做动量收支。

### 7.3 可复现

- 所有测试固定 seed；
- 每个结果记录 model fingerprint、simulator 版本、dt、robot_id；
- 不使用可变全局缓存；
- 返回 numpy copies，不共享 MuJoCo buffer。

### 7.4 性能目标

- 状态提取应是轻量只读操作；
- probe 不能改变物理结果；
- 初期不强制实时，但要记录每次提取/每步开销；
- 不能把大额 numpy/object allocation 放进每个物理子步的热循环。

## 8. 计划中的关键风险

| 风险 | 触发方式 | 应对 |
|---|---|---|
| `body_inertia` frame 用错 | 手工 L 与子树角动量系统性不一致 | P0/P1 单独审计，不进入下一步 |
| `xfrc_applied` 坐标/作用点误读 | 外力场景收支不闭合 | 受控纯力/纯力矩实验验证 |
| `contacts_vec` 时间对齐偏差 | 单步残差系统性偏一个方向 | 前后状态/力采样方案对照，多步积分 |
| 接触点力被误写成足踝 wrench | 角动量残差或非法能力扩大 | 保留点力与力臂，域检查禁止虚构 torque |
| `set_core_state` 未恢复 solver 状态 | teleport 后接触/动量不可复现 | 收支只在 live physical_step 上验；teleport 只做静态检查 |
| 低维预测扩大能力 | 预测要求不存在的接触/力矩 | P4 domain gate 必须先完成 |
| 成功定义混入视觉质量 | 质心可行动作仍歪斜 | 任务完成、稳定接管、姿态质量分开验收 |
| 模块变成规划器 | 提前写 solver 掩盖底层错误 | P0–P5 停止条件不满足时不进入规划 |

## 9. 里程碑验收清单

### P0 完成

- [ ] model fingerprint 可复现；
- [ ] body/geom/joint/actuator 归属表通过一致性测试；
- [ ] `xfrc_applied`、`contacts_vec.force_world`、`body_inertia` 坐标语义有实测结论。

### P1 完成

- [ ] `CentroidalState` API 冻结；
- [ ] 手工求和与 MuJoCo 子树接口一致；
- [ ] 平移、旋转、robot 隔离测试通过。

### P2 完成

- [ ] 所有接触被分类；
- [ ] 外部作用方向可解释；
- [ ] 未知来源显式报错。

### P3 完成

- [ ] 静止、自由运动、外力、接触场景收支闭合；
- [ ] 单步和多步残差都有报告；
- [ ] 系统性误差被解释或消除。

### P4 完成

- [ ] 非法接触/非法 wrench/域外输入被拒绝；
- [ ] 接触能力不夸大；
- [ ] 适用域文档与诊断齐全。

### P5 完成

- [ ] 短时预测误差通过冻结阈值；
- [ ] 域外输入不会输出伪成功；
- [ ] 预测速度与误差均有报告。

### P6 完成

- [ ] `CentroidalPlan` 与全身细化接口明确；
- [ ] 与 direct Reclip 在同预算下对比；
- [ ] 只在系统性指标更优时才进入大规模 expert 生成。

## 10. 建议执行顺序

推荐先按最小闭环实现：

1. `contracts.py`
2. `model_spec.py` + `test_model_spec.py`
3. `state.py` + `test_state.py`
4. `external_loads.py` + `test_external_loads.py`
5. `probe.py` + `balance.py` + 相关测试
6. `domain.py`
7. `predict.py`

每个 P 阶段完成都必须：

- 运行新增测试；
- 运行 `mocap/humanoid21/clip` 现有测试确认无回归；
- 生成或更新阶段报告；
- 更新本计划中的已完成状态；
- 单独 commit + push。

首个实现 PR 建议只包含 P0–P1：契约、模型指纹、状态提取和交叉验证。这样可以在未接触复杂接触力语义前，先锁定最难被察觉的坐标/惯量错误。
