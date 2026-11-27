> **状态（2026-11-26 更新）：** 本文档描述的 SimChar 全系统对齐方案（gear/KP/KD 覆盖）
> **已放弃并回滚**（实验证明动画级超人扭矩不利于从零探索的站立学习，且参数耦合
> 复杂度远超预期）。但其中"目标范围扩展"（alpha / scale_eff 机制）已独立保留并
> 重新落地为 `Humanoid21Simulator(target_beta=...)` 构造参数——不含任何关节
> 参数修改。本文档 §3 的 alpha 推导仍然有效，其余与 profile/gear 相关的内容
> 仅作历史参考。

# Actuation Profile 设计说明：SimChar 对齐版执行器配置

> 本文档记录 `actuation_profile` 机制的设计与 `simchar` profile 的全部决策依据。
> 所有依赖的外部数据均在文末给出来源链接。

## 1. 背景与问题

humanoid21 的执行器参数继承自 DeepMind Control Suite 的 humanoid 模型。
该模型原本面向 **direct torque control**（策略直接输出力矩），而 humanoid21
在外层封装了位置 PD 控制，导致两个结构性问题：

1. **部分关节扭矩不足**：踝 20 Nm、腰 40 Nm、髋侧向/旋转轴 40 Nm，
   对高动态搏击明显偏弱（踝 20 Nm 甚至不满足双脚踮脚尖的静力学需求 ≈28 Nm）；
2. **接近限位时力矩收敛**：action ∈ [-1,1] 被精确映射到机械关节限位，
   PD 控制律 `τ = KP(target - q) - KD·q̇` 在 `q → target` 时力矩趋零，
   关节无法在接近限位时持续输出满力矩。

## 2. 总体方案：ActuationProfile 机制

在 `Humanoid21Simulator` 构造函数新增 `actuation_profile` 参数（默认 `'legacy'`），
blueprint 的 `simulator.config` 可直接透传：

```yaml
simulator:
  cls: "envs.humanoid21.simulator:Humanoid21Simulator"
  config:
    actuation_profile: "simchar"   # 缺省为 legacy
```

### 2.1 关键机制决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 扭矩修改方式 | **运行时覆盖 `model.actuator_gear`**，不改 XML | 保持单一模型数据源，避免多 XML 漂移；覆盖 mjModel 参数是 MuJoCo domain randomization 的标准做法 |
| legacy 兼容 | legacy = 空覆盖 + beta=0，**同一代码路径**，无 if/else 分叉 | beta=0 时 alpha 严格为 1.0，scale_eff 与 scale 逐位一致；已用轨迹 sha256 逐位验证（见 §6） |
| 观测归一化 | **不变**，仍按机械限位归一化 | 观测语义（"关节处于物理边界的什么位置"）与控制目标范围解耦；policy 需要知道真实物理边界 |
| KP/KD | 纳入 profile 覆盖项 | 扭矩与增益是耦合系统，见 §3.3 |

### 2.2 目标范围扩展公式（alpha）

设计准则（用户提出）：**关节到达机械限位时，剩余 error 恰好仍能输出满力矩**。

推导（零速度假设）：满力矩要求 `KP·err ≥ gear`（ctrl 饱和条件），
即饱和角 `e_sat = gear / KP`。要求关节在限位 `q_upper` 处仍有
`target_max - q_upper = e_sat`，而 `target_max = reference + scale·alpha`、
`q_upper = reference + scale`，得：

```
alpha = 1 + beta · gear / (KP · scale)
```

其中 `scale = (upper - lower) / 2` 为机械半幅，`beta` 为余量系数
（1.0 = 严格刚好；>1 可补偿带速接近限位时 `KD·q̇` 阻尼损耗）。
alpha 逐关节由 gear/KP/scale 机械推导，**不是手调超参**。

策略接口不变：action ∈ [-1,1]，内部 target = reference + scale·alpha·action。

## 3. SimChar 对齐的决策依据

### 3.1 对齐对象：为什么是 DeepMimic/SimChar

DeepMimic（彭学斌）的 humanoid3d 是**已验证能完成后空翻、旋风踢、滚动起身等
高动态动作**的位置 PD 控制人形，与 humanoid21 控制架构同构
（policy 输出关节目标角 → 底层 PD → 力矩），体重几乎相同
（45 kg vs 本模型 40.844 kg），是最接近的存在性证明参照。

SimChar 源参数（humanoid3d）：chest/hip TorqueLim 200、neck 50、shoulder 100、
knee 150、ankle 90、elbow 60 Nm；KP：chest 1000 / hip·knee 500 / ankle·shoulder 400 /
elbow 300 / neck 100；KD 全部为 0.1×KP；policy 30 Hz，底层 PD ~600 Hz。
来源见 §7 链接 L1–L4。

### 3.2 质量缩放：× 0.9076

本模型单台质量 **40.844 kg**（mjModel.body_mass 实测总和 81.688 kg / 2），
SimChar humanoid3d 为 **45 kg**（BodyDefs 质量求和，见 L1）。
扭矩需求随体重缩放，故全部扭矩 cap × 40.844/45 = **0.9076**。

### 3.3 系统对齐：扭矩与 KP/KD 必须一起搬

只搬扭矩不搬 KP 会导致 alpha 爆炸：alpha 公式中 `e_sat = gear/KP`，
gear 放大而 KP 不变时 e_sat 同比放大（如踝 gear 82 / KP 100 → e_sat 47°，
alpha 1.94）。SimChar 自身的 e_sat 全部维持在 0.2–0.4 rad（11°–23°）
的温和区间，因为它的 KP 随扭矩同步放大。**因此对齐单位为整组
(gear, KP, KD) 三元组，不单独对齐扭矩。**

KD 维持 SimChar 的 0.1×KP 比例——该比例是作者公开说明的调参启发式
（先设 kd = 0.1·kp，再把 kp 调到刚好能跟踪而不失稳，见 L5），
且与 humanoid21 原有 KP/KD 比例一致，不构成额外变量。

### 3.4 spherical → hinge 映射：每轴全额

SimChar 的髋/踝/肩/腰为 spherical 关节，扭矩上限是三维力矩**向量范数**；
humanoid21 为串联 hinge，每轴独立 cap。采用**每 hinge 轴给全额对齐值**：
保证任意单方向最大能力与 SimChar 一致。代价是三轴同时发力时能力包络
大于 spherical 球面（立方体 vs 球），但该工况极少出现，且替代方案
（按方向使用频率分配）需要引入无依据的主观比例，违背可解释性原则。

### 3.5 hip_y 一并对齐到 182 Nm

hip_y 现状 120 Nm 虽接近人体需求，但本 profile 的目标是**严格系统对齐**，
且 SimChar 的 spherical hip 三轴共享 200 Nm 范数上限——矢状面发力时
可用接近全额。若实验表明摆动腿动力过剩，可在后续 profile 中单独下调。

### 3.6 被动动力学差异：已知但不做补偿

SimChar 关节**无任何被动刚度/阻尼**（角色定义文件中无 stiffness/damping
字段，全部力矩来自 PD，见 L1/L6）；humanoid21 大关节自带 stiffness 10–20、
damping 5、踝 6/3。这意味着同等 KP 下本模型有效刚度更高、静态重力矩
部分由被动元件承担。由于被动元件的影响依赖姿态无法简单折算，
**不做补偿，作为已知偏差记录**；其稳定性影响需通过训练实验评估。

### 3.7 控制频率不对齐

SimChar：policy 30 Hz / PD ~600 Hz；humanoid21：policy 20 Hz / PD 500 Hz。
30 Hz 在 500 Hz 物理下为 16.67 子步非整数，无法干净对齐；
且决策频率不是当前的瓶颈，**维持不变**。

### 3.8 beta = 1.0

严格按"恰好满力矩"准则，不留余量。带速冲向限位时阻尼项会削弱有效扭矩，
如需补偿可调 beta（1.1–1.2），作为后续实验旋钮暴露。

## 4. 最终配置表（simchar profile）

| 关节组 | gear (Nm) | KP | KD | e_sat | alpha | 原 gear |
|---|---:|---:|---:|---:|---:|---:|
| abdomen z/y/x | 182 | 1000 | 100 | 10.4° | 1.20–1.30 | 40 |
| hip_x | 182 | 500 | 50 | 20.9° | 2.04 | 40 |
| hip_z | 182 | 500 | 50 | 20.9° | 1.44 | 40 |
| hip_y | 182 | 500 | 50 | 20.9° | 1.25 | 120 |
| knee | 136 | 500 | 50 | 15.6° | 1.19 | 80 |
| ankle_y / ankle_x | 82 | 400 | 40 | 11.7° | 1.23 | 20 |
| shoulder1/2 | 91 | 400 | 40 | 13.0° | 1.18 | 20 |
| elbow | 54 | 300 | 30 | 10.3° | 1.14 | 40 |

注：hip_x alpha 偏高源于其机械半幅仅 20°（窄行程关节的固有性质，
不随参数配置改变）；其余关节 alpha 均在 1.14–1.44 的温和区间。

## 5. 风险提示

1. **这是动画级（超人）执行器配置**：SimChar 面向视觉效果，无功率真实性约束。
   对照参照系——Unitree H1 实物机器人踝 45–59 Nm@47kg、膝 360 Nm、髋 220 Nm（L7）；
   人体跑步踝峰值力矩 ~2.9 Nm/kg、跳落地膝/髋 3.1–3.4 Nm/kg（L8/L9）——
   simchar profile 的踝 2.0、髋 4.5 Nm/kg 处于"动画级"，高于实物机器人与人体需求。
   双机器人都是超人执行器时，打击/被击飞动力学可能失真。
2. **撞限位成为常态工况**：目标范围超出限位后，饱和动作会持续满力矩顶在
   机械限位上，RL 可能学会利用限位约束反作用力。需监控 limit-hit 占比与约束力量级。
3. **无 torque-speed curve**：MuJoCo motor 在高速下仍可输出满扭矩，峰值功率
   （τ·ω）需列入训练监控。
4. 旧 checkpoint 与新 action 语义不兼容，需重新训练。

## 6. 验证结果（2026-11-26）

| 测试 | 结果 |
|---|---|
| legacy 回归 | 200 步固定动作序列轨迹 sha256 与改动前**逐位一致**（1411fabf…）；alpha 全 1.0 |
| simchar 饱和角性质 | 5 个抽查关节 ×上下限位：关节置限位 + 满格动作 → ctrl 精确 = ±1.0，残差 err 与 e_sat 理论值逐位吻合（全部 PASS） |
| 数值稳定性 | 15000 物理步零动作 + 300 步随机动作压力测试，全程 finite 无 NaN |
| 开环站立对照 | legacy 与 simchar 开环均倒地（0.100 vs 0.104 m）——双足人形无反馈控制的预期现象，非 profile 退化 |

后续建议验证（未做）：benchmark.py 正弦/阶跃跟踪对比、limit-hit ratio 与
峰值功率监控、被动刚度对 KP 稳定性的影响。

## 7. 外部数据来源

- **L1** DeepMimic humanoid3d 角色定义（质量/关节类型/TorqueLim/无被动刚度）:
  https://github.com/xbpeng/DeepMimic/blob/master/data/characters/humanoid3d.txt
- **L2** PD 增益配置（Kp/Kd 表, UpdateRate 30）:
  https://github.com/xbpeng/DeepMimic/blob/master/data/controllers/humanoid3d_ctrl.txt
- **L3** PD 控制律实现（revolute/spherical, exponential map）:
  https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/sim/PDController.cpp
- **L4** spherical 扭矩上限为向量范数 clamp:
  https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/sim/SimBodyJoint.cpp
- **L5** 作者调参启发式（kd = 0.1·kp）— GitHub issue #174:
  https://github.com/xbpeng/DeepMimic/issues/174
- **L6** PD 目标范围宽于机械限位的 action bounds 构造:
  https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/sim/CtCtrlUtil.cpp
- **L7** Unitree H1 官方规格（膝 360 / 髋 220 / 踝 45–59 / 臂 75 Nm, 47 kg）:
  https://www.unitree.com/h1/
- **L8** 跑步等运动踝跖屈力矩 ~2.9 Nm/kg（Deschamps 2020 引用）:
  https://doi.org/10.1002/jfa2.70078
- **L9** 单腿跳落地膝/髋/踝力矩 2.8–3.4 Nm/kg:
  https://www.jstage.jst.go.jp/article/jpts/25/9/25_jpts-2013-094/_pdf
- **L10** 出拳力量躯干旋转贡献 ~45.5%（boxing jab 分析）:
  https://ojs.ub.uni-konstanz.de/cpa/article/download/3649/3430
- **L11** 拳峰峰值速度 11.95±1.84 m/s、肩角速度 ~1070°/s（15 名拳击手 3D 动力学）:
  https://chesterrep.openrepository.com/server/api/core/bitstreams/3e769a4f-d46b-4921-a2e9-093e0a80e436/content
- **L12** 髋外展 torque 与跆拳道踢击脚速显著相关:
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0235582
