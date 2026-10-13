# M0 资产审计报告

> 状态：进行中（随工作包推进增量更新）
>
> 对应大纲：`README.md` §9 M0「资产审计与契约」。
> 退出标准：来源可追溯，片段真实内容与关节语义明确。

## W1 机器人模型事实核查（battle_circular_v2.xml）

事实来源：MuJoCo 3.8.0 编译后模型（`mocap/humanoid21/robot_spec.py` 提取，
机器可读结果见 `robot_spec.json`，经 `Humanoid21Meta.validate()` 通过）。

### 控制语义（已核实，与 CONTROLSPEC.md 一致）

- action ∈ [-1,1]²¹ → 目标角 = `mid + action·half_range`（mid/half 来自 jnt_range）
- 固定 KP/KD 的 PD 力矩 → `ctrl = clip(torque/gear, ctrlrange)`
- 物理步 0.002 s（500 Hz），`phy_steps_per_action=25` → **控制频率 20 Hz**
- 自检：standing 初始 action 解码后与 `joint_pos=0` 一致（max err 4.4e-5 rad）

### 关节轴事实（世界系，standing 姿态；此时 root 朝 +x）

| 语义组 | 关节 | 世界轴 | 限位(°) | gear | 备注 |
|---|---|---|---|---|---|
| 躯干 | abdomen_z/y/x | z / y / x | [-45,45] [-75,30] [-35,35] | 40 | 干净主轴 |
| 右腿 | hip_x / hip_z / hip_y | +x / +z / +y | [-30,10] [-60,35] [-150,20] | 40/40/120 | 干净主轴 |
| | knee | **−y** | [-160,2] | 80 | 负轴 |
| | ankle_y / ankle_x | +y / **(x,z)混合(0.89,0,0.45)** | [-50,50] ×2 | 20 | ankle_x 非纯轴 |
| 左腿 | hip_x / hip_z | **−x / −z** | 同右 | 40 | 镜像负轴 |
| | knee / ankle_y | −y / +y | 同右 | 80/20 | |
| | ankle_x | **(−0.89,0,−0.45)** | [-50,50] | 20 | 混合轴+镜像 |
| 右臂 | shoulder1 | **(0.82,0.41,0.41)** | [-85,60] | 20 | 混合轴，非纯屈伸 |
| | shoulder2 / elbow | **(0,−0.71,0.71)** | [-85,60] / [-100,50] | 20/40 | 同轴两关节 |
| 左臂 | shoulder1 | (−0.82,0.41,−0.41) | [-85,60] | 20 | 镜像 |
| | shoulder2 / elbow | (0,−0.71,−0.71) | 同右 | 20/40 | |

### 关键发现

1. **Hamstring 肌腱耦合**：`0.5·hip_y − 0.5·knee ∈ [−0.3, 2] rad`，即
   `hip_y − knee ∈ [−34.4°, +229.2°]`。髋部深屈（hip_y 负向）+ 膝伸直组合违反约束
   ——例如"直腿高抬"类人动作物理不可行。重定向必须逐帧检查，而不是只查单关节限位。
2. **手臂不是直链**：shoulder 轴为混合轴且 upper_arm→lower_arm→hand 的
   偏移是折线（0.18,−0.18,−0.18)→(0.18,0.18,0.18)。**逐关节角度复制（retarget_v2
   思路）在手臂上必然错误**；M1 必须用基于世界系姿态/位置的 IK 式重定向。
3. **语义标签纠正**：retarget_v2 把 CMU femur ry(abduction) 映射到 hip_z，
   但实际 hip_x（绕世界 x）才是侧展/roll，hip_z 是扭转/yaw——语义错位。
4. **JOINT_LIMITS/JOINT_ORDER 与当前模型完全一致**（v1→v2 限位未变；
   与 meta.CONTROLLED_JOINTS 顺序一致，max diff < 1e-14 deg）。旧表的数值可复用，
   但映射语义需按真实轴重推导。
5. 体形：总质量 40.8 kg，腿长 thigh≈0.40 m / shin≈0.39 m，与人比例明显不同；
   root 为 freejoint（root_a/root_b），nq=56, nv=54, nu=42（双机）。
6. contact excludes 仅 waist_lower↔thigh（左右×双机）——其余自碰撞均生效，
   重定向姿态需防自碰。
7. 旧代码勘误：`baseline/humanoid21/mocap/visualize.py` 引用
   `Humanoid21Meta.ARENA_XML`——该属性实际在 `Humanoid21Simulator.ARENA_XML`，
   旧可视化脚本在当前代码下 import 即报错，属过时件。

## W2 存量资产盘点

工具：`mocap/humanoid21/audit_assets.py`；结果：`assets_index.json`（sha256/帧数/质量指标全量）。

### 原始数据（baseline/humanoid21/mocap/data/raw/，已在 git）

- 4 个 ASF（subject 13/14/15/17）：31 bones、56 DOF、units_length=0.45（inch 系）、
  units_angle=deg、root order `TX TY TZ RX RY RZ`——四者骨架定义一致。
- 29 个 AMC：全部解析成功，无缺帧缺骨、无非有限值。
  时长 3.5–77.8 s（按 CMU 标准 120 Hz 假定值折算，待 W4 官方描述交叉确认）。
- 时长分布：13_01~24（9–48 s）、14_01~03（~46 s）、15_13（78 s）、17_10（23 s）。

### 重定向产物（data/retargeted/，已在 git）

- 29 组 `(T,21)` 角度 npy + 29 个 `_norm` 归一化版本，全部有限、零限位违例
  （容差 1e-4°，最大超界 7e-15 即浮点噪声）、无冻结帧。
- **6.5–15.6% 的关节值恰好压在限位上**——DOF-copy 重定向的裁剪痕迹，
  人体关节角超出机器人限位时被静默截断（主要是肩/髋大幅动作帧）。
- 数据只有关节角：无根部轨迹、无关键点、无接触、无速度。
  **结论：可视为"低可信度的运动学近似"，不满足大纲对参考运动的最低信息要求。**

### 旧代码可复用性裁定

| 文件 | 裁定 | 依据 |
|---|---|---|
| `asf_parser.py` | 复用 | 4/4 ASF 解析正确，字段完整 |
| `amc_parser.py` | 复用 | 29/29 AMC 解析正确；依赖"AMC 行序=ASF dof 序"约定（CMU 成立） |
| `fk.py` | 复用（待 W3 数值验证） | CMU FK 约定实现完整；W3 用骨架尺寸合理性校验 |
| `retarget.py` | 参考实现候选 | 走"FK→相对旋转→分解到真实关节轴"路线，含混合轴处理，思路正确；未经数值验证 |
| `retarget_v2.py` | 过时 | DOF 直拷+符号猜测；肩/踝混合轴与 hip_x/hip_z 语义错位；当前 npy 均由它生成 |
| `visualize.py` | 损坏 | `Humanoid21Meta.ARENA_XML` 不存在（实际在 simulator），且依赖交互 viewer |
| `prepare_data.py` | 模式参考 | 写死 subject 列表与代理地址；下载流程可参考 |

## W3 片段内容核验

（待执行）

## W4 许可核验

（待执行）

## W5 数据契约

见 `DATASPEC.md`（待执行）

## W6 最小动作集

（待执行）

## M0 退出标准自查

（待执行）
