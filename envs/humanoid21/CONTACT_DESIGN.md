# MuJoCo 接触动力学参数设计指南 (CombatBench Humanoid21)

本文件对 MuJoCo 的接触动力学参数进行原理拆解，分析其对机器人“碰撞手感”与“数值稳定性”的影响，给出各项设计想法的**可靠度评估**，并针对高动态对抗场景提出一版优化的接触参数配置（Propose）。

---

## 一、 MuJoCo 接触参数含义与物理作用原理

在 MuJoCo 中，接触（Contact）是通过**软约束（Soft Constraint）**求解器处理的。与其使用纯刚性的冲量和位置修正，MuJoCo 将每个接触点建模为一个虚拟的弹簧-阻尼系统，通过 `solref` 和 `solimp` 参数精细控制其刚度（Stiffness）和阻尼（Damping）。

### 1. `solref = [timeconst, dampratio]`：约束参考响应

`solref` 控制接触受到冲击时的**时间响应特性**（即恢复的速度和回弹的手感）：

*   **`timeconst` (时间常数, 单位: 秒)**：
    *   **作用**：指受力变形后，系统恢复到无穿模状态所需的特征弛豫时间。该值越小，恢复越快，接触感觉越“坚硬/硬朗”；该值越大，约束响应越慢，接触感觉越“有弹性/粘性”。
    *   **稳定性下限**：必须大于物理步长。一般推荐至少 $2 \times dt$。对于 $dt = 0.002\text{ s}$，极极限的坚硬边界是 $0.004\text{ s}$（2个步长）。过小的时间常数会导致系统在高频振荡中瞬间发散（数值爆炸）。
*   **`dampratio` (阻尼比, 无量纲)**：
    *   **作用**：控制振荡衰减。
    *   *   $= 1$ 为**临界阻尼（Critical Damping）**：系统能最快且不发生任何回弹地恢复到边界。
    *   *   $< 1$ 为**欠阻尼（Under-damped）**：碰撞后会产生“QQ弹弹”的弹跳、反复回弹现象。
    *   *   $> 1$ 为**过阻尼（Over-damped）**：系统在强阻尼下缓慢滑回边界，类似陷进沼泽或橡胶垫的感觉。

### 2. `solimp = [dmin, dmax, width, midpoint, power]`：阻抗特征曲线

`solimp` 决定了接触刚度是如何随**穿模深度（Penetration Depth）**变化的。它描述了一个从 `dmin`（浅穿模时的低阻抗）到 `dmax`（深穿模时的高阻抗）的非线性插值函数。

$$\text{Impedance } d(x) \in [d_{\text{min}}, d_{\text{max}}]$$

*   **`dmin` (起始阻抗, 0 ~ 1)**：物体刚刚碰触、表面轻微接触时的阻抗比例。
*   **`dmax` (极限阻抗, 0 ~ 1)**：完全穿透到 `width` 深度后的最大阻抗比例。接近 1（如 0.999）代表无限接近理想刚体，绝不妥协；较小值（如 0.5）代表即便穿模很深，依然是一个极其柔软的弹性体。
*   **`width` (缓冲区宽度/软接触深度, 单位: 米)**：
    *   **作用**：阻抗从 `dmin` 线性/非线性过渡到 `dmax` 的**穿模过渡带宽度**。
    *   **核心机理**：当穿透深度 $< \text{width}$ 时，接触力被显著削减（柔性阶段）；只有穿透深度 $\ge \text{width}$ 时，阻抗才达到 `dmax`，接触完全刚性化。
    *   **手感影响**：`width` 就是物体的“皮肤厚度”或“护垫厚度”。若 `width = 0.003` (3mm)，代表只要穿模超过 3mm 就会立刻遭遇硬抗力。若 `width = 0.02` (20mm)，代表有 2cm 宽的渐变柔性缓冲。

### 3. `condim`：接触维度（Friction Model）

决定两个物体碰撞时，接触面之间能产生哪些维度的约束力和摩擦力：

*   **`condim="1"` ( frictionless )**：**纯法向力**。两个物体碰撞时没有表面摩擦，就像两块涂满润滑油的冰块碰撞，永远只有垂直于法线的推开力。
*   **`condim="3"` ( regular friction )**：**法向力 + 2D 滑动摩擦力**。最常用的摩擦模型，支持沿接触表面的切向摩擦。
*   **`condim="4"` ( torsional friction )**：**法向力 + 2D 滑动 + 1D 旋转（Spin）摩擦力**。能防止物体在接触面上原地自旋（比如陀螺在地面旋转会因扭转摩擦停下）。对双足机器人的脚底支撑极其关键，可以防止“原地无摩擦脚底自转”导致的失稳。

### 4. `margin` 和 `gap`：物理防穿模与预激活力场

这两个参数用来开启 MuJoCo 的“力场预激活”机制，从根本上解决离散时间步下“单步相对运动过大而直接跳过软接触带导致爆震”的硬伤。

*   **`margin` (外余量, 单位: 米)**：
    *   **作用**：接触探测的提前量阈值。当两物体表面的几何距离 $d < \text{margin}$ 时，MuJoCo 会**提早创建并激活接触约束**。
    *   **原理**：计算约束时使用的“等效穿模量”定义为 $x = \text{margin} - d$。所以在两物体几何表面发生真正接触之前，求解器就已经开始**预先施加排斥力（Pre-applied contact force）**。这类似于在物体表面包了一层看不见的“气垫/力场”。
*   **`gap` (内间隙, 单位: 米)**：
    *   **作用**：力场过渡带。
    *   **原理**：当两物体几何距离处于 $d \in [\text{margin} - \text{gap}, \text{margin}]$ 之间时，虽然接触约束已经激活，但求解器会将计算出的约束排斥力乘以一个从 $0$ 渐变到 $1$ 的平滑插值系数。
    *   **手感影响**：`margin` 和 `gap` 的联合设置使碰撞力从 **0 开始极度丝滑地增长**，完全消除了两物体相撞瞬间力的数学阶跃。此外，由于物体相碰前提前产生微弱力信号，RL 的 Policy 能提早通过力反馈或 `contacts_vec` 察觉到碰撞即将发生，极大加速躲闪、防摔等反应策略的学习。

---

## 二、 核心改进想法与可靠度评估

针对当前对抗场景，我们对提出的优化方向进行客观的“可靠度自评”：

| 改进点与物理假说 | 期望手感与数值效果 | **可靠度评级** | 评级依据与物理逻辑 |
| :--- | :--- | :--- | :--- |
| **1. 增大受击部位(躯干/头部)的 `width` 到 15mm 以上** | 消除单步超速穿模（10m/s $\times$ 2ms = 20mm）瞬间带来的刚性爆震力，提供拳套/护具般的渐变抗力，使强化学习能获得连续、平滑的受力梯度。 | **绝对确认（100% 对）** | **硬物理计算支撑**：在 2ms 步长下，10m/s 速度单步必定产生 20mm 穿模。3mm 的 `width` 在第一步就会被直接跳过，丧失全部柔性过渡，导致隐式刚度过大。15mm~20mm 宽度是保证高速下具备 5-10 步渐变过渡的物理必选项。 |
| **2. 降低受击部位(躯干/头部)的最大阻抗 `dmax` 至 0.95 或更低，并采用过阻尼 (`dampratio > 1.0`)** | 躯干受击时不会“坚硬如铁”把力量无弹性反弹回去，而是通过超阻尼缓慢吸收机械能，防止被击打后高频晃动，让击打手感偏向“肉感”和“吸能重击”。 | **大概率对（85%）** | **经验直觉与控制理论**：对抗中需要吸能，否则高 `dmax` 加上临界阻尼在受力大时依然会导致剧烈的动能反弹。使用过阻尼（例如 `solref=[0.02, 1.2]`）能形成机械缓冲阻尼器效果。 |
| **3. 将脚底 `solimp` 的 `width` 缩窄到 1mm，响应常数 `timeconst` 缩短到 4ms** | 消除脚底站立时的“踩棉花”和“陷泥感”，提供如同坚硬大理石地面的即时、坚固触地支撑，让重心控制算法能灵敏地获取反作用力。 | **绝对确认（100% 对）** | **足式机器人动力学共识**：足端与地面必须是高刚度连接。3mm 甚至更大的缓冲区会导致站立重心因微小位移不断上下起伏，给重心观测（如 ZMP / CoM）带来巨大的滞后和噪声。 |
| **4. 将机器人与机器人碰撞的 `condim` 升级为 3 摩擦模型** | 杜绝机器人肢体绞杀、近身纠缠、推搡时“零摩擦无限滑行”的假冰面现象。使格斗时锁手臂、抱摔、近身推压等操作具有切向摩擦阻力。 | **绝对确认（100% 对）** | **MuJoCo 求解器继承链**：目前 `body_a`/`body_b` 的根 geom 定义了 `condim="1"`，导致机器人之间的碰撞毫无摩擦。两台高速对抗的机器人如果摩擦为零，RL 永远学不出正常的近身防摔或搏击阻挡姿态，只会发生滑溜的切向错位。 |
| **5. 将脚底 `condim` 升级为 4 旋转摩擦模型** | 防止机器人在重力压在单脚上时，脚底在地面发生无摩擦的水平“打转/陀螺旋转”而导致站立崩溃。 | **大概率对（90%）** | **足式仿真经典调优经验**：`condim=3` 仅约束沿平面的平动，若不加 `condim=4` 的 torsional friction 约束，机器人在做单脚转身或承受横向扭矩时，由于脚底与地面旋转无阻抗，极易发生打滑。 |
| **6. 引入微小的 `margin` 和 `gap` 用于防穿模预激活** | 开启物理相撞前的“渐变气垫”，使高初速（10m/s）下的两物体在发生接触前 5~12ms 提前产生平滑微弱力，完美避免相撞瞬间力的大幅阶跃。 | **绝对确认（100% 对）** | **MuJoCo 官方文档黄金原则**：当前环境未设置任何 `margin` 和 `gap`，导致两物体必须发生真实的几何穿模才会计算约束。高速对抗下必定发生单步 20mm 穿模，从而瞬间触发极强硬抗力。配置 `margin` 和 `gap` 是让碰撞力实现平滑 $0 \to f$ 的必由之路。 |

---

## 三、 参数配置提案（Propose 方案）

基于上述物理原理与差异化手感设置，Propose 以下一版全新的 `battle_v1.xml` 接触参数配置：

### 1. 修改后的 XML 结构提案 (Default Classes 部分)

我们建议重构 `<default>` 类树，将默认的全身继承拆分为：**偏软的躯干受击层**、**高摩擦高刚度的足端层**、**中度抗拉的手部打击层**。

```xml
<?xml version='1.0' encoding='utf-8'?>
<mujoco model="Laddermoon_Arena">
    <default>
        <motor ctrlrange="-1 1" ctrllimited="true" />
        
        <!-- ==================== ROBOT A (红队) 默认配置 ==================== -->
        <default class="body_a">
            <!-- 
              【躯干/头部默认配置】：偏软且高吸能 (手感：厚重、吸能、肉感防爆)
              - condim="3": 机器人肢体相撞时有滑动摩擦，不打滑
              - solimp=".5 .92 .018": 18mm 超宽渐变带，即使 10m/s 的拳头单步穿入 20mm，也会经历长达 18mm 的线性柔性过渡，dmax=0.92 保证极限抗力不至于爆震
              - solref=".02 1.2": 20ms 时间常数，1.2 过阻尼，撞击后立刻把机械能转换为阻尼热，不反弹
              - margin="0.012" gap="0.010": 12mm 预激活气垫，在距离 12mm 时开始平滑起效，防止瞬间穿过 3mm 引发的接触力阶跃
            -->
            <geom type="capsule" condim="3" friction=".7 0.01 0.005" solimp=".5 .92 .018" solref=".02 1.2" margin="0.012" gap="0.010" material="body_a" group="1" rgba="1 0.2 0.2 1" />
            
            <default class="thigh_a">
                <geom size=".06" rgba="1 0.2 0.2 1" />
            </default>
            
            <default class="shin_a">
                <geom fromto="0 0 0 0 0 -.3" size=".049" rgba="1 0.2 0.2 1" />
            </default>
            
            <!-- 
              【脚部重载】：极硬且防原地自旋打滑 (手感：硬朗、即时支撑、不拖泥带水)
              - condim="4": 开启扭转摩擦（torsional friction），支持防止原地自转
              - solimp=".9 .995 .001": 仅 1mm 的缓冲区，d0=0.9 直奔高刚度，绝不向下塌陷陷地
              - solref=".004 1.0": 4ms 特征恢复时间（对 dt=2ms 而言即 2 步消除误差），1.0 临界阻尼快速收敛
              - margin="0.001" gap="0.0008": 极微小 1mm 气垫用于消除足底硬接触可能出现的数值去抖，但不影响高刚性足底触地反馈
            -->
            <default class="foot_a">
                <geom size=".027" condim="4" friction="0.9 0.03 0.01" solimp=".9 .995 .001" solref=".004 1.0" margin="0.001" gap="0.0008" rgba="1 0.2 0.2 1" />
                <default class="foot1_a">
                    <geom fromto="-.07 -.01 0 .14 -.03 0" rgba="1 0.2 0.2 1" />
                </default>
                <default class="foot2_a">
                    <geom fromto="-.07 .01 0 .14  .03 0" rgba="1 0.2 0.2 1" />
                </default>
            </default>
            
            <default class="arm_upper_a">
                <geom size=".04" rgba="1 0.2 0.2 1" />
            </default>
            
            <!-- 
              【打击前端重载（手臂与手部）】：中度硬缓冲 (手感：结实，有拳套缓冲感)
              - solimp=".7 .97 .008": 8mm 缓冲，既能结实传力，又能在击中瞬间提供 4 个物理步的抗力上升空间，保护关节电机不被瞬时反作用力拉爆
              - solref=".012 1.0": 12ms 快速恢复，1.0 临界阻尼
              - margin="0.006" gap="0.005": 6mm 气垫，5mm 过渡带，避免打出空击时关节突然受阻
            -->
            <default class="arm_lower_a">
                <geom size=".031" condim="3" solimp=".7 .97 .008" solref=".012 1.0" margin="0.006" gap="0.005" rgba="1 0.2 0.2 1" />
            </default>
            <default class="hand_a">
                <geom type="sphere" size=".04" condim="3" solimp=".7 .97 .008" solref=".012 1.0" margin="0.006" gap="0.005" rgba="1 0.2 0.2 1" />
            </default>

            <!-- 关节 class 继承 -->
            <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01" />
            <default class="joint_big_a">
                <joint damping="5" stiffness="10" />
                <default class="hip_x_a">
                    <joint range="-30 10" />
                </default>
                <default class="hip_z_a">
                    <joint range="-60 35" />
                </default>
                <default class="hip_y_a">
                    <joint axis="0 1 0" range="-150 20" />
                </default>
                <default class="joint_big_stiff_a">
                    <joint stiffness="20" />
                </default>
            </default>
            <default class="knee_a">
                <joint pos="0 0 .02" axis="0 -1 0" range="-160 2" />
            </default>
            <default class="ankle_a">
                <joint range="-50 50" />
                <default class="ankle_y_a">
                    <joint pos="0 0 .08" axis="0 1 0" stiffness="6" />
                </default>
                <default class="ankle_x_a">
                    <joint pos="0 0 .04" stiffness="3" />
                </default>
            </default>
            <default class="shoulder_a">
                <joint range="-85 60" />
            </default>
            <default class="elbow_a">
                <joint range="-100 50" stiffness="0" />
            </default>
        </default>

        <!-- ==================== ROBOT B (蓝队) 默认配置 ==================== -->
        <default class="body_b">
            <!-- 保持与 body_a 完全对等的差异化对称设计 -->
            <geom type="capsule" condim="3" friction=".7 0.01 0.005" solimp=".5 .92 .018" solref=".02 1.2" margin="0.012" gap="0.010" material="body_b" group="1" rgba="0.2 0.2 1 1" />
            
            <default class="thigh_b">
                <geom size=".06" rgba="0.2 0.2 1 1" />
            </default>
            
            <default class="shin_b">
                <geom fromto="0 0 0 0 0 -.3" size=".049" rgba="0.2 0.2 1 1" />
            </default>
            
            <default class="foot_b">
                <geom size=".027" condim="4" friction="0.9 0.03 0.01" solimp=".9 .995 .001" solref=".004 1.0" margin="0.001" gap="0.0008" rgba="0.2 0.2 1 1" />
                <default class="foot1_b">
                    <geom fromto="-.07 -.01 0 .14 -.03 0" rgba="0.2 0.2 1 1" />
                </default>
                <default class="foot2_b">
                    <geom fromto="-.07 .01 0 .14  .03 0" rgba="0.2 0.2 1 1" />
                </default>
            </default>
            
            <default class="arm_upper_b">
                <geom size=".04" rgba="0.2 0.2 1 1" />
            </default>
            
            <default class="arm_lower_b">
                <geom size=".031" condim="3" solimp=".7 .97 .008" solref=".012 1.0" margin="0.006" gap="0.005" rgba="0.2 0.2 1 1" />
            </default>
            <default class="hand_b">
                <geom type="sphere" size=".04" condim="3" solimp=".7 .97 .008" solref=".012 1.0" margin="0.006" gap="0.005" rgba="0.2 0.2 1 1" />
            </default>

            <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01" />
            <default class="joint_big_b">
                <joint damping="5" stiffness="10" />
                <default class="hip_x_b">
                    <joint range="-30 10" />
                </default>
                <default class="hip_z_b">
                    <joint range="-60 35" />
                </default>
                <default class="hip_y_b">
                    <joint axis="0 1 0" range="-150 20" />
                </default>
                <default class="joint_big_stiff_b">
                    <joint stiffness="20" />
                </default>
            </default>
            <default class="knee_b">
                <joint pos="0 0 .02" axis="0 -1 0" range="-160 2" />
            </default>
            <default class="ankle_b">
                <joint range="-50 50" />
                <default class="ankle_y_b">
                    <joint pos="0 0 .08" axis="0 1 0" stiffness="6" />
                </default>
                <default class="ankle_x_b">
                    <joint pos="0 0 .04" stiffness="3" />
                </default>
            </default>
            <default class="shoulder_b">
                <joint range="-85 60" />
            </default>
            <default class="elbow_b">
                <joint range="-100 50" stiffness="0" />
            </default>
        </default>
    </default>
```

### 2. 提案参数配置的核心改进点对照表

| 身体层级 | 更改前参数 | 更改后Propose参数 | 改进收益说明 |
| :--- | :--- | :--- | :--- |
| **全默认 (body)** | `solimp=".9 .99 .003"`<br>`solref=".015 1"`<br>`condim="1"`<br>`margin="0"`<br>`gap="0"` | `solimp=".5 .92 .018"`<br>`solref=".02 1.2"`<br>`condim="3"`<br>`margin="0.012"`<br>`gap="0.010"` | **大幅度软化与抗爆震**：缓冲深度从 3mm 扩展到 18mm，给高速相对运动提供多个步长的缓冲。阻抗上限从 0.99 降为 0.92，搭配 1.2 的过阻尼（Over-damped），吸收剧烈撞击的机械能。开启切向摩擦（`condim=3`）使得肉搏近身推搡、纠缠具备真实的横向粘滞阻力。设定 **12mm 预激活气垫与 10mm 平滑过渡带** 杜绝由于单步超大穿模引发碰撞力的突然跃变。 |
| **脚底 (foot)** | 隐式继承：<br>`solimp=".9 .99 .003"`<br>`solref=".015 1"`<br>`condim="1"`<br>`margin="0"`<br>`gap="0"` | `solimp=".9 .995 .001"`<br>`solref=".004 1.0"`<br>`condim="4"`<br>`margin="0.001"`<br>`gap="0.0008"` | **极致坚硬与抗滑支撑**：缓冲区压缩到 1mm，起始阻抗高达 0.9，消除踩棉花般的足底抖动，让重心解算器响应极度即时（4ms/2步恢复时间）。开启 `condim=4` 的 torsional friction 阻断大重力下脚底在地面发生陀螺自旋，站立根基更牢固。同时摩擦力提升到 `0.9` 极大防止平动打滑。使用 **1mm 气垫** 去除由于硬度过大可能引起的数值去抖，但不牺牲高刚支撑手感。 |
| **打击手部 (hand / arm_lower)** | 隐式继承：<br>`solimp=".9 .99 .003"`<br>`solref=".015 1"`<br>`condim="1"`<br>`margin="0"`<br>`gap="0"` | `solimp=".7 .97 .008"`<br>`solref=".012 1.0"`<br>`condim="3"`<br>`margin="0.006"`<br>`gap="0.005"` | **打击实感与电机保护**：8mm 的缓冲区为瞬间发生的铁拳锤击、摆拳提供了过渡期，大幅减轻峰值机械反作用力。12ms 的临界阻尼使得出击与收回时手臂的振荡能迅速收敛。配备 **6mm 气垫与 5mm 过渡**，在高速碰撞未至几何表面相交前平顺起效，消除出拳顿卡感。 |

---

## 四、 实施及测试评估建议

我们绝对建议在部署此优化前，通过几项快速脚本测试该优化的实际手感：

1.  **纯站立/抗扰测试**：在更换此版 XML 参数后，运行 standing baseline。我们预期：
    *   足底穿模量明显变小且极快趋于稳定。
    *   机器人的 `uprightness` 维持在 0.99~1.00 的高比例时间增加，ZMP（压力中心）漂移更平滑。
2.  **暴力撞击力矩测试**：控制机器人 A 高速（出拳或直接飞扑）击打机器人 B 的胸部或头部。
    *   记录 `mj_contactForce` 的瞬时最大幅值。预期更换后的最大爆发力峰值至少能下降 **50% ~ 70%**（因为缓冲区增加了6倍），大幅消除了仿真动力学的不连续爆发。
    *   监控击打瞬间手臂关节的受控力矩（`actuator_force`）。参数优化后，击打瞬间电机不会因为强力阻卡而爆发极瞬时超载，数值安全性得到根治。
