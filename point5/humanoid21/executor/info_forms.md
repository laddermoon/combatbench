# 信息形态与互转

执行层里流动着三种信息形态。**Pose 是枢纽**——一切转换都经过它，
不要发明旁路。

## 三种形态

| 形态 | 载体 | 内容 | 性质 |
|------|------|------|------|
| **Observation** | `obs (96,)` | 模拟器感知：归一化关节角/速度、root 高度+重力向量+速度、实测接触力 | 事实（带归一化/压缩编码） |
| **Pose** | `Pose` | 精确无位置姿态：`qpos(21,)` + `root_z` + `g_body` + 可选速度（qvel/linvel/angvel）；**不携带世界 xy/yaw** | 精确状态节点 |
| **P5** | `P5.pos (5,3)` | 几何意图语言：五点关键点，局部帧（踝中点原点、躯干 +x、重力 -z） | 意图/形状（无速度、无位置） |
| **SupportLandMark** | `SupportLandMark` | 支撑语义：`com(3,)` + 双脚 `FootSupport(pos, force_ratio)` | 解释态（姿态→支撑结论） |

> SupportLandMark 是第四种"投影"形态——从 Pose/P5 推出的支撑结论，
> 不是独立数据源。

## 互转矩阵

```
from\to          P5              Pose              SupportLandMark
P5                                pose.from_p5      经过Pose转
                                    (solver反解,近似)  (近似)
Pose             精确 p.p5()       ——                精确 lm
Observation      经过Pose转(精确)   pose.from_obs     经过Pose转(精确)
```

### 边语义

| 边 | API | 精度 | 说明 |
|----|-----|------|------|
| obs → Pose | `Pose.from_obs(obs)` | 精确 | 反归一化关节角 + sqrt 解压速度；com_vel/ang_mom 经 `mj_subtreeVel` 精确 |
| obs → P5 | `Pose.from_obs(obs).p5()` | 精确 | 过 Pose，不要另写解码器 |
| obs → LM | `Pose.from_obs(obs).support_state_on_p5_axis()` + 力度覆盖 | 几何精确 / **力度 Pose 给不出** | **f12 的实现**：几何经 Pose 精确，但脚接触力在 Pose 里**没有**（动力学量非运动学量）——必须再用 obs[52:54] 实测值覆盖 |
| P5 → Pose | `Pose.from_p5(p5)` | 近似 | `solve_equilibrium` 反解关节角——**不可逆**，意图→姿态有损 |
| P5 → LM | `Pose.from_p5(p5).support_landmark_on_p5_axis()` | 近似 | 过 Pose，经 solver 近似 |
| Pose → P5 | `pose.p5()` | 精确 | 自身局部帧 FK 直读 |
| Pose → LM | `pose.support_landmark_on_p5_axis()` | 精确 | 质量加权 com + 落地判带均分力度（规划态分配） |

## 关键约定

- **位置是 anchor 的事**：Pose 无位置。要世界/局部坐标先 `pose.anchor(point, xy, yaw)`；
  `p5()`/`com_height()` 不需要锚定，`com()`/`foot_pos()`/`com_vel()`/`ang_mom()` 要。
- **速度只在 Pose**：P5 纯几何。动量来自 obs 速度字段，规划态 Pose（from_p5）调 com_vel 会拒。
- **⚠️ obs → Pose 丢接触力**：Pose 只还运动学（qpos+速度），脚接触力是动力学量（还依赖当前 ctrl/执行器力矩——obs 里没有动作字段）。所以 `obs→Pose→LM` 几何精确、力度**必须用 obs 实测值补回**，不能指望 Pose 重建（准静态下 mj_forward 重建≈0.44 vs 实测 0.445，动态态偏差大）。
- **force_ratio 两种来源**：实测（obs[52:54]，f12 覆盖后返回）vs 规划（落地判带均分，
  `support_landmark_on_p5_axis` 默认）——当前态是测量，未来态是设计。
- **不可逆链**：P5→Pose 有损（solver），所以执行层规划从 obs 侧（精确）出发，
  P5 侧只当意图输入。

## 执行层消费点

| 函数 | 转换 | 输出 |
|------|------|------|
| `f1(obs)` | obs → Pose → LM + 实测力度覆盖 | `I1.lm` |
| `f2(obs)` | obs → Pose → subtree 动量 | `I2.com_vel/ang_mom` |
| `f4`（规划） | 需要时 P5 → Pose 校验可达 | clip |
