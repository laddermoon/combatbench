# CPose

姿态的机器人无关描述（规划 IR）。P5 给端点世界坐标，CPose 给端点身体局部向量。

| 字段 | 语义 |
|---|---|
| `pelvis_pos`, `pelvis_pry` | Pelvis 位姿（所在帧；Location，可省） |
| `torso_pry` | Torso 相对 Pelvis 的朝向 |
| `left/right_hand` | 手球心−肩旋转中心，torso 系向量 |
| `left/right_ankle` | 踝心−髋旋转中心，pelvis 系向量 |

- 局部系：X+ 前、Y+ 左、Z+ 上（右手系）
- `pry` = (pitch, roll, yaw)：意图通道（参考 R=Rz·Ry·Rx）；
  机械解释由机型适配器自定，躯干可含任意铰链
- 肢向量单位 = LMax，故 |v| = 伸展度 ∈ (0,1]
- 不表达：脚朝向、髋自旋、头部姿态
