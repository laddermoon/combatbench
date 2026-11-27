# Humanoid28 环境

基于 MimicKit humanoid 形态（28 受控自由度、50.05 kg、pelvis 根节点）的双机器人对抗环境。
实现方式、数据接口与契约风格完全参照 `humanoid21`，但控制与动力学语义对齐 MimicKit。

## 来源与许可 (Provenance & License)

机器人模型来源于 **MimicKit** 项目的
`data/assets/humanoid/humanoid.xml`（DeepMimic/AMP 谱系资产）：

- 上游项目: https://github.com/xbpeng/MimicKit
- 作者: Xue Bin Peng 及 contributors
- 许可证: **Apache License 2.0**（见本目录 `assets/LICENSE.mimickit`）
- 原始资产拷贝于: `assets/mimickit_humanoid.xml`（未经修改的单机器人 XML）

本环境对该资产做了以下**修改**（依据 Apache-2.0 第 4 条予以声明）：

1. 关节 `stiffness` / `damping` 属性被清零——在 MimicKit 语义中它们是
   **PD 控制增益**而非被动动力学参数（Newton engine 在 `_build_controls`
   中也做同样的剥离）。PD 增益值被提取到 `robot_actuation.json`。
2. geom `condim` 由 1 改为 3（Isaac Gym 中接触始终带摩擦，`condim=1`
   在 MuJoCo 中会得到无摩擦的机器人间接触，是资产移植假象而非上游语义）。
3. 复制为 `_a` / `_b` 双机器人实例并加入圆形竞技场（与 humanoid21 相同场地）。

`battle_humanoid28.xml` 与 `robot_actuation.json` **均由
`generate_arena_xml.py` 脚本生成，禁止手工编辑**；修改后重新运行脚本即可。

## 与 humanoid21 的关键差异

| 项 | humanoid21 | humanoid28 |
|---|---|---|
| 受控 DOF | 21 | **28**（多 neck xyz / shoulder z / ankle z） |
| 单机器人质量 | 40.84 kg | 50.05 kg |
| root body | torso | **pelvis** |
| 命名风格 | `hand_right` | `right_hand`（side-first） |
| action 语义 | `[-1,1]` 归一化 | **弧度制位置目标**（MimicKit 格式） |
| action bounds | 关节机械限位 | **MimicKit 扩展限位**（组内 1.2x / 单曲柄 1.4x） |
| 本体观测 | 归一化 + sqrt 压缩 | **原始弧度值，无压缩** |
| relative_vel | 历史遗留错误语义 | **修正契约**（v_opp − v_self 旋入 ego 系） |
| PD 增益 | 自研标定 | **MimicKit 原值**（kp=XML stiffness, kd=XML damping） |
| 扭矩上限 | 自研 | = motor gear（MimicKit） |
| 观测维度 | 96 | **110**（28+28+13+2+39） |
| batch framework | 实验性支持 | **不支持**（决策 #13） |

相同点：500 Hz 物理 / 20 Hz 决策（每 action step 25 物理子步）、圆形竞技场、
CombatScoringPlugin 伤害规则（当前参数为 humanoid21 占位值，**未标定**）、
plugin/observer/blueprint 框架结构。

## 文件结构

```
humanoid28/
├── bootstrip.md                    # 用户决策清单（14 条，本环境的权威需求）
├── generate_arena_xml.py           # XML 生成脚本（唯一事实来源）
├── battle_humanoid28.xml           # 【生成物】双机器人竞技场，禁止手改
├── robot_actuation.json            # 【生成物】per-joint kp/kd/gear/range 表
├── assets/
│   ├── mimickit_humanoid_upstream.xml  # 未修改的上游资产（对照用）
│   └── LICENSE.mimickit            # Apache-2.0 许可证文本
├── meta.py                         # Humanoid28Meta：joint/body/keypoint/actuation 元数据
├── simulator.py                    # Humanoid28Simulator（MuJoCo 后端 + MimicKit 语义 PD）
├── plugins.py                      # CombatScoringPlugin / FrozenRobotPlugin
├── observer_plugins.py             # CombatScoringObserver
├── blueprint.yaml                  # 标准比赛环境蓝图
├── CONTROLSPEC.md                  # 控制契约
├── DATASPEC.md                     # 数据接口契约
├── OBSERVATION_zh.md               # 110 维观测空间详解
└── tests/test_humanoid28_core.py   # 核心契约测试（24 项）
```

## 快速开始

```bash
cd /data1/mono/things/combatbench

# 测试
PYTHONPATH=. python3 -m pytest envs/humanoid28/tests/test_humanoid28_core.py -q

# 运行一个 round（随机策略）
PYTHONPATH=. python3 - <<'PY'
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
import numpy as np
text = open('envs/humanoid28/blueprint.yaml', encoding='utf-8').read()
rt = ParameterizedEnvBlueprint.from_yaml(text).materialize().build()
rt.reset(seed=42)
while not rt.is_episode_over():
    rt.step(np.zeros(28, np.float32), np.zeros(28, np.float32))
print(rt.get_agent_termination())
PY
```

注意：**action 全零不是"中立"**——弧度制位置目标下 `0` 表示全部关节回到
零位（T-pose 手臂侧平举）。要保持 reset 后的站立姿态，应使用
`sim.get_action()` 返回的当前目标。

## 修改 XML 的正确姿势

```bash
# 1. 编辑 generate_arena_xml.py（或上游资产/接触参数）
# 2. 重新生成
python3 envs/humanoid28/generate_arena_xml.py
# 3. 跑测试验证
PYTHONPATH=. python3 -m pytest envs/humanoid28/tests/test_humanoid28_core.py -q
```

## 未移植 / 后续工作

- `BalanceAnalysisObserver`（humanoid21 的平衡分析 observer）未移植——
  依赖踝关节语义适配（humanoid28 踝为 x/y/z 三自由度而非 x/y）。
- 战斗/伤害参数为 humanoid21 占位值（决策 #10），**未针对 50kg / 200Nm
  执行器标定**——高动态执行器可能使 KO 过快，正式训练前需重新校准
  `force_scale` / `MAX_DAMAGE_FORCE`。
- 被动站立不是稳定的——PD 保持姿态 ≠ 平衡控制，站立是策略的任务
  （与 humanoid21 一致）。
- MimicKit motion clip 兼容层（expmap → serial-hinge 转换）未实现；
  当前 init pose 的肩部 ±1.5708 已按"串联 hinge 顺序值"直接处理，
  与上游数值一致，但若未来加载 `.pkl` 动作数据需先做 spherical 解缠。
