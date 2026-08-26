# 击攻势能方案 (Strike Potential Proposal)

## 1. 目标

定义一个密集、始终可追求的标量信号 `strike_potential ∈ [0, 1]`，衡量机器人当前状态的进攻潜力。它不依赖"打中"这个稀疏事件——好的进攻姿态本身就得分。

预期引导效果：

- 远距离 → 接近对手
- 近距离但背对 → 转身面对
- 面对但手收回 → 蓄力 / 出拳
- 高速挥击朝向对手 → 势能最高
- 打中后手速归零 → 势能下降，需收回重蓄

## 2. 势能分解

```
strike_potential = 0.5 × approach_score + 0.5 × strike_capability
```

- `approach_score`：是否在有效攻击位置且朝向正确（位置 + 朝向）
- `strike_capability`：手能否打到对手 + 是否在发力（可达性 + 动力学）

两部分各占 0.5，使势能在"靠近并朝向"和"有能力出拳"两个维度上独立贡献。

## 3. approach_score（占 0.5）

### 3.1 距离分量

```python
dist = ||self_torso_xy - opp_torso_xy||   # 水面投影距离

dist_score = clip((D_FACE - dist) / (D_FACE - D_STRIKE), 0, 1)
```

| 距离 | dist_score |
|---|---|
| ≤ 0.7 m (D_STRIKE) | 1.0 |
| 0.7 ~ 1.5 m | 线性 ramp |
| ≥ 1.5 m (D_FACE) | 0.0 |

复用 `standup_fight` 中已有的 `D_FACE = 1.5`、`D_STRIKE = 0.7`。

### 3.2 朝向分量

```python
forward_xy = torso 前向单位向量（XY 投影）
to_opp_xy = (opp_torso_xy - self_torso_xy) 归一化

facing = max(0, dot(forward_xy, to_opp_xy))
```

正面朝向对手时 `facing = 1`，背对时 `facing = 0`。

### 3.3 合成

```python
approach_score = dist_score × facing
```

乘法关系：距离再近，背对也没有 approach_score。

## 4. strike_capability（占 0.5）

```python
strike_capability = reachability × strike_dynamics
```

### 4.1 reachability（可达性）

如果对手不动，当前手的位置能否够到对手的受击部位。

受击部位取对手 `head` 和 `torso`，计算每只手到最近受击部位的距离：

```python
targets = [opp_head_pos, opp_torso_pos]

for each hand (left, right):
    reach_dist = min(||hand_pos - target|| for target in targets)
    reach_score = clip((REACH_RADIUS - reach_dist) / REACH_RADIUS, 0, 1)

reachability = max(left_reach, right_reach)   # 任一手够到即可
```

| 手到目标距离 | reach_score |
|---|---|
| ≤ 0 (贴住) | 1.0 |
| 0 ~ REACH_RADIUS | 线性 |
| ≥ REACH_RADIUS | 0.0 |

`REACH_RADIUS` 初定 0.5 m，需实测标定。

### 4.2 strike_dynamics（发力动力学）

手朝对手受击部位运动的进攻性，区分"挥击"和"蓄力"两种高势能状态。

```python
for each hand:
    hand_vel = body_linvel_world[hand]          # 3D 世界系线速度
    to_target = nearest_target_pos - hand_pos
    to_target_hat = to_target / ||to_target||
    approach_speed = dot(hand_vel, to_target_hat)   # 正=朝目标挥击

    speed_score = clip(approach_speed / SPEED_FULL, -1, 1)   # [-1, 1]

    if reach_dist < CHARGE_DIST:
        # 近距离蓄力：速度低也给保底分
        dynamics = max(speed_score, CHARGE_BASELINE)
    else:
        # 远距离：必须有正向速度才得分
        dynamics = max(speed_score, 0)

strike_dynamics = max(left_dynamics, right_dynamics)   # 任一手在发力即可
```

| 状态 | dynamics |
|---|---|
| 手远离目标，静止 | 0 |
| 手远离目标，朝目标高速运动 | → 1.0 |
| 手在目标附近，高速挥击 | → 1.0 |
| 手在目标附近，低速（蓄力） | CHARGE_BASELINE (0.5) |
| 手远离目标，收回（负速度） | 0（clip 掉负值） |

参数初定：

- `SPEED_FULL = 1.0 m/s`（满速阈值，需实测出拳速度）
- `CHARGE_DIST = 0.3 m`（蓄力判定距离）
- `CHARGE_BASELINE = 0.5`（蓄力保底分）

### 4.3 合成

```python
strike_capability = reachability × strike_dynamics
```

乘法关系：手够不到时 capability = 0，无论手速多快。

## 5. 完整公式

```
strike_potential = 0.5 × dist_score × facing
                 + 0.5 × reachability × strike_dynamics
```

## 6. 预期行为表

| 场景 | approach | capability | 势能 | 引导效果 |
|---|---|---|---|---|
| 远距离，朝向对 | 0.3 | 0 | 0.15 | 接近对手 |
| 近距离，朝向对，手收回 | 0.5 | 0.25（蓄力） | 0.375 | 可以出拳 |
| 近距离，朝向对，高速挥击 | 0.5 | 0.5 | 0.50 | 势能最高 |
| 打中后（手速≈0，手在目标处）| 0.5 | 0.25 | 0.375 | 势能下降，需收回重蓄 |
| 近距离，背对对手 | 0 | 0 | 0 | 转身面对 |
| 远距离，朝向不对 | 0 | 0 | 0 | 先转身再接近 |

## 7. 插件设计

### 7.1 文件

```
baseline/humanoid21/end2end/strike_potential_observer.py
```

### 7.2 接口

```python
class StrikePotentialObserver(BaseObserverPlugin):
    """每步输出击攻势能标量 ∈ [0, 1]。

    依赖 simulator 的 derived_state：
      - body_xipos[hand_left / hand_right]
      - body_xipos[opp_head / opp_torso]
      - body_linvel_world[hand_left / hand_right]
      - body_xquat[torso]  (朝向计算)

    输出:
      {"strike_potential": float}
    """

    # 可配置参数
    D_FACE: float = 1.5
    D_STRIKE: float = 0.7
    REACH_RADIUS: float = 0.5
    CHARGE_DIST: float = 0.3
    CHARGE_BASELINE: float = 0.5
    SPEED_FULL: float = 1.0

    def on_pre_episode(self, ctx): ...
    def on_post_action_step(self, ctx): ...
    def get_output(self) -> Dict[str, float]: ...
```

### 7.3 数据来源

| 数据 | 来源 | 说明 |
|---|---|---|
| self/opp torso XY | `body_xipos[torso]` | 水面投影 |
| torso forward | `body_xquat[torso]` | 旋转 [1,0,0] 后 XY 投影 |
| hand 位置 | `body_xipos[hand_left/right]` | 世界系质心 |
| hand 速度 | `body_linvel_world[hand_left/right]` | 世界系线速度 |
| opp head/torso 位置 | `body_xipos[opp_head/torso]` | 对手受击部位 |

所有数据均已在 `simulator._get_robot_derived_state` 中提供，无需修改 simulator。

## 8. 在实验中使用

### 8.1 作为新 reward channel

在 `standup_fight` 中新增 `r_strike_potential`：

| 属性 | 值 |
|---|---|
| reward | `strike_potential`（始终追求高） |
| γ | 0.99（长期信号，势能是持续目标） |
| actor_weight | `φ²`（倒地时不追求进攻势能） |

### 8.2 与现有 damage channel 的关系

- `r_strike_potential`：**密集引导**，始终塑造进攻姿态
- `r_damage_dealt`：**稀疏结果**，打中才得分，距离门控
- `r_damage_taken`：**稀疏惩罚**，被打才扣分，距离门控

势能引导机器人进入高进攻潜力状态，damage 在真正命中时给额外奖励。两者互补。

### 8.3 取分势能（bootstrip.md 第 35 行的设想）

bootstrip.md 提到"取分势能只有在攻击势能高时才起效"。当前 damage channel 的距离硬门控已经实现了类似效果（远距离时 aw=0）。如果后续需要更精细的耦合，可以将 damage 的 aw 从硬门控改为 `strike_potential²`，使伤害奖励与进攻势能二次方挂钩。

## 9. 待标定参数

| 参数 | 初值 | 标定方法 |
|---|---|---|
| REACH_RADIUS | 0.5 m | 机器人站立、手臂前伸时 hand 到 torso 的距离 |
| SPEED_FULL | 1.0 m/s | 出拳时手速实测 |
| CHARGE_DIST | 0.3 m | 蓄力时手到目标的距离 |
| CHARGE_BASELINE | 0.5 | 蓄力状态势能占比，可调 |

## 10. 风险与注意事项

1. **势能饱和**：如果 strike_dynamics 的 SPEED_FULL 设太低，机器人会发现"微动手"就能拿满分，不需要真正发力。需实测出拳速度后标定。
2. **蓄力漏洞**：CHARGE_BASELINE 过高时，机器人可能学会"把手贴在对手身上不动"拿保底分。CHARGE_DIST 需要配合 reachability 一起限制。
3. **对手移动归因**：approach_score 用绝对距离，对手后退时机器人不移动也能涨分。如果成为问题，可改为径向接近速度（类似 r_radial）。当前先用绝对距离，保持简单。
4. **双手取 max**：任一手满足即可，避免双手都必须同时发力。但也可能导致机器人只练一只手。可观察后决定是否改为 `left + right` 或加权。
