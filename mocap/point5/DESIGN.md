# Point5 业务语言 v1 设计

> 类型：设计契约  
> 状态：v1 已按本文件实现  
> 范围：`mocap/point5/` —— 5 点业务描述语言的 Simulator-Unaware 核心库

## 1. 定位

5 点模型是**业务层的动作意图表达**，不是机器人状态表示，也不是规划器。

- 人或战术模型用它描述"动作应该是什么样"（头/双手/双脚在哪）
- 本库负责把稀疏关键帧合成连续参考信号（纯线性过渡）
- 下游（Harness / 训练 adapter）负责把参考变成评分、观测或控制——**不在本目录**

类比：建筑设计（本目录）→ 结构/机电设计（centroidal/全身细化）→ 施工（控制）。

## 2. 硬边界

```text
禁止 import: mujoco, envs.*, baseline.*, centroidal_dynamics, 任何 simulator
禁止产出:   action、关节目标、接触力、动量、全身状态
输入:      任务描述对象 / 实测 Point5State（由外部 adapter 构造）
输出:      按 (seg, s) 或 t 采样的 Point5State 参考信号
```

锚点解析通过 `Resolver` 协议注入；本库只提供 `StaticStartResolver`。

## 3. 词表

固定 5 个语义名，不可扩展：

```python
POINT_NAMES = ("head", "hand_right", "hand_left", "foot_right", "foot_left")
```

**每个 Keyframe 必须给全部 5 个点写 PointSpec。** 不存在"未指定点"语义——
v1 不做 free/mask/先验填充，"某点受不受约束"是 Harness 层的决定，不是语言的。

## 4. 数据类型

### 4.1 `Point5State`（实测与参考共用）

```python
Point5State:
    pos:  (5,3) float64        # 世界坐标
    time: float = 0.0
    vel:  (5,3) float64 | None # v1 参考恒为 None；实测 adapter 可填
    frame: str = "world"       # 标签，不解析
    metadata: dict
```

实测值（adapter 填）和参考值（本库产出）用同一类型，下游比较才对称。
v1 的轨迹**不含速度**：不存在起点速度、终点速度、插值速度。

### 4.2 锚点（全部静态）

```python
WorldPoint(p)              # 固定世界点
RootOffset(v)              # 相对"任务开始时 root 坐标系"的偏移（含朝向）
SelfPointOffset(point, v)  # 相对"任务开始时自己某个关键点"的世界系偏移
```

所有 anchor 在 `build` 时用 `Resolver.resolve()` 一次性解析成世界坐标点，
之后不再变化。v1 没有动态锚点（不跟踪对手，不跟踪运动中的自身）。

`Resolver` 是协议：`resolve(anchor) -> vec3`。库内实现
`StaticStartResolver(initial: Point5State, root_pos, root_rot)`。

`initial` 只用于 anchor 解析与诊断对比，不进入插值：
keyframe[0] 的 spec 是硬约束，`sample(seg0, 0)` 返回 kf0 的 target。

### 4.3 `PointSpec`

```python
PointSpec:
    target:    Anchor           # 必填
    tolerance: float = 0.05     # m，标注用
    weight:    float = 1.0      # 标注用，评分/训练层消费
```

`tolerance`/`weight` 是**元数据标注**，合成器不消费，留给评分/校验/训练层。

**没有 hold 字段。** 相邻 keyframe 写相同 target，线性插值天然产生静止段；
需要保持某点不动，就在该段两端写同一个 target。

### 4.4 `Keyframe` / `Point5Task`

```python
Keyframe:   specs: {name: PointSpec}   # 必须恰好覆盖 5 个点
            label: str = ""            # "prepare"/"strike"/"recover" 业务标签

Point5Task: name: str
            keyframes: tuple[Keyframe, ...]   # K >= 2
            durations: tuple[float, ...]      # len = K-1，全 > 0
            metadata: dict
    duration = sum(durations)
```

**时间模型：段时长 + 段内归一化。** keyframe 不带绝对时间；
采样用 `sample(seg, s∈[0,1])` 或 `sample_time(t)`（内部做 t→seg 映射）。
变速编辑 = 改 `durations`，不动几何。

### 4.5 接触约定

**脚的 pos z ≈ 0 即触地。** 约定常量 `GROUND_Z = 0.0` +
辅助函数 `is_planted(pos, tol)`。标注/判定语义留给下游。

## 5. 参考合成（本库唯一的"计算"）

### 5.1 输入

```python
resolve_task(task, resolver) -> ResolvedTask      # 所有 anchor → 世界点 (K,5,3)
Point5Reference(resolved, task)
```

### 5.2 插值

段 `i`（kf_i → kf_{i+1}）内逐点线性过渡：

```python
pos(seg, s) = (1 - s) * P[seg] + s * P[seg + 1]     # s ∈ [0,1]
```

无切向量、无速度项、无平滑核。几何上是 keyframe 序列的分段线性折线。

### 5.3 采样与出界

- `sample(seg, s)`：seg∉[0,K-2] 或 s∉[0,1] → `SampleOutOfRange`
- `sample_time(t)`：t < 0 或 t > duration → `SampleOutOfRange`（**不静默钳制**）
- `frames(dt)`：dense 采样，返回 pos `(T,5,3)` + t/seg/s 标签数组
- 参考 `Point5State.vel` 恒为 `None`；metadata 带 seg/s/t/label

## 6. 校验（仅结构层，v1）

构造时 fail-loud（`TaskValidationError`）：

- 每个 keyframe 恰好覆盖 5 个已知点名（不多不少）
- `len(keyframes) >= 2`，`len(durations) == K-1`，durations 全 > 0
- tolerance > 0，weight >= 0
- anchor 类型已知、字段为有限 (3,) 向量

不做：可达性包络、速度上限、模糊接触带——属于领域 lint，后续。

## 7. 序列化

`to_dict()` / `from_dict()`，JSON-safe（list 不是 ndarray）。
anchor 带 type 标签：`{"type":"world","pos":[...]}` 等。
这是 human authoring / 编辑工具的存储格式。

## 8. 明确不做（防止烂尾的清单）

- 点缺省 / 先验填充 / mask / "不受约束"语义
- 速度规格、速度输出、平滑插值（Hermite/样条/割线切向量）
- hold/reach/pass 模式枚举（等值 target 即静止段）
- 动态 anchor（对手绑定、运动中自身点绑定）
- 区域目标（球/盒），v1 只有 点+tolerance 标注
- 评分/奖励、观测特征、PPO 插件、MuJoCo 提取器（全是 adapter 的活）
- 可达性/速度合理性 lint
- mirror/time_scale/concat 等编辑运算（后续单独立项）

## 9. 文件

```text
mocap/point5/
├── DESIGN.md        # 本文
├── CONTEXT.md       # AI 上下文 memo
├── contracts.py     # §3-§4 + 校验 + 序列化 + Resolver 协议
├── trajectory.py    # §5 解析、插值与采样
└── test_contracts.py / test_trajectory.py
```
