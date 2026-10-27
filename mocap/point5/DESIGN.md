# Point5 业务语言 v1 设计

> 类型：设计契约  
> 状态：方案待评审，未实现  
> 范围：`mocap/point5/` —— 5 点业务描述语言的 Simulator-Unaware 核心库

## 1. 定位

5 点模型是**业务层的动作意图表达**，不是机器人状态表示，也不是规划器。

- 人或战术模型用它描述"动作应该是什么样"（头/双手/双脚在哪）
- 领域模型（本库）负责把稀疏关键帧合成连续参考信号
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
    pos:  (5,3) float64      # 世界坐标
    vel:  (5,3) float64 | None
    time: float
    frame: str = "world"     # 标签，不解析
    metadata: dict
```

实测值（adapter 填）和参考值（本库产出）用同一类型，下游比较才对称。

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

### 4.3 `PointSpec`

```python
PointSpec:
    target:    Anchor           # 必填
    hold:      bool = False     # True → 本段内钉在 target 上不动
    tolerance: float = 0.05     # m，标注用
    weight:    float = 1.0      # 标注用，评分/训练层消费
    contact:   str | None       # 仅 "planted"|"swing"|None，标注用
```

- `hold` 是**段级语义**，挂在段左端 keyframe 的 spec 上：该段（kf_i→kf_{i+1}）
  内此点恒定等于 kf_i 的 resolved target，不受 kf_{i+1} 目标影响。
- `tolerance`/`weight`/`contact` 都是**元数据标注**，合成器不消费它们
  （contact 不自动贴地）。它们是给评分/校验/训练层读的。

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
辅助函数 `is_planted(pos, tol)`。不做枚举推断，z 在模糊区间的
输入由校验层 warning（v1 只做结构校验，不 warning——留给 F6）。

## 5. 参考合成（本库唯一的"计算"）

### 5.1 输入

```python
build_reference(task, initial: Point5State, resolver) -> Point5Reference
```

`initial` 是任务开始时的实测 5 点状态：静态 anchor 的解析基准，
也是第 0 段插值的左端来源（keyframe[0] 的 spec 仍然是硬约束——
initial 只用于 anchor 解析和诊断对比，不替代 kf0）。

### 5.2 插值策略

段 `i`（kf_i → kf_{i+1}，时长 d_i）内，对每个点做三次 Hermite 插值：

```
H(s) = h00(s)·p0 + h10(s)·d·m0 + h01(s)·p1 + h11(s)·d·m1
```

边界切向量规则（**不定义速度，内部自动**）：

| 位置 | 切向量 |
|---|---|
| 任务首端 (kf_0 左) | 0 |
| 任务末端 (kf_{K-1} 右) | 0 |
| 内部边界 kf_i | 割线 `(p_{i+1} - p_{i-1}) / (d_{i-1} + d_i)` |

后果：内部 keyframe 天然是**经过点**（带速通过），"击打点"不需要
显式速度；首尾是停止点。

`hold=True` 覆盖插值：该段 pos = 常数、vel = 0。

输出参考的 `vel` 是 Hermite 解析导数（派生量，不是规格）。

### 5.3 采样与出界

- `sample(seg, s)`：seg 越界或 s∉[0,1] → 报错
- `sample_time(t)`：t < 0 或 t > duration → 报错（**不静默钳制**）
- `frames(dt)`：dense 采样 `(T,5,3)` pos/vel + seg/s 标签

## 6. 校验（仅结构层，v1）

构造时 fail-loud（`TaskValidationError`）：

- 每个 keyframe 恰好覆盖 5 个已知点名
- `len(keyframes) >= 2`，`len(durations) == len-1`，durations 全 > 0
- tolerance > 0，weight >= 0，contact ∈ {planted, swing, None}
- anchor 类型已知、字段有限

不做：可达性包络、速度上限、模糊接触带——属于 F6 领域 lint，后续。

## 7. 序列化

`to_dict()` / `from_dict()`，JSON-safe（list 不是 ndarray）。
anchor 带 type 标签：`{"type":"world","pos":[...]}` 等。
这是 human authoring / 编辑工具的存储格式。

## 8. 明确不做（防止烂尾的清单）

- 点可以缺省 / 先验填充 / mask
- 速度规格、pass/reach 模式枚举（内部切向量自动生成即可）
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
├── trajectory.py    # §5 合成与采样
└── test_contracts.py / test_trajectory.py
```
