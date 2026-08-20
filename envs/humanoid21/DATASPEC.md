# Humanoid21 数据规范 (Data Specification)

## 1. 核心理念 (Core Principles)
- **按主体隔离**: 策略层绝不能获得包含双机器人的混合数据（如全局 `qpos`）。所有方法必须返回 `Dict[str, np.ndarray]`，并在外层按 `robot_a` 和 `robot_b` 区分。
- **局部坐标系优先**: 除非必要（如朝向、高度），否则机器人的速度、角速度及对手的相对位置，一律转换到以自身 `Torso` 为原点的局部坐标系下。
- **全局归一化**: 具有物理限位的观测特征（位置、速度等）默认映射到 `[-1, 1]` 的无量纲区间。
- **Sandbox 闭包**: 插件/观察者只能通过 `ctx.accessor`（读）与 `ctx.mutator`（写，仅在允许钩子）访问数据。这两个代理严格白名单暴露 `IDataAccessor` / `IDataMutator` 的方法；backend 特有字段（如 MuJoCo 的 `model` / `data` / `_robot_cache`）**不可达**。若观察者需要某项物理量，请先在本规范中登记，再在 `Humanoid21Simulator.get_static_data()` / `get_derived_state()` 中填入数据。这样新增 backend 时替换实现即可，观察者无需改动。

---

## 2. 静态属性 (Static Properties)
**接口**: `get_static_data()`
- **定义**: 描述仿真器与各机器人的固定参数集。不会在 episode 内变化，应当在 `on_pre_episode` / 构造时一次性读取。
- **结构**:

### 2.1 按机器人分离的字段 (`result['robot_a']` / `result['robot_b']`)

| 键 | 类型 | 说明 |
|---|---|---|
| `dof_names` | `List[str]`，长度 21 | 受控自由度短名（不带 `_red`/`_blue` 后缀） |
| `body_names` | `List[str]` | 机器人躯干子树下**全部** body 的全名（含后缀），按 body id 稳定排序。观察者若要遍历 body，必须以此顺序为准。 |
| `body_masses_by_name` | `Dict[str, float]` | `body_names` 每个名字对应的 body 质量 (kg)。可直接用于 CoM 加权。 |
| `joint_names` | `List[str]` | 机器人子树下**全部** joint 的全名（含根部 freejoint、受控 21 dof、踝关节 2-DoF 等）。顺序与 body 列表一致的稳定性。 |
| `controlled_joint_names` | `List[str]` | 21 个受控 joint 的全名（带后缀）；`dof_names` 的带后缀版本。 |
| `root_joint_name` | `str` | 根 freejoint 的全名，例如 `root_red`。 |
| `keypoint_body_names` | `Dict[str, str]` | 语义角色 → body 全名。键集合目前为 `torso`/`head`/`pelvis`/`foot_left`/`foot_right`/`hand_left`/`hand_right`。观察者应用语义键而不是字符串拼接。 |
| `keypoint_joint_names` | `Dict[str, str]` | 语义角色 → joint 全名。当前仅覆盖踝关节 4 项：`ankle_x_left`/`ankle_x_right`/`ankle_y_left`/`ankle_y_right`。 |
| `joint_limits` | `ndarray`，`shape=(21, 2)` | 受控关节物理限位 `[min, max]` (rad)。与 `dof_names` / `controlled_joint_names` 对齐。 |

### 2.2 全局字段 (`result['dt']` / `result['ground_geom_name']`)

| 键 | 类型 | 说明 |
|---|---|---|
| `dt` | `float` | 单个物理子步仿真时长 (s)。等价于 `Humanoid21Simulator.DT`。 |
| `ground_geom_name` | `str` | 地面 geom 名（`"ground"`）。用于在 `derived_state['contacts_vec']` 中筛选机器人–地面接触（CAT_ENV_GROUND），避免硬编码字符串。 |

### 2.3 设计理由

前身 schema 仅提供 `dof_names` / `body_names` / `joint_limits` 三项，不足以支撑 body 级计算（质心、支撑力、关节锚点）。当观察者需要这类量时只能穿透到 `simulator.model`/`_robot_cache` —— 这违反 `IDataAccessor` 封装。本规范的扩展使观察者完全不需要 backend 句柄：任何 body 或 joint 都可以"按名字检索"。

新加入字段 invariant：

- **名字长度守恒**：`len(body_names) == len(body_masses_by_name)`；`len(joint_names) == len(derived_state[agent]['joint_world_anchor'])`。
- **名字对齐**：`body_names` 中的字符串必须同时是 `derived_state[agent]['body_xpos']` 等 per-body 字典的键。
- **不可变**：全部字段在一个 simulator 实例的生命周期内不变。

---

## 3. 核心状态 (Core State)
**接口**: `get_core_state()`
- **定义**: 能唯一决定单体机器人在空间中"姿态与本体运动"的最小数据集。基准为物理模型的根节点（带有 `freejoint` 的 Torso）。
- **结构**:
  - **`root_pos`** (3,): Torso 的绝对世界坐标 `(x, y, z)`。
  - **`root_rot`** (4,): Torso 的绝对姿态四元数 `[w, x, y, z]`。
  - **`root_vel_local`** (3,): Torso 在**自身局部坐标系**下的线速度。
  - **`root_angular_vel_local`** (3,): Torso 在**自身局部坐标系**下的角速度。
  - **`joint_pos_norm`** (21,): **归一化关节位置** `[-1, 1]`。
    - 计算公式: `(qpos - reference) / scale`
    - `reference` 为关节上下限的中间值，`scale` 为关节总行程的 1/2。
  - **`joint_vel_norm`** (21,): **归一化关节角速度**。
    - 计算公式: `qvel / scale`
    - 物理含义: 当前速度每秒能跨越的"半量程"数。它是 `joint_pos_norm` 对时间的精确导数。

**模块一：本体感知 (42维)**

| 数据项 | 维度 | 说明 |
|---|---|---|
| 归一化关节角度 | 21 | `joint_pos_norm` |
| 归一化关节角速度 | 21 | `joint_vel_norm` |

---

## 4. 派生数据 (Derived State)
**接口**: `get_derived_state()`
- **定义**: 面向机器学习特征工程、碰撞检测和奖励计算的丰富感知数据。**每个物理子步之后都会被刷新**，因此可以反映当前瞬时状态。
- **结构**: 包含全局对抗信息、全局向量化接触数据 (contacts_vec)、per-agent 高层视角、per-agent 低层物理量。

### 4.1 全局对抗信息 (Shared / Global)
放置在字典的最外层，供环境或中心化评论家(Critic)使用：
- **`torso_distance`** (ndarray shape=(1,)): 双方 Torso 之间的欧氏距离。
- **`contacts_vec`** (Dict): 向量化 SoA (Struct of Arrays) 接触数据，基于 MuJoCo 原生 ID + AFF 阵营分类。

#### 4.1.1 接触分类模型 (MuJoCo 原生 ID + AFF)

接触数据使用 MuJoCo 原生 geom/body ID 作为实体标识，辅以 AFF 阵营码进行分类：

| 层级 | 名称 | 取值 | 说明 |
|------|------|------|------|
| AFF | Affiliation | 0=Env, 1=robot_a, 2=robot_b | 共 3 类，标识 geom/body 所属阵营 |
| geom ID | MuJoCo Geom ID | 原生 int | 通过 `mujoco.mj_id2name(model, mjOBJ_GEOM, id)` 查名称 |
| body ID | MuJoCo Body ID | 原生 int | 通过 `mujoco.mj_id2name(model, mjOBJ_BODY, id)` 查名称 |

**静态查找表**（由 `Humanoid21Meta.build_runtime_tables()` 构建，在 `get_static_data()` 中暴露）：

| 表 | 类型 | 说明 |
|---|---|---|
| `geom_id_to_name` | `Dict[int, str]` | MuJoCo geom ID → geom 名称 |
| `body_id_to_name` | `Dict[int, str]` | MuJoCo body ID → body 名称 |
| `body_id_to_aff` | `Dict[int, int]` | body ID → 0(env)/1(robot_a)/2(robot_b) |
| `geom_id_to_aff` | `Dict[int, int]` | geom ID → 0(env)/1(robot_a)/2(robot_b) |

#### 4.1.2 `contacts_vec` 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `ncon` | `int` | 接触数量 |
| `geom1` / `geom2` | `ndarray(ncon,) int32` | MuJoCo 原生 geom ID |
| `body1` / `body2` | `ndarray(ncon,) int32` | MuJoCo 原生 body ID (= `model.geom_bodyid[geom]`) |
| `aff1` / `aff2` | `ndarray(ncon,) int8` | AFF 阵营 (0=env, 1=robot_a, 2=robot_b) |
| `force_mag` | `ndarray(ncon,) float32` | 接触力标量 (N) |
| `force_world` | `ndarray(ncon, 3) float32` | 世界坐标系 3D 力 (geom1 对 geom2 施加的力) |
| `position` | `ndarray(ncon, 3) float32` | 接触点世界坐标 |
| `normal` | `ndarray(ncon, 3) float32` | 接触法向量 (指向 geom2) |
| `frame` | `ndarray(ncon, 3, 3) float32` | 接触坐标系 [n; t1; t2] (行存储) |

**符号约定**：`force_world` 遵循 MuJoCo 约定——`geom1` 对 `geom2` 施加的力。对 `geom1` 的反作用力为取反。

**示例**（脚与地面接触合力，使用 contacts_vec）：

```python
import numpy as np

cv = derived_state['contacts_vec']
AFF_ENV = 0

# ground_geom_id 和 foot_body_id 从 static_data 获取
ground = static_data['ground_geom_id']
foot_l = static_data[agent]['keypoint_body_ids']['foot_left']
foot_r = static_data[agent]['keypoint_body_ids']['foot_right']

if cv['ncon'] > 0:
    # foot is geom2, env is geom1: force on foot
    m1 = (cv['geom1'] == ground) & (cv['body2'] == foot_l)
    # foot is geom1, env is geom2: force on env, negate
    m2 = (cv['geom2'] == ground) & (cv['body1'] == foot_l)

    support = np.zeros(3)
    if m1.any():
        support += cv['force_world'][m1].sum(axis=0)
    if m2.any():
        support -= cv['force_world'][m2].sum(axis=0)
```

### 4.2 单边视角信息 (Per-Robot Views)
分别放置在 `robot_a` 和 `robot_b` 的键下，供策略网络感知博弈态势。

#### 4.2.1 模块二：全局状态 (10维)
**接口**: `robot_view['root_state']`

| 数据项 | 维度 | 坐标系 | 说明 |
|---|---|---|---|
| `height` | 1 | 世界 | Z轴高度，判断是否倒地 |
| `projected_gravity` | 3 | 机体 | 重力方向单位向量 `-R[2, :]`，站直时为 `(0,0,-1)`；yaw 不变 |
| `linear_vel` | 3 | 机体 | 机体系线速度（由 `qvel[0:3]` 世界系值左乘 `R^T` 得到） |
| `angular_vel` | 3 | 机体 | 机体系角速度（MuJoCo free joint `qvel[3:6]` 本就是机体系，直接取用） |

> 旧版的 `local_orientation`（6维，机体轴在世界系下的表示）已被 `projected_gravity` 取代。
> 前者包含绝对 yaw，在缺少 X/Y 坐标时属于不可用的冗余维度；后者绕重力轴旋转不变。
> 标量 `uprightness` 仍保留，其值等于 `-projected_gravity[2]`。

#### 4.2.2 模块三：触觉力反馈 (2维)
**接口**: `robot_view['feet_forces']`

| 数据项 | 维度 | 说明 |
|---|---|---|
| `feet_forces` | 2 | 左脚、右脚与地面的接触受力大小 |

#### 4.2.3 模块四：对手观测 (39维)
**原则**: "以我为中心" + "动静结合"
**坐标系**: 所有对手坐标均转换至自身（Ego）的局部坐标系下。

##### 4.2.3.1 对手基础位姿 (9维)
**接口**: `robot_view['opponent_basic_pose']`

| 数据项 | 维度 | 物理含义 |
|---|---|---|
| `relative_pos` | 3 | 对手根关节 - 自身根关节（局部坐标系） |
| `relative_vel` | 3 | 相对速度（局部坐标系） |
| `face_vector` | 3 | 对手朝向的单位向量在Ego坐标系中，与自身同向为 (1,0,0) |

##### 4.2.3.2 对手关键点位置 (15维)
**接口**: `robot_view['opponent_keypoint_pos']`

| 数据项 | 维度 | 作用 |
|---|---|---|
| `head` | 3 | 头部中心点 |
| `hand_right` | 3 | 右手中心点 |
| `hand_left` | 3 | 左手中心点 |
| `foot_right` | 3 | 右脚中心点 |
| `foot_left` | 3 | 左脚中心点 |

##### 4.2.3.3 对手关键点速度 (15维)
**接口**: `robot_view['opponent_keypoint_vel']`

| 数据项 | 维度 |
|---|---|
| `head` | 3 |
| `hand_right` | 3 |
| `hand_left` | 3 |
| `foot_right` | 3 |
| `foot_left` | 3 |

### 4.3 单边低层物理量 (Per-Agent Low-Level Physics Arrays)

面向需要每个 body / 每个 joint 物理量的观察者（质心、支撑力、踝锚点、接触分析等）。**键均以 body / joint 全名为索引**，并与 §2 的 `body_names` / `joint_names` 完全对齐。值均为 `float32` ndarray 且是 simulator 内部缓冲区的**拷贝**，观察者可安全保留引用。

接口: `get_derived_state()[robot_id][<字段>]`

| 字段 | 类型 | 含义 |
|---|---|---|
| `body_xpos` | `Dict[str, ndarray(3,)]` | body 坐标系原点的世界位置 (m) |
| `body_xipos` | `Dict[str, ndarray(3,)]` | body 惯性中心的世界位置 (m)；**用于质心计算** |
| `body_xquat` | `Dict[str, ndarray(4,)]` | body 姿态四元数 `[w, x, y, z]` |
| `body_linvel_world` | `Dict[str, ndarray(3,)]` | body 在世界系下的瞬时线速度 (m/s)，等价于 MuJoCo `data.cvel[body, 3:6]` |
| `body_angvel_world` | `Dict[str, ndarray(3,)]` | body 在世界系下的瞬时角速度 (rad/s)，等价于 `data.cvel[body, 0:3]` |
| `joint_world_anchor` | `Dict[str, ndarray(3,)]` | 每个关节铰链锚点的世界坐标 (m)。对 `freejoint` 没有几何意义（但仍提供）。 |

**设计理由**:
- 在此层直接拷贝 MuJoCo `data.xpos` / `xipos` / `xquat` / `cvel` / `xanchor`，而不是仅通过高层观测维度压缩，是为了让 backend 无关的观察者能做任意线性代数计算而无需回到 `simulator.data`。
- 使用 **名字**而非 id 做键，避免将 MuJoCo 的 body/joint id 语义泄漏到上层。
- 观察者若要遍历所有 body，请使用 `static_data[robot_id]['body_names']` 的顺序以获得稳定排序（按 body id 排序）。

**示例（去脚加权质心）**:

```python
static = accessor.get_static_data()[agent]
derived = accessor.get_derived_state()[agent]
foot_l = static['keypoint_body_names']['foot_left']
foot_r = static['keypoint_body_names']['foot_right']
bodies = [n for n in static['body_names'] if n not in (foot_l, foot_r)]
m = np.array([static['body_masses_by_name'][n] for n in bodies])
p = np.array([derived['body_xipos'][n] for n in bodies])
com = (p * m[:, None]).sum(0) / m.sum()
```

---

## 5. 观测空间总结 (Observation Space Summary)

**总维度**: 93 维（每个机器人）

| 模块 | 维度 | 接口 | 说明 |
|------|------|------|------|
| 模块一：本体感知 | 42 | `get_core_state()[robot_id]['joint_pos_norm']`<br>`get_core_state()[robot_id]['joint_vel_norm']` | 关节角度和角速度 |
| 模块二：全局状态 | 10 | `get_derived_state()[robot_id]['root_state']` | 高度、重力投影、速度 |
| 模块三：触觉力反馈 | 2 | `get_derived_state()[robot_id]['feet_forces']` | 足底受力 |
| 模块四：对手观测 | 39 | `get_derived_state()[robot_id]['opponent_basic_pose']`<br>`get_derived_state()[robot_id]['opponent_keypoint_pos']`<br>`get_derived_state()[robot_id]['opponent_keypoint_vel']` | 对手位姿、关键点 |
| **完整观测** | **93** | `get_derived_state()[robot_id]['observation']` | 所有模块平铺后的完整观测 |

**平铺向量布局**:

| 索引 | 长度 | 内容 | 坐标系 |
|---|---|---|---|
| `[0:21]` | 21 | 归一化关节角度 | — |
| `[21:42]` | 21 | 归一化关节角速度 | — |
| `[42:45]` | 3 | `projected_gravity` | 机体 |
| `[45:46]` | 1 | `height` | 世界 |
| `[46:49]` | 3 | `linear_vel` | 机体 |
| `[49:52]` | 3 | `angular_vel` | 机体 |
| `[52:54]` | 2 | `feet_forces` | 标量 |
| `[54:93]` | 39 | 对手观测 | 机体 |

除 `height` 和 `feet_forces` 外，所有向量均在机体系，因此整个观测对世界系的平移与 yaw 旋转不变。

**完整观测获取**:

完整 93 维观测直接包含在 `get_derived_state()[robot_id]['observation']` 中：

```python
derived_state = sim.get_derived_state()

# robot_a 完整观测 (93维) - 直接获取
robot_a_obs = derived_state['robot_a']['observation']  # 93维，包含所有模块
```

如果需要单独访问各模块的数据：

```python
derived_state = sim.get_derived_state()

# 模块一：本体感知 (42维) - 需要从 get_core_state 获取
core_state = sim.get_core_state()
joint_pos_norm = core_state['robot_a']['joint_pos_norm']  # 21维
joint_vel_norm = core_state['robot_a']['joint_vel_norm']  # 21维

# 模块二：全局状态 (13维)
root_state = derived_state['robot_a']['root_state']

# 模块三：触觉力反馈 (2维)
feet_forces = derived_state['robot_a']['feet_forces']

# 模块四：对手观测 (39维)
opponent_basic = derived_state['robot_a']['opponent_basic_pose']  # 9维
opponent_keypoint_pos = derived_state['robot_a']['opponent_keypoint_pos']  # 15维
opponent_keypoint_vel = derived_state['robot_a']['opponent_keypoint_vel']  # 15维
```

**数据结构层级**:
```
get_derived_state()
├── torso_distance (全局)
├── contacts_vec (全局, MuJoCo 原生 ID + AFF 阵营分类, SoA 向量化)
└── robot_a / robot_b
    ├── root_state (模块二: 10维)
    │   ├── height              (世界系)
    │   ├── projected_gravity   (机体系)
    │   ├── linear_vel          (机体系)
    │   └── angular_vel         (机体系)
    ├── feet_forces (模块三: 2维)
    ├── opponent_basic_pose (模块四.1: 9维)
    ├── opponent_keypoint_pos (模块四.2: 15维)
    ├── opponent_keypoint_vel (模块四.3: 15维)
    ├── observation (93维平铺: 模块一+二+三+四)
    ├── uprightness (兼容旧版)
    └── opponent_in_local (兼容旧版)
```
