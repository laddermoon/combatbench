# CombatBench 环境模块设计说明

## 目录结构

```
envs/
├── core/               # 核心引擎 - 与具体机器人无关
│   ├── base_robot.py   # 机器人抽象基类
│   └── physics.py      # MuJoCo 物理引擎封装
│
├── framework/          # 仿真框架 - 通用仿真基础设施
│   ├── open_simulator.py   # 仿真器抽象接口
│   ├── base_hook.py        # Hook 机制定义
│   ├── simrunner.py        # 仿真运行器
│   └── rl_env.py           # RL 环境基类
│
├── humanoid21/        # Humanoid21 机器人特定实现
│   ├── humanoid21.py       # Humanoid21 仿真器
│   ├── robot.py            # 21-DOF 人形机器人
│   ├── collision.py        # 碰撞检测
│   ├── scoring.py          # 血量计分
│   └── humanoid21_base_hook.py  # Humanoid21 特定 Hooks
│
├── preset_envs.py     # 预置 Gym 环境 (Humanoid21NonFallEnv, Humanoid21FallEnv)
├── combat_gym.py      # [OBSOLETE] 旧版环境，保留用于参考
└── __init__.py        # 导出所有公开接口
```

---

## 核心架构设计

### 设计原则

1. **分层解耦**: 框架层与具体机器人实现分离
2. **Hook 扩展**: 通过 Hook 机制实现灵活的功能扩展
3. **状态分离**: 核心状态(可写)与衍生状态(只读)分离

### 三层架构

```
┌─────────────────────────────────────────────────┐
│              Gym 环境层 (RL 层)                  │
│  SimpleCombatEnv → preset_envs.py               │
│  - Observation, Reward, Done                    │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│            仿真框架层 (Framework)                │
│  SimRunner + Hook 机制                          │
│  - 仿真循环控制                                  │
│  - Hook 调度管理                                 │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│            仿真器层 (Simulator)                  │
│  OpenSimulator 接口                             │
│  - 物理步进                                      │
│  - 状态读写                                      │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│            物理引擎层 (Physics)                   │
│  MuJoCo                                         │
└─────────────────────────────────────────────────┘
```

---

## 数据分类

### 1. 静态数据 (Static Data)
- 场景模型 (XML)
- 机器人配置 (PD 参数、关节限制等)
- **获取方式**: `simulator.get_static_data()`

### 2. 核心状态 (Core State) - 可读写
- 广义坐标 `q` (关节位置、基座位置/朝向)
- 广义速度 `q̇` (关节速度、基座线/角速度)
- **获取方式**: `f_get_core_state()`
- **修改方式**: `f_set_core_state()`

### 3. 衍生状态 (Derived State) - 只读
- 接触点、接触力
- 末端执行器位置、关键点位置
- 雅可比矩阵、惯性矩阵
- **获取方式**: `f_get_derived_state()`

### 4. 传感器数据 (Sensor Data) - 只读
- IMU (加速度、角速度)
- 触摸传感器
- 力传感器
- **获取方式**: `f_get_sensor_data()`

---

## Hook 机制

### Hook 调用时序

```
Episode:
  PRE_EPISODE     → 重置状态
  └─ Action Loop:
       PRE_ACTION_STEP  → 解析动作、控制模式
       └─ Physics Loop (N 次):
            PRE_PHY_STEP    → 施加扰动
            physical_step()
            POST_PHY_STEP   → 执行约束、采集视频帧
       POST_ACTION_STEP → 终止判定、观测构建、奖励计算
  POST_EPISODE    → 清理资源
```

### Hook 接口

```python
class BaseHook(ABC):
    @property
    def name(self) -> str: ...

    @property
    def priority(self) -> int: ...  # 数值越大越先执行

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_core_state: Callable = None,
        f_get_derived_state: Callable = None,
        f_get_sensor_data: Callable = None,
        f_set_core_state: Callable = None,
        **kwargs
    ) -> bool:  # 返回 True 表示终止 episode
```

### InvokeType 枚举

| 类型 | 时机 | 用途 |
|------|------|------|
| PRE_EPISODE | Episode 开始前 | 重置 Hook 内部状态 |
| POST_EPISODE | Episode 结束后 | 清理资源 |
| PRE_ACTION_STEP | 动作步开始前 | 解析动作、设置控制模式 |
| POST_ACTION_STEP | 动作步结束后 | 终止判定、观测构建、奖励计算 |
| PRE_PHY_STEP | 物理步前 | 施加扰动 |
| POST_PHY_STEP | 物理步后 | 执行约束、采集视频帧 |

---

## 核心接口

### OpenSimulator (仿真器抽象)

```python
class OpenSimulator(ABC):
    # 控制接口
    def set_action(action: Dict[str, np.ndarray]) -> None: ...
    def physical_step() -> None: ...

    # 数据获取
    def get_core_state() -> Dict[str, Any]: ...     # q, q̇
    def get_derived_state() -> Dict[str, Any]: ...  # 接触、运动学
    def get_sensor_data() -> Dict[str, Any]: ...    # IMU、触摸
    def get_static_data() -> Dict[str, Any]: ...    # 模型、配置

    # 状态修改
    def set_core_state(state: Dict[str, Any]) -> None: ...

    # 视频渲染
    def get_broadcastview_image() -> np.ndarray: ...
```

### BaseRobot (机器人抽象)

```python
class BaseRobot(ABC):
    # 动作接口
    def apply_action(action: np.ndarray) -> None: ...

    # 观测接口
    def get_observation(opponent_robot=None) -> np.ndarray: ...

    # 状态接口
    def get_position() -> np.ndarray: ...
    def reset(position, orientation) -> None: ...
```

### StepDataBuilder (观测/奖励构建)

```python
class StepDataBuilder(BaseHook):
    def build_step_data(
        self,
        f_get_core_state: Callable,
        f_get_derived_state: Callable,
        f_get_sensor_data: Callable,
    ) -> Tuple[observation, reward, info]: ...

    def get_observation_space(self) -> spaces.Space: ...
```

---

## 使用示例

### 创建自定义环境

```python
from combatbench.envs.framework import SimpleCombatEnv, StepDataBuilder
from combatbench.envs.humanoid21 import Humanoid21Simulator

class MyStepDataBuilder(StepDataBuilder):
    def build_step_data(self, f_get_core_state, f_get_derived_state, f_get_sensor_data):
        # 自定义观测、奖励逻辑
        obs = {...}
        reward = {...}
        info = {...}
        return obs, reward, info

    def get_observation_space(self):
        return spaces.Dict(...)

# 创建环境
simulator = Humanoid21Simulator()
step_builder = MyStepDataBuilder()
env = SimpleCombatEnv(
    simulator=simulator,
    step_data_builder=step_builder,
    hooks=[...],  # 自定义 Hooks
)
```

### 创建自定义 Hook

```python
from combatbench.envs.framework import BaseHook, InvokeType

class MyHook(BaseHook):
    @property
    def name(self) -> str:
        return "my_hook"

    @property
    def priority(self) -> int:
        return 0

    def invoke(self, invoke_type: InvokeType, **kwargs) -> bool:
        if invoke_type == InvokeType.POST_PHY_STEP:
            # 在物理步后执行自定义逻辑
            pass
        return False  # 不终止
```

---

## 模块职责

| 模块 | 职责 | 依赖 |
|------|------|------|
| `core/` | 机器人无关的基础组件 | MuJoCo |
| `framework/` | 仿真运行框架 | core/ |
| `humanoid21/` | Humanoid21 具体实现 | framework/, core/ |
| `preset_envs.py` | 预置 Gym 环境 | humanoid21/, framework/ |

---

## 设计优势

1. **可扩展性**: 通过 Hook 机制轻松添加新功能
2. **可复用性**: 框架层可用于其他机器人仿真
3. **可测试性**: 各层独立，便于单元测试
4. **灵活性**: 支持多种环境配置 (NonFall/Fall, 不同持续时间等)
