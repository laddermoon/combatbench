# 强化学习环境构建框架

简洁版框架，最简单的实现方式。

## 核心思想

只需要一个组件：
1. **StepDataBuilder** - 构建 observation、reward 和 info

终止条件：
- **时间到**：通过 `match_duration` 参数自动处理
- **其他**：Hook 返回 `True`

## 快速开始

```python
from things.combatbench.envs import (
    Humanoid21Simulator,
    SimpleCombatEnv,
    DefaultStepDataBuilder,
)

# 1. 创建仿真器
simulator = Humanoid21Simulator(gui=False, initial_distance=2.0)

# 2. 创建组件
step_builder = DefaultStepDataBuilder()

# 3. 创建环境
env = SimpleCombatEnv(
    simulator=simulator,
    step_data_builder=step_builder,
    match_duration=30.0,  # 30秒后自动终止
)

# 4. 使用
obs, info = env.reset()
action = {
    'robot_a': env.action_space['robot_a'].sample(),
    'robot_b': env.action_space['robot_b'].sample(),
}
obs, reward, terminated, truncated, info = env.step(action)
```

## 组件

### StepDataBuilder（必需）

构建观测、奖励和信息：

```python
from things.combatbench.envs.rl_env import StepDataBuilder
from gymnasium import spaces

class MyStepDataBuilder(StepDataBuilder):
    def build_step_data(
        self,
        f_get_core_state,
        f_get_derived_state,
        f_get_sensor_data,
    ):
        # 获取观测
        derived_state = f_get_derived_state()
        obs_a = derived_state['robots']['robot_a']['observation']
        obs_b = derived_state['robots']['robot_b']['observation']

        # 计算奖励
        reward_a = 0.0
        reward_b = 0.0

        # 构建信息
        core_state = f_get_core_state()
        info = {
            'step': core_state.get('step_count', 0),
        }

        return {
            'robot_a_obs': obs_a.astype(np.float32),
            'robot_b_obs': obs_b.astype(np.float32),
        }, {
            'robot_a': reward_a,
            'robot_b': reward_b,
        }, info

    def get_observation_space(self):
        return spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(127,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(127,), dtype=np.float32),
        })
```

### Hook（可选，用于控制终止）

```python
from things.combatbench.envs import BaseHook, InvokeType

class MyTerminationHook(BaseHook):
    def invoke(self, invoke_type: InvokeType, f_get_core_state=None, **kwargs):
        if invoke_type == InvokeType.POST_ACTION_STEP:
            # 检查终止条件
            if should_terminate:
                return True  # 终止
        return False
```

## 内置 Hook

| Hook | 作用 |
|------|------|
| `HealthTerminationHook` | 血量归零终止 |

## 终止控制

有两种终止方式：

1. **时间限制**（自动）
```python
env = SimpleCombatEnv(
    ...,
    match_duration=30.0,  # 30秒后自动终止
)
```

2. **Hook 返回 True**
```python
class CustomTerminationHook(BaseHook):
    def invoke(self, invoke_type: InvokeType, **kwargs):
        if invoke_type == InvokeType.POST_ACTION_STEP:
            if your_condition:
                return True  # 终止
        return False

env = SimpleCombatEnv(
    ...,
    hooks=[CustomTerminationHook()],
)
```

## Hook 调用时机

| 时机 | 说明 |
|------|------|
| `PRE_EPISODE` | Episode 开始前 |
| `POST_EPISODE` | Episode 结束后 |
| `PRE_ACTION_STEP` | 动作步开始前 |
| `POST_ACTION_STEP` | 动作步结束后 |
| `PRE_PHY_STEP` | 物理步前 |
| `POST_PHY_STEP` | 物理步后 |

可以指定 Hook 只在特定时机调用：
```python
env = SimpleCombatEnv(
    ...,
    hooks=[
        MyHook(),  # 所有时机
        (MyHook2(), [InvokeType.POST_PHY_STEP]),  # 仅物理步后
    ],
)
```

## 完整示例

参见 `example_rl_env.py`

