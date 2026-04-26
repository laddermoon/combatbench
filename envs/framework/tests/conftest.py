"""
Framework 测试配置与共享 Fixtures
"""
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class MockSimulator:
    """
    轻量级模拟物理引擎，用于测试

    实现 BaseSimulator 的所有接口，但不执行真实物理模拟
    """
    def __init__(self):
        self._state = {
            "qpos": np.zeros(100, dtype=np.float32),
            "qvel": np.zeros(100, dtype=np.float32),
            "robot_a": {
                "root_position": np.array([0.0, 0.0, 1.28]),
                "root_orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            "robot_b": {
                "root_position": np.array([2.0, 0.0, 1.28]),
                "root_orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            },
        }
        self._action = {"robot_a": np.zeros(21), "robot_b": np.zeros(21)}
        self._static_data = {"dt": 0.002, "robot_info": {}}
        self._derived_state = {"contacts": []}
        self._sensor_data = np.zeros(10)
        self._image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._is_closed = False

    # IDataAccessor 接口
    def get_static_data(self) -> Dict[str, Any]:
        return self._static_data.copy()

    def get_core_state(self) -> Dict[str, Any]:
        # 返回深拷贝，避免外部修改
        import copy
        result = {
            "qpos": self._state["qpos"].copy(),
            "qvel": self._state["qvel"].copy(),
            "robot_a": copy.deepcopy(self._state["robot_a"]),
            "robot_b": copy.deepcopy(self._state["robot_b"]),
        }
        # 添加任意其他状态键（用于测试）
        for key, value in self._state.items():
            if key not in result:
                result[key] = value
        return result

    def get_derived_state(self) -> Dict[str, Any]:
        return self._derived_state.copy()

    def get_sensor_data(self) -> Dict[str, Any]:
        return {"sensordata": self._sensor_data.copy()}

    def get_action(self) -> Dict[str, Any]:
        return self._action.copy()

    def get_broadcastview_image(self) -> Any:
        return self._image.copy()

    # IDataMutator 接口
    def set_core_state(self, state: Dict[str, Any]) -> None:
        if self._is_closed:
            return
        # 处理标准状态
        if "qpos" in state:
            self._state["qpos"] = state["qpos"].copy()
        if "qvel" in state:
            self._state["qvel"] = state["qvel"].copy()
        if "robot_a" in state:
            self._state["robot_a"] = state["robot_a"].copy()
        if "robot_b" in state:
            self._state["robot_b"] = state["robot_b"].copy()
        # 处理任意其他状态键（用于测试）
        for key, value in state.items():
            if key not in ("qpos", "qvel", "robot_a", "robot_b"):
                self._state[key] = value

    def set_action(self, action: Dict[str, Any]) -> None:
        if self._is_closed:
            return
        self._action = {k: v.copy() if hasattr(v, 'copy') else v
                       for k, v in action.items()}

    # BaseSimulator 接口
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> None:
        self._state["qpos"] = np.zeros(100, dtype=np.float32)
        self._state["qvel"] = np.zeros(100, dtype=np.float32)
        self._action = {"robot_a": np.zeros(21), "robot_b": np.zeros(21)}

    def physical_step(self) -> None:
        if self._is_closed:
            return
        # 模拟物理步：简单增加一些噪声
        self._state["qpos"] += np.random.randn(100) * 0.001

    def get_physical_frequency(self) -> float:
        return 1.0 / self._static_data["dt"]

    def close(self) -> None:
        self._is_closed = True


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_simulator():
    """提供 MockSimulator 实例"""
    sim = MockSimulator()
    yield sim
    sim.close()


@pytest.fixture
def mock_simulator_fresh():
    """每次测试都提供全新的 MockSimulator"""
    return MockSimulator()


# 导入框架模块（用于测试）
from envs.framework.context import SimContext, ReadOnlySimContext, TerminationReason
from envs.framework.plugin import BasePlugin
from envs.framework.runtime_plugin import BaseObserverPlugin
from envs.framework.env_runtime import EnvRuntime


@pytest.fixture
def sim_context(mock_simulator):
    """提供 SimContext 实例"""
    return SimContext(mock_simulator)


@pytest.fixture
def runtime(mock_simulator):
    """提供基础 EnvRuntime 实例"""
    return EnvRuntime(
        simulator=mock_simulator,
        phy_steps_per_action=10,
        max_steps=100,
    )


@pytest.fixture
def runtime_with_plugins(mock_simulator):
    """提供带有测试插件的 EnvRuntime"""
    from envs.framework.common_plugins import TimeoutPlugin

    return EnvRuntime(
        simulator=mock_simulator,
        plugins=[TimeoutPlugin(max_steps=100)],
        phy_steps_per_action=10,
        max_steps=100,
    )


# =============================================================================
# 测试用插件
# =============================================================================

class CallTrackingPlugin(BasePlugin):
    """记录所有钩子调用次数的测试插件"""

    def __init__(self, require_mutator: bool = False):
        self._call_counts = {hook: 0 for hook in [
            "on_pre_episode", "on_pre_action_step", "on_pre_phy_step",
            "on_post_phy_step", "on_post_action_step", "on_post_episode",
            "on_attach", "on_detach",
        ]}
        self._require_mutator = require_mutator
        self._received_mutator = None

    @property
    def name(self) -> str:
        return "call_tracker"

    @property
    def priority(self) -> int:
        return 0

    @property
    def require_mutator(self) -> bool:
        return self._require_mutator

    def _track(self, hook_name):
        self._call_counts[hook_name] += 1
        # 记录是否收到了 mutator
        import inspect
        frame = inspect.currentframe()
        locals_dict = frame.f_back.f_locals
        self._received_mutator = locals_dict.get('ctx', None)

    def on_pre_episode(self, ctx):
        self._track("on_pre_episode")

    def on_pre_action_step(self, ctx):
        self._track("on_pre_action_step")

    def on_pre_phy_step(self, ctx):
        self._track("on_pre_phy_step")

    def on_post_phy_step(self, ctx):
        self._track("on_post_phy_step")

    def on_post_action_step(self, ctx):
        self._track("on_post_action_step")

    def on_post_episode(self, ctx):
        self._track("on_post_episode")

    def on_attach(self):
        self._track("on_attach")

    def on_detach(self):
        self._track("on_detach")

    def get_call_counts(self):
        return self._call_counts.copy()

    def did_receive_mutator(self):
        """最后一次调用时是否收到了 mutator"""
        if self._received_mutator:
            return self._received_mutator.mutator is not None
        return False


class MutatorAttemptingPlugin(BasePlugin):
    """尝试在各个钩子中写入状态的测试插件"""

    def __init__(self, write_in_readonly_hook: bool = False):
        self.write_in_readonly_hook = write_in_readonly_hook
        self.attempted_writes = []
        self.successful_writes = []

    @property
    def name(self) -> str:
        return "mutator_attempter"

    @property
    def require_mutator(self) -> bool:
        return True

    def _try_write(self, ctx, hook_name):
        """尝试写入状态"""
        self.attempted_writes.append(hook_name)

        if ctx.mutator is not None:
            state = ctx.accessor.get_core_state()
            state["test_marker"] = hook_name
            ctx.mutator.set_core_state(state)
            self.successful_writes.append(hook_name)

    def on_pre_action_step(self, ctx):
        self._try_write(ctx, "on_pre_action_step")

    def on_pre_phy_step(self, ctx):
        self._try_write(ctx, "on_pre_phy_step")

    def on_post_phy_step(self, ctx):
        self._try_write(ctx, "on_post_phy_step")

    def on_post_action_step(self, ctx):
        if self.write_in_readonly_hook:
            self._try_write(ctx, "on_post_action_step")


class ExceptionPlugin(BasePlugin):
    """在指定钩子抛出异常的测试插件"""

    def __init__(self, explode_at: str = None):
        self.explode_at = explode_at

    @property
    def name(self) -> str:
        return "exception_plugin"

    @property
    def priority(self) -> int:
        return 100

    def _maybe_explode(self, hook_name):
        if self.explode_at == hook_name:
            raise RuntimeError(f"Exploded at {hook_name}")

    def on_pre_episode(self, ctx):
        self._maybe_explode("on_pre_episode")

    def on_pre_action_step(self, ctx):
        self._maybe_explode("on_pre_action_step")

    def on_pre_phy_step(self, ctx):
        self._maybe_explode("on_pre_phy_step")

    def on_post_phy_step(self, ctx):
        self._maybe_explode("on_post_phy_step")

    def on_post_action_step(self, ctx):
        self._maybe_explode("on_post_action_step")

    def on_post_episode(self, ctx):
        self._maybe_explode("on_post_episode")


class TerminationRequestPlugin(BasePlugin):
    """在特定时机请求终止的测试插件"""

    def __init__(self, terminate_at: str = None, reason: str = "test"):
        self.terminate_at = terminate_at
        self.reason = reason

    @property
    def name(self) -> str:
        return "termination_requester"

    def _maybe_terminate(self, ctx, hook_name):
        if self.terminate_at == hook_name:
            ctx.request_termination(self.reason)

    def on_pre_action_step(self, ctx):
        self._maybe_terminate(ctx, "on_pre_action_step")

    def on_pre_phy_step(self, ctx):
        self._maybe_terminate(ctx, "on_pre_phy_step")

    def on_post_phy_step(self, ctx):
        self._maybe_terminate(ctx, "on_post_phy_step")

    def on_post_action_step(self, ctx):
        self._maybe_terminate(ctx, "on_post_action_step")


class CountingObserver(BaseObserverPlugin):
    """记录调用次数的 Observer"""

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0
        self.episode_count = 0
        self.refresh_count = 0
        self._output = 0

    def on_pre_episode(self, ctx):
        self.reset_count += 1

    def on_post_action_step(self, ctx):
        self.step_count += 1

    def on_post_episode(self, ctx):
        self.episode_count += 1

    def on_manual_refresh(self, ctx):
        self.refresh_count += 1

    def get_output(self):
        return self._output


class StateModifyingPlugin(BasePlugin):
    """修改状态的插件，用于测试竞争"""

    def __init__(self, value: str, priority: int = 0):
        self.value = value
        self._priority = priority

    @property
    def name(self) -> str:
        return f"state_modifier_{self.value}"

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def require_mutator(self) -> bool:
        return True

    def on_post_phy_step(self, ctx):
        state = ctx.accessor.get_core_state()
        state["test_value"] = self.value
        ctx.mutator.set_core_state(state)
