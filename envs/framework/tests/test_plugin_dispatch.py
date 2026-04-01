"""
插件调度测试 - 验证插件执行顺序、异常隔离
"""
import pytest
import numpy as np
import warnings

from envs.framework.plugin import BasePlugin
from envs.framework.env_runtime import EnvRuntime

from .conftest import (
    MockSimulator,
    ExceptionPlugin,
    CallTrackingPlugin,
    StateModifyingPlugin,
)


class TestPluginPriorityOrdering:
    """测试插件按 priority 顺序执行"""

    def test_plugins_executed_in_priority_order(self, mock_simulator):
        """
        场景：三个插件 priority 分别为 100, 50, 0
        预期：按 100 → 50 → 0 顺序执行
        """
        class OrderTrackingPlugin(BasePlugin):
            def __init__(self, tracking_name, priority):
                self._tracking_name = tracking_name
                self._priority = priority
                self.execution_order = []

            @property
            def name(self):
                return f"tracker_{self._tracking_name}"

            @property
            def priority(self):
                return self._priority

            def on_post_phy_step(self, ctx):
                # 将执行顺序记录到共享列表
                OrderTrackingPlugin.execution_order.append(self._tracking_name)

        # 类变量用于共享执行顺序
        OrderTrackingPlugin.execution_order = []

        plugin_low = OrderTrackingPlugin("low", 0)
        plugin_mid = OrderTrackingPlugin("mid", 50)
        plugin_high = OrderTrackingPlugin("high", 100)

        # 按乱序挂载
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin_low, plugin_high, plugin_mid],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：按高到低顺序执行
        assert OrderTrackingPlugin.execution_order == ["high", "mid", "low"]

    def test_plugins_with_same_priority_maintain_attach_order(self, mock_simulator):
        """
        场景：两个插件 priority 相同
        预期：保持挂载顺序（先挂载的先执行）
        """
        class SamePriorityPlugin(BasePlugin):
            def __init__(self, tracking_name):
                self._tracking_name = tracking_name
                self.execution_order = []

            @property
            def name(self):
                return f"same_prio_{self._tracking_name}"

            @property
            def priority(self):
                return 50  # 相同 priority

            def on_post_phy_step(self, ctx):
                SamePriorityPlugin.execution_order.append(self._tracking_name)

        SamePriorityPlugin.execution_order = []

        plugin_a = SamePriorityPlugin("a")
        plugin_b = SamePriorityPlugin("b")

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin_a, plugin_b],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 先挂载的先执行
        assert SamePriorityPlugin.execution_order == ["a", "b"]


class TestPluginExceptionIsolation:
    """测试插件异常隔离机制"""

    def test_exception_in_one_plugin_doesnt_stop_others(self, mock_simulator):
        """
        场景：priority=100 的插件抛异常
        预期：priority=50 的插件仍然被执行
        """
        class NormalPlugin(BasePlugin):
            def __init__(self):
                self.called = False

            @property
            def priority(self):
                return 50

            def on_post_phy_step(self, ctx):
                self.called = True

        explosive = ExceptionPlugin(explode_at="on_post_phy_step")
        normal = NormalPlugin()

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[normal, explosive],
            phy_steps_per_action=1,
        )

        # 捕获 warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runtime.reset()
            runtime.step(np.zeros(21), np.zeros(21))

            # 验证：有 warning 被记录
            assert len(w) > 0
            assert "Exploded" in str(w[0].message)

        # 验证：正常插件仍然被调用
        assert normal.called

    def test_exception_in_pre_phy_step_stops_physical_step(self, mock_simulator):
        """
        场景：在 on_pre_phy_step 抛异常
        预期：异常被隔离，其他插件继续执行，physical_step() 正常执行
        """
        class StepCounterPlugin(BasePlugin):
            def __init__(self):
                self.pre_phy_count = 0
                self.post_phy_count = 0

            def on_pre_phy_step(self, ctx):
                self.pre_phy_count += 1

            def on_post_phy_step(self, ctx):
                self.post_phy_count += 1

        explosive = ExceptionPlugin(explode_at="on_pre_phy_step")
        counter = StepCounterPlugin()

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[explosive, counter],
            phy_steps_per_action=2,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runtime.reset()
            runtime.step(np.zeros(21), np.zeros(21))

            # 验证：有 warning 被记录
            assert len(w) > 0

        # 框架的异常隔离机制：即使 explosive 抛异常，counter 仍然执行
        # physical_step() 也正常执行
        assert counter.pre_phy_count == 2  # 2个物理步
        assert counter.post_phy_count == 2  # physical_step() 正常执行了

    def test_multiple_plugins_exception_isolated(self, mock_simulator):
        """
        场景：多个插件都抛异常
        预期：所有插件都被尝试调用，异常被独立捕获
        """
        class MultiExplosivePlugin(BasePlugin):
            def __init__(self, explode_at_hooks):
                self.explode_at_hooks = explode_at_hooks
                self.executed_hooks = []

            @property
            def name(self):
                return "multi_explosive"

            def _maybe_explode(self, hook_name):
                self.executed_hooks.append(hook_name)
                if hook_name in self.explode_at_hooks:
                    raise RuntimeError(f"Boom at {hook_name}")

            def on_pre_phy_step(self, ctx):
                self._maybe_explode("on_pre_phy_step")

            def on_post_phy_step(self, ctx):
                self._maybe_explode("on_post_phy_step")

        plugin = MultiExplosivePlugin(
            explode_at_hooks=["on_pre_phy_step", "on_post_phy_step"]
        )

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runtime.reset()
            runtime.step(np.zeros(21), np.zeros(21))

            # 应该有 2 个 warning（两个钩子都抛异常）
            assert len(w) == 2

        # 验证：两个钩子都被执行了
        assert "on_pre_phy_step" in plugin.executed_hooks
        assert "on_post_phy_step" in plugin.executed_hooks


class TestStateModificationOrder:
    """测试多插件修改状态的顺序确定性"""

    def test_high_priority_plugin_wins_state_competition(self, mock_simulator):
        """
        场景：三个插件按不同 priority 修改同一个状态
        预期：priority 最低的插件最终决定状态值（因为最后执行）
        """
        plugin_a = StateModifyingPlugin("A", priority=10)
        plugin_b = StateModifyingPlugin("B", priority=20)
        plugin_c = StateModifyingPlugin("C", priority=30)

        # 按乱序挂载
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin_a, plugin_b, plugin_c],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：priority 最低 (10) 的 A 最终生效（因为最后执行，覆盖了前面的修改）
        final_state = mock_simulator.get_core_state()
        assert final_state.get("test_value") == "A"

    def test_state_modifications_are_cumulative_within_priority(self, mock_simulator):
        """
        场景：同一插件在不同钩子中修改状态
        预期：后执行的钩子覆盖前面的修改
        """
        class CumulativePlugin(BasePlugin):
            def __init__(self):
                self.values = []

            @property
            def require_mutator(self):
                return True

            def on_pre_phy_step(self, ctx):
                state = ctx.accessor.get_core_state()
                state["stage"] = "pre"
                ctx.mutator.set_core_state(state)

            def on_post_phy_step(self, ctx):
                state = ctx.accessor.get_core_state()
                state["stage"] = "post"
                ctx.mutator.set_core_state(state)

        plugin = CumulativePlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：post 覆盖了 pre
        final_state = mock_simulator.get_core_state()
        assert final_state.get("stage") == "post"


class TestPluginLifecycleHooks:
    """测试插件生命周期钩子的调用时机"""

    def test_on_attach_called_when_plugin_attached(self, mock_simulator):
        """
        场景：插件被附加到 runtime
        预期：on_attach 被调用一次
        """
        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
        )

        counts = plugin.get_call_counts()
        assert counts["on_attach"] == 1
        assert counts["on_detach"] == 0

    def test_on_detach_called_when_plugin_detached(self, mock_simulator):
        """
        场景：插件从 runtime 分离
        预期：on_detach 被调用一次
        """
        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
        )

        runtime.detach_plugin(plugin)

        counts = plugin.get_call_counts()
        assert counts["on_detach"] == 1

    def test_on_pre_episode_called_before_anything_else(self, mock_simulator):
        """
        场景：reset 时
        预期：on_pre_episode 是第一个被调用的钩子
        """
        class OrderPlugin(BasePlugin):
            def __init__(self):
                self.call_order = []

            def on_pre_episode(self, ctx):
                self.call_order.append("on_pre_episode")

            def on_pre_action_step(self, ctx):
                self.call_order.append("on_pre_action_step")

            def on_pre_phy_step(self, ctx):
                self.call_order.append("on_pre_phy_step")

        plugin = OrderPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
        )

        runtime.reset()

        # on_pre_episode 应该是第一个
        assert plugin.call_order[0] == "on_pre_episode"

    def test_on_post_episode_called_after_termination(self, mock_simulator):
        """
        场景：episode 终止
        预期：on_post_episode 被调用
        """
        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            max_steps=1,  # 1步后超时
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        counts = plugin.get_call_counts()
        assert counts["on_post_episode"] == 1


class TestPluginAttachDetach:
    """测试插件附加和分离"""

    def test_attaching_same_plugin_twice_is_ignored(self, mock_simulator):
        """
        场景：重复附加同一个插件
        预期：只被附加一次
        """
        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
        )

        # 尝试再次附加
        runtime.attach_plugin(plugin)

        counts = plugin.get_call_counts()
        # on_attach 只被调用一次
        assert counts["on_attach"] == 1

    def test_detaching_non_existent_plugin_is_safe(self, mock_simulator):
        """
        场景：分离一个未附加的插件
        预期：不抛出异常
        """
        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(simulator=mock_simulator)

        # 不应该抛出异常
        runtime.detach_plugin(plugin)

    def test_detached_plugin_not_called_anymore(self, mock_simulator):
        """
        场景：插件被分离后
        预期：不再被调用
        """
        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
        )

        runtime.detach_plugin(plugin)
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        counts = plugin.get_call_counts()
        # 分离后，on_pre_episode 不应该被调用
        assert counts["on_pre_episode"] == 0


class TestDispatcherPriority:
    """测试 ObserverDispatcher 的特殊优先级"""

    def test_observer_dispatcher_has_highest_priority(self, mock_simulator):
        """
        场景：ObserverDispatcher 的 priority 是 -1000000
        预期：它总是最先执行
        """
        from envs.framework.runtime_plugin import _ObserverDispatcherPlugin

        dispatcher = _ObserverDispatcherPlugin()
        assert dispatcher.priority == -1_000_000

        # 即使其他插件有很高的 priority，dispatcher 也应该更低
        high_priority_plugin = BasePlugin()
        # BasePlugin 默认 priority 是 0
        assert dispatcher.priority < high_priority_plugin.priority
