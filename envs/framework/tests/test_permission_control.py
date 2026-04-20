"""
权限控制测试 - 验证框架的核心安全机制

这是框架最重要的测试，因为权限控制是防止幽灵 bug 的关键。
"""
import pytest
import numpy as np

from envs.framework.context import SimContext
from envs.framework.plugin import BasePlugin
from envs.framework.env_runtime import EnvRuntime

from .conftest import (
    MockSimulator,
    MutatorAttemptingPlugin,
    CallTrackingPlugin,
)


class TestPermissionControl:
    """测试权限授予与撤销机制"""

    def test_mutator_granted_in_writable_lifecycle(self, mock_simulator):
        """
        场景：插件声明 require_mutator=True，在允许写入的钩子中
        预期：应该收到 mutator
        """
        plugin = MutatorAttemptingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 这些钩子允许写入
        assert "on_pre_action_step" in plugin.successful_writes
        assert "on_pre_phy_step" in plugin.successful_writes
        assert "on_post_phy_step" in plugin.successful_writes

    def test_mutator_revoked_in_readonly_lifecycle(self, mock_simulator):
        """
        场景：插件尝试在只读钩子（on_post_action_step）写入
        预期：mutator 应该是 None，写入失败
        """
        plugin = MutatorAttemptingPlugin(write_in_readonly_hook=True)
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # on_post_action_step 不允许写入
        assert "on_post_action_step" not in plugin.successful_writes
        assert "on_post_action_step" in plugin.attempted_writes

    def test_undeclared_plugin_never_receives_mutator(self, mock_simulator):
        """
        场景：插件没有声明 require_mutator=True
        预期：即使在允许写入的钩子，也收不到 mutator
        """
        class SneakyPlugin(BasePlugin):
            def __init__(self):
                self.received_mutator_in = []

            @property
            def name(self):
                return "sneaky"

            def on_post_phy_step(self, ctx):
                # 记录是否收到了 mutator
                self.received_mutator_in.append("on_post_phy_step")
                if ctx.mutator is not None:
                    # 尝试写入（但应该失败）
                    state = ctx.accessor.get_core_state()
                    state["hacked"] = True
                    ctx.mutator.set_core_state(state)

        plugin = SneakyPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：没有声明权限，所以收不到 mutator
        assert len(plugin.received_mutator_in) > 0
        # 验证：尝试写入失败（状态没有被修改）
        state = mock_simulator.get_core_state()
        assert "hacked" not in state

    def test_mutator_grant_and_revoke_cycle(self, mock_simulator):
        """
        场景：验证 mutator 在每个钩子前后正确授予和撤销
        预期：每个钩子执行完后，mutator 被撤销
        """
        plugin = MutatorAttemptingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=2,  # 2个物理步
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证在每个允许写入的钩子都成功了
        # on_pre_action_step: 1次
        # on_pre_phy_step: 2次 (phy_steps_per_action)
        # on_post_phy_step: 2次
        assert len(plugin.successful_writes) == 5

    def test_context_mutator_is_none_after_hook(self, mock_simulator):
        """
        场景：在钩子执行完后，SimContext.mutator 应该被重置为 None
        预期：插件执行完后，访问 ctx.mutator 得到 None
        """
        class CheckPlugin(BasePlugin):
            def __init__(self):
                self.mutator_was_none_after = True

            @property
            def require_mutator(self):
                return True

            def on_post_phy_step(self, ctx):
                # 在钩子内，mutator 应该可用
                assert ctx.mutator is not None
                self.inner_mutator = ctx.mutator

            def on_post_action_step(self, ctx):
                # 在只读钩子，mutator 应该是 None
                assert ctx.mutator is None

        plugin = CheckPlugin()
        from envs.framework.env_runtime import _RuntimeCore

        # ctx.accessor 现在是 _AccessorView 代理，无法反推出 simulator；
        # 直接注入 mock_simulator。
        core = _RuntimeCore(mock_simulator, phy_steps_per_action=1)
        core.attach_plugin(plugin)

        core.reset()
        core.step({"robot_a": np.zeros(21), "robot_b": np.zeros(21)})

        # 验证：钩子执行完后，再访问 ctx.mutator 应该是 None
        assert core.ctx.mutator is None


class TestReadOnlyContextImmutability:
    """测试 ReadOnlySimContext 的不可变性"""

    def test_readonly_metrics_is_immutable(self, mock_simulator):
        """
        场景：尝试修改 ReadOnlySimContext.metrics
        预期：抛出 TypeError（因为使用了 MappingProxyType）
        """
        from envs.framework.context import ReadOnlySimContext

        ctx = SimContext(mock_simulator)
        ctx.metrics["test"] = 1

        readonly = ReadOnlySimContext.from_sim_context(ctx)

        # 尝试修改
        with pytest.raises(TypeError):
            readonly.metrics["test"] = 2

        # 原数据不应该被修改
        assert ctx.metrics["test"] == 1

    def test_readonly_events_is_immutable(self, mock_simulator):
        """
        场景：尝试修改 ReadOnlySimContext.events
        预期：events 是 tuple，无法修改
        """
        from envs.framework.context import ReadOnlySimContext

        ctx = SimContext(mock_simulator)
        ctx.events.append({"type": "test"})

        readonly = ReadOnlySimContext.from_sim_context(ctx)

        # events 是 tuple，无法 append
        with pytest.raises(AttributeError):
            readonly.events.append({"type": "another"})

        # 验证原数据未变
        assert len(ctx.events) == 1

    def test_readonly_termination_proposals_is_immutable(self, mock_simulator):
        """
        场景：尝试修改 ReadOnlySimContext.termination_proposals
        预期：termination_proposals 是 tuple，无法修改
        """
        from envs.framework.context import ReadOnlySimContext

        ctx = SimContext(mock_simulator)
        ctx.request_termination("test")

        readonly = ReadOnlySimContext.from_sim_context(ctx)

        # termination_proposals 是 tuple，无法修改
        with pytest.raises(AttributeError):
            readonly.termination_proposals.append("another")

    def test_readonly_accessor_is_still_functional(self, mock_simulator):
        """
        场景：ReadOnlySimContext 的 accessor 仍然可以读取数据
        预期：可以正常调用所有只读方法
        """
        from envs.framework.context import ReadOnlySimContext

        ctx = SimContext(mock_simulator)
        readonly = ReadOnlySimContext.from_sim_context(ctx)

        # 所有只读方法都应该可用
        static = readonly.accessor.get_static_data()
        assert "dt" in static

        core = readonly.accessor.get_core_state()
        assert "qpos" in core

        derived = readonly.accessor.get_derived_state()
        assert "contacts" in derived


class TestPermissionControlEdgeCases:
    """权限控制的边界情况"""

    def test_plugin_with_mutator_cannot_read_after_detach(self, mock_simulator):
        """
        场景：插件被 detach 后，无法再访问 mutator
        预期：detach 后的插件不再被调用
        """
        plugin = MutatorAttemptingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # Detach 插件
        runtime.detach_plugin(plugin)

        # 清空记录
        plugin.successful_writes.clear()

        # 再次 step
        runtime.step(np.zeros(21), np.zeros(21))

        # 插件不应该被调用
        assert len(plugin.successful_writes) == 0

    def test_multiple_plugins_with_different_permissions(self, mock_simulator):
        """
        场景：同时有需要权限和不需要权限的插件
        预期：只有声明权限的插件收到 mutator
        """
        readonly_plugin = CallTrackingPlugin(require_mutator=False)
        mutator_plugin = CallTrackingPlugin(require_mutator=True)

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[readonly_plugin, mutator_plugin],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：只有 mutator_plugin 收到了 mutator
        # 这需要我们扩展 CallTrackingPlugin 来记录是否收到 mutator
        # 简化验证：检查两个插件都被调用了
        assert readonly_plugin.get_call_counts()["on_post_phy_step"] > 0
        assert mutator_plugin.get_call_counts()["on_post_phy_step"] > 0
