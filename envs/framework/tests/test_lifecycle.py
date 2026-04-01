"""
生命周期测试 - 验证钩子调用顺序和终止传播机制
"""
import pytest
import numpy as np

from envs.framework.context import TerminationReason
from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin

from .conftest import (
    MockSimulator,
    CallTrackingPlugin,
    TerminationRequestPlugin,
)


class TestLifecycleHookOrder:
    """测试生命周期钩子的调用顺序"""

    def test_complete_lifecycle_call_sequence(self, mock_simulator):
        """
        场景：执行完整的 episode
        预期：钩子按以下顺序调用
        1. on_attach
        2. on_pre_episode
        3. on_pre_action_step
        4. (on_pre_phy_step → physical_step → on_post_phy_step) × N
        5. on_post_action_step
        6. on_post_episode
        """
        plugin = CallTrackingPlugin(require_mutator=True)
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=2,
            max_steps=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        counts = plugin.get_call_counts()

        # 验证调用顺序和次数
        assert counts["on_attach"] == 1
        assert counts["on_pre_episode"] == 1
        assert counts["on_pre_action_step"] == 1
        assert counts["on_pre_phy_step"] == 2  # phy_steps_per_action
        assert counts["on_post_phy_step"] == 2
        assert counts["on_post_action_step"] == 1
        assert counts["on_post_episode"] == 1

    def test_multiple_steps_call_hooks_repeatedly(self, mock_simulator):
        """
        场景：执行多个 action step
        预期：每个 step 都调用对应的钩子
        """
        plugin = CallTrackingPlugin(require_mutator=True)
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            phy_steps_per_action=1,
            max_steps=3,
        )

        runtime.reset()

        # Step 1
        runtime.step(np.zeros(21), np.zeros(21))
        counts = plugin.get_call_counts()
        assert counts["on_pre_action_step"] == 1
        assert counts["on_post_action_step"] == 1

        # Step 2
        runtime.step(np.zeros(21), np.zeros(21))
        counts = plugin.get_call_counts()
        assert counts["on_pre_action_step"] == 2
        assert counts["on_post_action_step"] == 2

        # Step 3 (触发超时)
        runtime.step(np.zeros(21), np.zeros(21))
        counts = plugin.get_call_counts()
        assert counts["on_pre_action_step"] == 3
        assert counts["on_post_action_step"] == 3
        assert counts["on_post_episode"] == 1


class TestTerminationPropagation:
    """测试终止请求的传播机制"""

    def test_termination_in_pre_phy_step_stops_physical_step(self, mock_simulator):
        """
        场景：在 on_pre_phy_step 请求终止
        预期：
        1. physical_step() 不被执行
        2. 当前物理步的 on_post_phy_step 不被执行
        3. on_post_action_step 仍然被执行
        """
        class StepCounter(BasePlugin):
            def __init__(self):
                self.pre_phy_calls = 0
                self.post_phy_calls = 0
                self.step_terminated = False

            @property
            def require_mutator(self):
                return True

            def on_pre_phy_step(self, ctx):
                self.pre_phy_calls += 1
                if self.pre_phy_calls == 5:
                    ctx.request_termination("early_stop")
                    self.step_terminated = True

            def on_post_phy_step(self, ctx):
                self.post_phy_calls += 1

        counter = StepCounter()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[counter],
            phy_steps_per_action=10,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 在第5个物理步终止
        assert counter.step_terminated
        assert counter.pre_phy_calls == 5
        # 第5步没有执行 post_phy_step（因为 physical_step 没执行）
        assert counter.post_phy_calls == 4

    def test_termination_in_post_phy_step_stops_remaining_steps(self, mock_simulator):
        """
        场景：在 on_post_phy_step 请求终止
        预期：当前物理步已完成，但剩余物理步不再执行
        """
        class StepCounter(BasePlugin):
            def __init__(self):
                self.count = 0

            @property
            def require_mutator(self):
                return True

            def on_post_phy_step(self, ctx):
                self.count += 1
                if self.count == 3:
                    ctx.request_termination("after_third")

        counter = StepCounter()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[counter],
            phy_steps_per_action=10,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 应该在第3个物理步后终止
        assert counter.count == 3

    def test_termination_in_pre_action_step_prevents_physical_steps(self, mock_simulator):
        """
        场景：在 on_pre_action_step 请求终止
        预期：不执行任何物理步
        """
        class PhyStepChecker(BasePlugin):
            def __init__(self):
                self.phy_steps_executed = 0

            def on_phy_step_attempt(self):
                self.phy_steps_executed += 1

            @property
            def require_mutator(self):
                return True

            def on_pre_phy_step(self, ctx):
                self.phy_steps_executed += 1

        class TerminateAtPreAction(BasePlugin):
            def on_pre_action_step(self, ctx):
                ctx.request_termination("before_any_phy")

        checker = PhyStepChecker()
        terminator = TerminateAtPreAction()

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[terminator, checker],
            phy_steps_per_action=10,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 没有执行任何物理步
        assert checker.phy_steps_executed == 0

    def test_termination_always_calls_on_post_episode(self, mock_simulator):
        """
        场景：在不同钩子中请求终止
        预期：on_post_episode 始终被调用
        """
        termination_points = [
            "on_pre_action_step",
            "on_pre_phy_step",
            "on_post_phy_step",
            "on_post_action_step",
        ]

        for point in termination_points:
            plugin = TerminationRequestPlugin(terminate_at=point)

            class EpisodeTracker(BasePlugin):
                def __init__(self):
                    self.post_episode_called = False

                def on_post_episode(self, ctx):
                    self.post_episode_called = True

            tracker = EpisodeTracker()
            runtime = EnvRuntime(
                simulator=mock_simulator,
                plugins=[plugin, tracker],
                phy_steps_per_action=5,
            )

            runtime.reset()
            runtime.step(np.zeros(21), np.zeros(21))

            # 无论在哪里终止，on_post_episode 都应该被调用
            assert tracker.post_episode_called, f"Failed for termination at {point}"

    def test_multiple_termination_proposals_are_preserved(self, mock_simulator):
        """
        场景：多个插件同时请求终止
        预期：所有终止原因都被保留
        """
        plugin_a = TerminationRequestPlugin(terminate_at="on_post_action_step", reason="reason_a")
        plugin_b = TerminationRequestPlugin(terminate_at="on_post_action_step", reason="reason_b")

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin_a, plugin_b],
            phy_steps_per_action=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：两个终止原因都被保留
        proposals = runtime.ctx.termination_proposals
        assert "reason_a" in proposals
        assert "reason_b" in proposals

    def test_terminated_runtime_cannot_step(self, mock_simulator):
        """
        场景：episode 终止后尝试继续 step
        预期：抛出 RuntimeError
        """
        plugin = TerminationRequestPlugin(terminate_at="on_post_action_step")
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            max_steps=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # Episode 已终止
        assert runtime.ctx.is_terminated

        # 尝试继续 step 应该抛出异常
        with pytest.raises(RuntimeError, match="termination"):
            runtime.step(np.zeros(21), np.zeros(21))


class TestEpisodeBoundary:
    """测试 Episode 边界处理"""

    def test_reset_clears_all_episode_state(self, mock_simulator):
        """
        场景：运行一个 episode 后 reset
        预期：所有 episode 状态被清理
        """
        class StatefulPlugin(BasePlugin):
            def on_post_action_step(self, ctx):
                ctx.metrics["accumulated"] = ctx.metrics.get("accumulated", 0) + 1
                ctx.events.append({"step": ctx.episode_step})

        plugin = StatefulPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            max_steps=5,
        )

        # Episode 1
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.ctx.metrics["accumulated"] == 2
        assert len(runtime.ctx.events) == 2

        # Episode 2
        runtime.reset()
        assert runtime.ctx.metrics == {}
        assert len(runtime.ctx.events) == 0
        assert runtime.ctx.episode_step == 0
        assert runtime.ctx.physics_step == 0

    def test_reset_after_termination(self, mock_simulator):
        """
        场景：episode 终止后 reset
        预期：可以正常开始新的 episode
        """
        plugin = TerminationRequestPlugin(terminate_at="on_post_action_step")
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            max_steps=1,
        )

        # Episode 1
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.ctx.is_terminated

        # Episode 2
        runtime.reset()
        assert not runtime.ctx.is_terminated
        assert runtime.ctx.episode_step == 0

        # 可以正常 step
        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.ctx.episode_step == 1

    def test_reset_clears_termination_proposals(self, mock_simulator):
        """
        场景：有终止提案的 episode 后 reset
        预期：termination_proposals 被清空
        """
        plugin = TerminationRequestPlugin(terminate_at="on_post_action_step")
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            max_steps=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert len(runtime.ctx.termination_proposals) > 0

        runtime.reset()
        assert len(runtime.ctx.termination_proposals) == 0


class TestActionHandling:
    """测试动作处理"""

    def test_action_set_before_on_pre_action_step(self, mock_simulator):
        """
        场景：验证动作在 on_pre_action_step 前被设置
        预期：插件可以在 on_pre_action_step 中读取/修改动作
        """
        class ActionInspector(BasePlugin):
            def __init__(self):
                self.action_in_pre_action_step = None

            @property
            def require_mutator(self):
                return True

            def on_pre_action_step(self, ctx):
                # 读取当前设置的动作
                self.action_in_pre_action_step = ctx.accessor.get_action()

        inspector = ActionInspectorPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[inspector],
        )

        runtime.reset()
        action_a = np.ones(21)
        action_b = np.zeros(21)
        runtime.step(action_a, action_b)

        # 验证：动作在 on_pre_action_step 时已设置
        assert inspector.action_in_pre_action_step is not None
        assert "robot_a" in inspector.action_in_pre_action_step
        assert "robot_b" in inspector.action_in_pre_action_step

    def test_action_modification_in_pre_action_step(self, mock_simulator):
        """
        场景：插件在 on_pre_action_step 修改动作
        预期：修改后的动作被使用
        """
        class ActionModifier(BasePlugin):
            @property
            def require_mutator(self):
                return True

            def on_pre_action_step(self, ctx):
                # 修改动作：全部置零
                action = ctx.accessor.get_action()
                action["robot_a"] = np.zeros(21)
                action["robot_b"] = np.zeros(21)
                ctx.mutator.set_action(action)

        modifier = ActionModifier()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[modifier],
        )

        runtime.reset()
        runtime.step(np.ones(21), np.ones(21))

        # 验证：动作被修改为零
        final_action = mock_simulator.get_action()
        assert np.allclose(final_action["robot_a"], 0.0)
        assert np.allclose(final_action["robot_b"], 0.0)


class ActionInspectorPlugin(BasePlugin):
    """用于测试动作检查的插件"""
    def __init__(self):
        self.action_in_pre_action_step = None

    @property
    def require_mutator(self):
        return True

    def on_pre_action_step(self, ctx):
        # 读取当前设置的动作
        self.action_in_pre_action_step = ctx.accessor.get_action()
