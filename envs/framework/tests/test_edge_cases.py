"""
边界情况测试 - 覆盖边缘场景和特殊条件
"""
import pytest
import numpy as np

from envs.framework.context import TerminationReason
from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin

from .conftest import MockSimulator, CountingObserver


class TestEmptyRuntime:
    """测试空 Runtime（无插件）"""

    def test_runtime_with_no_plugins(self, mock_simulator):
        """
        场景：Runtime 没有任何插件
        预期：正常运行，只是没有额外逻辑
        """
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[],
            observer_plugins={},
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证基本功能正常
        assert runtime.ctx.episode_step == 1

    def test_runtime_with_no_observer_plugins(self, mock_simulator):
        """
        场景：Runtime 没有 observer plugins
        预期：get_observer_output() 返回 None
        """
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={},
        )

        runtime.reset()
        output = runtime.get_observer_output("nonexistent")
        assert output is None


class TestMultipleObserversSameName:
    """测试同名 Observer 处理"""

    def test_attaching_observer_replaces_existing(self, mock_simulator):
        """
        场景：同一名称挂载不同的 observer 实例
        预期：新的替换旧的
        """
        from .conftest import CountingObserver

        observer_a = CountingObserver()
        observer_b = CountingObserver()

        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer_a},
        )

        runtime.reset()
        assert observer_a.reset_count == 1

        # 替换为 observer_b
        runtime.attach_observer_plugin("test", observer_b)
        observer_a.reset_count = 0  # 清零

        runtime.reset()
        assert observer_a.reset_count == 0  # 旧的不再被调用
        assert observer_b.reset_count == 1  # 新的被调用


class TestTerminationFlags:
    """测试终止标志解析"""

    def test_timeout_returns_truncated_true(self, mock_simulator):
        """
        场景：只有 TIMEOUT 终止原因
        预期：terminated=False, truncated=True
        """
        from envs.framework.common_plugins import TimeoutPlugin

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[TimeoutPlugin(max_steps=1)],
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        terminated, truncated = runtime.get_termination_flags()
        assert terminated is False
        assert truncated is True

    def test_ko_returns_terminated_true(self, mock_simulator):
        """
        场景：只有 KO 终止原因
        预期：terminated=True, truncated=False
        """
        class KOPlugin:
            def on_post_action_step(self, ctx):
                ctx.request_termination(TerminationReason.KO)

        # 由于不能直接继承 BasePlugin，用简单方式
        from envs.framework.plugin import BasePlugin

        class RealKOPlugin(BasePlugin):
            def on_post_action_step(self, ctx):
                ctx.request_termination(TerminationReason.KO)

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[RealKOPlugin()],
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        terminated, truncated = runtime.get_termination_flags()
        assert terminated is True
        assert truncated is False

    def test_timeout_with_ko_returns_terminated_true(self, mock_simulator):
        """
        场景：TIMEOUT 和 KO 同时存在
        预期：terminated=True, truncated=False（KO 优先）
        """
        from envs.framework.plugin import BasePlugin
        from envs.framework.common_plugins import TimeoutPlugin

        class KOWithTimeoutPlugin(BasePlugin):
            def on_post_action_step(self, ctx):
                # 同时请求两种终止
                ctx.request_termination(TerminationReason.TIMEOUT)
                ctx.request_termination(TerminationReason.KO)

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[KOWithTimeoutPlugin()],
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        terminated, truncated = runtime.get_termination_flags()
        # KO 存在时，返回 terminated=True
        assert terminated is True
        assert truncated is False


class TestSharedInfo:
    """测试共享信息构建"""

    def test_shared_info_contains_metrics(self, mock_simulator):
        """
        场景：插件写入 metrics
        预期：get_shared_info() 包含这些 metrics
        """
        from envs.framework.plugin import BasePlugin

        class MetricsWriter(BasePlugin):
            def on_post_action_step(self, ctx):
                ctx.metrics["test_metric"] = 42
                ctx.events.append({"type": "test"})

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[MetricsWriter()],
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        shared_info = runtime.get_shared_info()
        assert shared_info["metrics"]["test_metric"] == 42
        assert len(shared_info["events"]) == 1

    def test_shared_info_contains_step_counts(self, mock_simulator):
        """
        场景：执行多个 step
        预期：shared_info 包含正确的步数
        """
        runtime = EnvRuntime(
            simulator=mock_simulator,
            max_steps=10,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        runtime.step(np.zeros(21), np.zeros(21))

        shared_info = runtime.get_shared_info()
        assert shared_info["episode_step"] == 2

    def test_shared_info_contains_termination_status(self, mock_simulator):
        """
        场景：episode 终止
        预期：shared_info 包含终止状态
        """
        from envs.framework.common_plugins import TimeoutPlugin

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[TimeoutPlugin(max_steps=1)],
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        shared_info = runtime.get_shared_info()
        assert shared_info["is_terminated"] is True
        assert TerminationReason.TIMEOUT in shared_info["termination_reasons"]


class TestRuntimeClose:
    """测试 Runtime 关闭"""

    def test_close_clears_plugins(self, mock_simulator):
        """
        场景：调用 runtime.close()
        预期：所有插件的 on_detach 被调用
        """
        from .conftest import CallTrackingPlugin

        plugin = CallTrackingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
        )

        runtime.close()

        counts = plugin.get_call_counts()
        assert counts["on_detach"] == 1

    def test_step_after_close_raises_error(self, mock_simulator):
        """
        场景：close 后尝试 step
        预期：抛出 RuntimeError
        """
        runtime = EnvRuntime(simulator=mock_simulator)
        runtime.close()

        # 关闭后 step 应该抛出异常（因为 _is_episode_active is False）
        with pytest.raises(RuntimeError, match="reset"):
            runtime.step(np.zeros(21), np.zeros(21))


class TestPhyStepsPerAction:
    """测试物理步数配置"""

    def test_phy_steps_per_action_zero(self, mock_simulator):
        """
        场景：phy_steps_per_action = 0
        预期：不执行物理步，但仍然调用钩子
        """
        from envs.framework.plugin import BasePlugin

        class Counter(BasePlugin):
            def __init__(self):
                self.pre_phy = 0
                self.post_phy = 0

            def on_pre_phy_step(self, ctx):
                self.pre_phy += 1

            def on_post_phy_step(self, ctx):
                self.post_phy += 1

        counter = Counter()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[counter],
            phy_steps_per_action=0,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 0 个物理步，所以钩子不被调用
        assert counter.pre_phy == 0
        assert counter.post_phy == 0

    def test_phy_steps_per_action_large(self, mock_simulator):
        """
        场景：phy_steps_per_action = 100
        预期：执行 100 个物理步
        """
        from envs.framework.plugin import BasePlugin

        class Counter(BasePlugin):
            def __init__(self):
                self.count = 0

            def on_post_phy_step(self, ctx):
                self.count += 1

        counter = Counter()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[counter],
            phy_steps_per_action=100,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        assert counter.count == 100


class TestMaxSteps:
    """测试最大步数限制"""

    def test_max_steps_zero(self, mock_simulator):
        """
        场景：max_steps = 0
        预期：第一次 step 后立即终止
        """
        runtime = EnvRuntime(
            simulator=mock_simulator,
            max_steps=0,
        )

        runtime.reset()
        # reset 后不会立即终止
        assert not runtime.ctx.is_terminated

        # 第一次 step 后终止
        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.ctx.is_terminated

    def test_max_steps_prevents_step(self, mock_simulator):
        """
        场景：达到 max_steps 后尝试继续 step
        预期：抛出 RuntimeError
        """
        runtime = EnvRuntime(
            simulator=mock_simulator,
            max_steps=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.ctx.episode_step == 1
        assert runtime.ctx.is_terminated

        # 尝试继续应该抛出异常
        with pytest.raises(RuntimeError, match="termination"):
            runtime.step(np.zeros(21), np.zeros(21))


class TestSimulatorClose:
    """测试 Simulator 关闭"""

    def test_runtime_close_closes_simulator(self, mock_simulator):
        """
        场景：runtime.close()
        预期：simulator.close() 被调用
        """
        mock_simulator._is_closed = False

        runtime = EnvRuntime(simulator=mock_simulator)
        runtime.close()

        assert mock_simulator._is_closed is True


class TestActionValidation:
    """测试动作验证"""

    def test_step_without_reset_raises_error(self, mock_simulator):
        """
        场景：在 reset 之前调用 step
        预期：抛出 RuntimeError
        """
        runtime = EnvRuntime(simulator=mock_simulator)

        with pytest.raises(RuntimeError, match="reset"):
            runtime.step(np.zeros(21), np.zeros(21))

    def test_step_after_termination_raises_error(self, mock_simulator):
        """
        场景：episode 终止后继续 step
        预期：抛出 RuntimeError
        """
        from envs.framework.common_plugins import TimeoutPlugin

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[TimeoutPlugin(max_steps=1)],
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # Episode 已终止
        assert runtime.ctx.is_terminated

        # 尝试继续 step 应该抛出异常
        with pytest.raises(RuntimeError, match="termination"):
            runtime.step(np.zeros(21), np.zeros(21))


class TestObserverDynamicAttachment:
    """测试动态附加 Observer"""

    def test_attach_observer_during_episode(self, mock_simulator):
        """
        场景：在 episode 中间附加 observer
        预期：新 observer 立即生效
        """
        from .conftest import CountingObserver

        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={},
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 附加 observer
        runtime.attach_observer_plugin("test", observer)
        observer.reset_count = 0

        runtime.step(np.zeros(21), np.zeros(21))

        # 新 observer 应该被调用
        assert observer.step_count == 1

    def test_detach_observer_during_episode(self, mock_simulator):
        """
        场景：在 episode 中间 detach observer
        预期：observer 不再被调用
        """
        from .conftest import CountingObserver

        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert observer.step_count == 1

        # Detach
        runtime.detach_observer_plugin("test")
        observer.step_count = 0

        runtime.step(np.zeros(21), np.zeros(21))

        # 不再被调用
        assert observer.step_count == 0
