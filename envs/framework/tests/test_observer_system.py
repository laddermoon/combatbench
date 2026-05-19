"""
Observer 系统测试 - 验证 Observer 插件调度和去重机制
"""
import pytest
import numpy as np

from envs.framework.observer_plugin import BaseObserverPlugin, _ObserverDispatcherPlugin
from envs.framework.context import ReadOnlySimContext
from envs.framework.env_runtime import EnvRuntime

from .conftest import (
    MockSimulator,
    CountingObserver,
)


class TestObserverDispatchTiming:
    """测试 Observer 在正确时机被调用"""

    def test_observer_on_pre_episode_called_after_runtime_reset(self, mock_simulator):
        """
        场景：调用 runtime.reset()
        预期：observer.on_pre_episode() 被调用一次
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()

        assert observer.reset_count == 1
        assert observer.step_count == 0
        assert observer.episode_count == 0

    def test_observer_on_post_action_step_called_after_step(self, mock_simulator):
        """
        场景：调用 runtime.step()
        预期：observer.on_post_action_step() 被调用一次
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        observer.step_count = 0  # 清零
        runtime.step(np.zeros(21), np.zeros(21))

        assert observer.step_count == 1
        assert observer.episode_count == 0

    def test_observer_on_post_episode_called_after_termination(self, mock_simulator):
        """
        场景：episode 终止
        预期：observer.on_post_episode() 被调用一次
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
            max_steps=1,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        assert observer.episode_count == 1

    def test_observer_on_manual_refresh(self, mock_simulator):
        """
        场景：调用 runtime.refresh_observers()
        预期：observer.on_manual_refresh() 被调用
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        runtime.refresh_observers()

        assert observer.refresh_count == 1


class TestObserverDeduplication:
    """测试 Observer 去重机制"""

    def test_same_observer_instance_deduplicated(self, mock_simulator):
        """
        场景：同一个 observer 实例挂载到多个名称
        预期：on_post_action_step 只被调用一次
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={},
        )

        # 同一实例挂载到两个名称
        runtime.attach_observer_plugin("obs_a", observer)
        runtime.attach_observer_plugin("obs_b", observer)

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 应该只调用一次
        assert observer.reset_count == 1
        assert observer.step_count == 1

    def test_different_instances_not_deduplicated(self, mock_simulator):
        """
        场景：不同 observer 实例挂载
        预期：每个都被调用
        """
        observer_a = CountingObserver()
        observer_b = CountingObserver()

        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={
                "obs_a": observer_a,
                "obs_b": observer_b,
            },
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 两个都被调用
        assert observer_a.step_count == 1
        assert observer_b.step_count == 1

    def test_detaching_one_name_keeps_other(self, mock_simulator):
        """
        场景：同一实例挂载为两个名称，detach 一个
        预期：observer 仍然被调用（因为还有另一个名称引用）
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={},
        )

        runtime.attach_observer_plugin("obs_a", observer)
        runtime.attach_observer_plugin("obs_b", observer)

        # Detach 一个名称
        runtime.detach_observer_plugin("obs_a")

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 还是被调用了（通过 obs_b）
        assert observer.reset_count == 1
        assert observer.step_count == 1

    def test_detaching_all_names_stops_observer(self, mock_simulator):
        """
        场景：同一实例挂载为两个名称，全部 detach
        预期：observer 不再被调用
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={},
        )

        runtime.attach_observer_plugin("obs_a", observer)
        runtime.attach_observer_plugin("obs_b", observer)

        # Detach 所有名称
        runtime.detach_observer_plugin("obs_a")
        runtime.detach_observer_plugin("obs_b")

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 不再被调用
        assert observer.reset_count == 0
        assert observer.step_count == 0


class TestObserverReadOnlyContext:
    """测试 Observer 接收的是 ReadOnlySimContext"""

    def test_observer_receives_readonly_context(self, mock_simulator):
        """
        场景：observer 在钩子中检查 context 类型
        预期：收到的是 ReadOnlySimContext，不是 SimContext
        """
        class TypeCheckingObserver(BaseObserverPlugin):
            def __init__(self):
                self.context_type = None
                self.had_mutator = None

            def on_post_action_step(self, ctx):
                self.context_type = type(ctx).__name__
                self.had_mutator = hasattr(ctx, 'mutator') and ctx.mutator is not None

            def get_output(self):
                return self.context_type

        observer = TypeCheckingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证收到的是 ReadOnlySimContext
        assert observer.context_type == "ReadOnlySimContext"
        # 验证没有 mutator
        assert observer.had_mutator is False

    def test_observer_cannot_modify_through_context(self, mock_simulator):
        """
        场景：observer 尝试通过 context 修改状态
        预期：ReadOnlySimContext 没有 mutator
        """
        class AttempterObserver(BaseObserverPlugin):
            def __init__(self):
                self.had_mutator = False

            def on_post_action_step(self, ctx):
                self.had_mutator = hasattr(ctx, 'mutator') and ctx.mutator is not None
                if hasattr(ctx, 'mutator') and ctx.mutator is not None:
                    # 尝试修改（不应该执行到这里）
                    ctx.mutator.set_core_state({"hacked": True})

            def get_output(self):
                return self.had_mutator

        observer = AttempterObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：没有 mutator 可用
        assert observer.get_output() is False


class TestObserverOutput:
    """测试 Observer 输出机制"""

    def test_get_observer_output_returns_latest(self, mock_simulator):
        """
        场景：observer 在 on_post_action_step 中更新输出
        预期：get_observer_output() 返回最新值
        """
        class ValueObserver(BaseObserverPlugin):
            def __init__(self):
                self._value = 0

            def on_pre_episode(self, ctx):
                self._value = 10

            def on_post_action_step(self, ctx):
                self._value += 1

            def get_output(self):
                return self._value

        observer = ValueObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        assert runtime.get_observer_output("test") == 10

        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.get_observer_output("test") == 11

        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.get_observer_output("test") == 12

    def test_get_observer_outputs_returns_all(self, mock_simulator):
        """
        场景：多个 observer
        预期：get_observer_outputs() 返回所有输出
        """
        observer_a = CountingObserver()
        observer_b = CountingObserver()

        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={
                "obs_a": observer_a,
                "obs_b": observer_b,
            },
        )

        runtime.reset()
        outputs = runtime.get_observer_outputs()

        assert "obs_a" in outputs
        assert "obs_b" in outputs
        assert outputs["obs_a"] == 0
        assert outputs["obs_b"] == 0

    def test_get_observer_output_of_detached_observer(self, mock_simulator):
        """
        场景：observer 被 detach
        预期：get_observer_output() 返回 None
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.detach_observer_plugin("test")

        output = runtime.get_observer_output("test")
        assert output is None


class TestObserverRefreshOptimization:
    """测试 Observer 刷新优化机制"""

    def test_observer_not_called_when_state_unchanged(self, mock_simulator):
        """
        场景：连续调用 refresh_observers() 但状态没变
        预期：observer 只被调用一次（token 机制）
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()

        # 多次刷新，但状态没变
        runtime.refresh_observers()
        runtime.refresh_observers()
        runtime.refresh_observers()

        # 应该只调用一次（第一次）
        assert observer.refresh_count == 1

    def test_force_refresh_ignores_token(self, mock_simulator):
        """
        场景：使用 force=True 刷新
        预期：即使状态没变，也被调用
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()

        # 强制刷新
        runtime.refresh_observers(force=True)
        runtime.refresh_observers(force=True)

        # 每次强制刷新都会调用
        assert observer.refresh_count == 2

    def test_state_change_triggers_refresh(self, mock_simulator):
        """
        场景：状态改变后自动刷新
        预期：observer 被重新调用
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
        )

        runtime.reset()
        observer.refresh_count = 0  # 清零

        # step 会改变状态
        runtime.step(np.zeros(21), np.zeros(21))

        # 验证：on_post_action_step 被调用
        assert observer.step_count == 1


class TestObserverInEpisode:
    """测试 Observer 在 episode 中的行为"""

    def test_observer_state_persists_across_steps(self, mock_simulator):
        """
        场景：observer 在每个 step 中记录值
        预期：每个 step 都被调用，状态正确维护
        """
        class StepRecordingObserver(BaseObserverPlugin):
            def __init__(self):
                self.steps = []

            def on_pre_episode(self, ctx):
                self.steps = []

            def on_post_action_step(self, ctx):
                self.steps.append(ctx.episode_step)

            def get_output(self):
                return self.steps

        observer = StepRecordingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
            max_steps=5,
        )

        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))  # episode_step becomes 1
        runtime.step(np.zeros(21), np.zeros(21))  # episode_step becomes 2
        runtime.step(np.zeros(21), np.zeros(21))  # episode_step becomes 3

        # 验证：每个 step 都被记录了
        steps = runtime.get_observer_output("test")
        assert 1 in steps  # step 1
        assert 2 in steps  # step 2
        assert 3 in steps  # step 3
        assert runtime.ctx.episode_step == 3

    def test_observer_reset_between_episodes(self, mock_simulator):
        """
        场景：多个 episode
        预期：observer 在每个 episode 开始时调用 on_pre_episode
        """
        observer = CountingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"test": observer},
            max_steps=1,
        )

        # Episode 1
        runtime.reset()
        assert observer.reset_count == 1
        runtime.step(np.zeros(21), np.zeros(21))

        # Episode 2
        runtime.reset()
        assert observer.reset_count == 2
        runtime.step(np.zeros(21), np.zeros(21))

        assert observer.reset_count == 2
