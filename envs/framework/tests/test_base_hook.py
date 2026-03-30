"""
测试 BaseHook, HookWrapper 和 InvokeType
"""

import pytest
from unittest.mock import Mock

from combatbench.envs.framework.base_hook import (
    BaseHook,
    HookWrapper,
    InvokeType,
)


# ==================== Test InvokeType ====================

class TestInvokeType:
    """测试 InvokeType 枚举"""

    def test_invoke_type_values(self):
        """测试所有 InvokeType 值存在"""
        assert hasattr(InvokeType, 'PRE_EPISODE')
        assert hasattr(InvokeType, 'POST_EPISODE')
        assert hasattr(InvokeType, 'PRE_ACTION_STEP')
        assert hasattr(InvokeType, 'POST_ACTION_STEP')
        assert hasattr(InvokeType, 'PRE_PHY_STEP')
        assert hasattr(InvokeType, 'POST_PHY_STEP')

    def test_invoke_type_uniqueness(self):
        """测试 InvokeType 值的唯一性"""
        values = set([
            InvokeType.PRE_EPISODE,
            InvokeType.POST_EPISODE,
            InvokeType.PRE_ACTION_STEP,
            InvokeType.POST_ACTION_STEP,
            InvokeType.PRE_PHY_STEP,
            InvokeType.POST_PHY_STEP,
        ])
        assert len(values) == 6


# ==================== Test BaseHook ====================

class TestBaseHook:
    """测试 BaseHook 基类"""

    def test_hook_is_abstract(self):
        """测试 BaseHook 是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            BaseHook()

    def test_hook_has_abstract_methods(self):
        """测试 BaseHook 有抽象方法"""
        assert hasattr(BaseHook, 'invoke')
        # invoke 是抽象方法，所以不能直接实例化
        from abc import ABC
        assert issubclass(BaseHook, ABC)


# ==================== Test MockHook ====================

class MockHook(BaseHook):
    """用于测试的 Mock Hook"""

    def __init__(self, name="mock_hook", priority=0):
        super().__init__()
        self._name = name
        self._priority = priority
        self.invoke_count = 0
        self.last_invoke_type = None
        self.attach_called = False
        self.detach_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def invoke(self, invoke_type, *args, **kwargs) -> bool:
        self.invoke_count += 1
        self.last_invoke_type = invoke_type
        return False  # 默认不终止

    def on_attach(self) -> None:
        self.attach_called = True

    def on_detach(self) -> None:
        self.detach_called = True


class TerminateHook(BaseHook):
    """用于测试终止的 Hook"""

    @property
    def name(self) -> str:
        return "terminate_hook"

    @property
    def priority(self) -> int:
        return 0

    def invoke(self, invoke_type, *args, **kwargs) -> bool:
        # 在 POST_ACTION_STEP 时终止
        return invoke_type == InvokeType.POST_ACTION_STEP


class TestMockHook:
    """测试 MockHook"""

    def test_mock_hook_properties(self):
        """测试 MockHook 属性"""
        hook = MockHook(name="test_hook", priority=10)
        assert hook.name == "test_hook"
        assert hook.priority == 10

    def test_mock_hook_invoke_tracking(self):
        """测试 MockHook 调用跟踪"""
        hook = MockHook()
        hook.invoke(InvokeType.PRE_EPISODE)
        hook.invoke(InvokeType.POST_ACTION_STEP)
        assert hook.invoke_count == 2
        assert hook.last_invoke_type == InvokeType.POST_ACTION_STEP

    def test_mock_hook_lifecycle_callbacks(self):
        """测试生命周期回调"""
        hook = MockHook()
        assert not hook.attach_called
        assert not hook.detach_called

        # 模拟附加
        hook.on_attach()
        assert hook.attach_called

        # 模拟分离
        hook.on_detach()
        assert hook.detach_called


# ==================== Test HookWrapper ====================

class TestHookWrapper:
    """测试 HookWrapper"""

    def test_empty_wrapper(self):
        """测试空的 HookWrapper"""
        wrapper = HookWrapper()
        assert wrapper.hooks == []
        assert len(wrapper.hooks) == 0

    def test_attach_hook(self):
        """测试附加 Hook"""
        wrapper = HookWrapper()
        hook = MockHook()

        wrapper.attach(hook)

        assert len(wrapper.hooks) == 1
        assert wrapper.hooks[0] == hook

    def test_attach_hook_with_priority(self):
        """测试附加 Hook 并设置优先级"""
        wrapper = HookWrapper()
        hook1 = MockHook(priority=10)
        hook2 = MockHook(priority=5)

        wrapper.attach(hook1)
        wrapper.attach(hook2)

        # 应该按优先级排序（数值越大越先执行）
        assert wrapper.hooks[0] == hook1  # priority 10
        assert wrapper.hooks[1] == hook2  # priority 5

    def test_attach_hook_with_invoke_types(self):
        """测试附加 Hook 并指定调用时机"""
        wrapper = HookWrapper()
        hook = MockHook()

        wrapper.attach(hook, invoke_types=[InvokeType.POST_ACTION_STEP])

        assert len(wrapper.hooks) == 1
        # 检查 Hook 的调用时机被正确设置
        hook_info = wrapper._hooks[0]
        assert InvokeType.POST_ACTION_STEP in hook_info[2]

    def test_detach_hook(self):
        """测试分离 Hook"""
        wrapper = HookWrapper()
        hook = MockHook()
        wrapper.attach(hook)

        wrapper.detach(hook)

        assert len(wrapper.hooks) == 0

    def test_detach_hook_by_id(self):
        """测试分离 Hook（通过 id）"""
        wrapper = HookWrapper()
        hook = MockHook()
        hook_id = id(hook)
        wrapper.attach(hook)

        # 创建另一个 Hook
        hook2 = MockHook(name="hook2")
        wrapper.attach(hook2)

        # 分离第一个 Hook
        wrapper.detach(hook)

        assert len(wrapper.hooks) == 1
        assert wrapper.hooks[0] == hook2
        assert id(wrapper.hooks[0]) != hook_id

    def test_detach_non_existent_hook(self):
        """测试分离不存在的 Hook（不抛异常）"""
        wrapper = HookWrapper()
        hook = MockHook()

        # 分离未附加的 Hook 应该不抛异常
        wrapper.detach(hook)
        assert len(wrapper.hooks) == 0

    def test_clear_hooks(self):
        """测试清除所有 Hooks"""
        wrapper = HookWrapper()
        hook1 = MockHook(name="hook1")
        hook2 = MockHook(name="hook2")
        wrapper.attach(hook1)
        wrapper.attach(hook2)

        wrapper.clear()

        assert len(wrapper.hooks) == 0

    def test_invoke_no_hooks(self):
        """测试没有 Hook 时的 invoke"""
        wrapper = HookWrapper()
        result = wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )
        assert result is False

    def test_invoke_single_hook(self):
        """测试单个 Hook 的 invoke"""
        wrapper = HookWrapper()
        hook = MockHook()
        wrapper.attach(hook)

        wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )

        assert hook.invoke_count == 1
        assert hook.last_invoke_type == InvokeType.POST_ACTION_STEP

    def test_invoke_multiple_hooks_priority_order(self):
        """测试多个 Hook 按优先级顺序调用"""
        wrapper = HookWrapper()
        hook1 = MockHook(name="hook1", priority=10)
        hook2 = MockHook(name="hook2", priority=20)
        hook3 = MockHook(name="hook3", priority=5)

        wrapper.attach(hook1)
        wrapper.attach(hook2)
        wrapper.attach(hook3)

        wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )

        # 验证调用顺序：hook2 (20) -> hook1 (10) -> hook3 (5)
        assert hook2.invoke_count == 1
        assert hook1.invoke_count == 1
        assert hook3.invoke_count == 1

    def test_invoke_with_invoke_types_filtering(self):
        """测试 invoke_types 过滤"""
        wrapper = HookWrapper()
        hook1 = MockHook(name="hook1")
        hook2 = MockHook(name="hook2")

        # hook1 只在 POST_ACTION_STEP 时调用
        wrapper.attach(hook1, invoke_types=[InvokeType.POST_ACTION_STEP])
        # hook2 在所有时机调用
        wrapper.attach(hook2)

        wrapper.invoke(
            invoke_type=InvokeType.PRE_EPISODE,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )

        # 只有 hook2 被调用（hook1 的 invoke_types 不包含 PRE_EPISODE）
        assert hook1.invoke_count == 0
        assert hook2.invoke_count == 1

    def test_invoke_termination(self):
        """测试 Hook 终止机制"""
        wrapper = HookWrapper()
        hook = TerminateHook()  # 在 POST_ACTION_STEP 时终止
        wrapper.attach(hook)

        # 第一个 Hook 不终止
        hook1 = MockHook(name="hook1")
        wrapper.attach(hook1)

        result = wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )

        assert result is True  # TerminateHook 返回 True
        assert hook1.invoke_count == 0  # hook1 未被调用（提前终止）

    def test_invoke_early_termination(self):
        """测试提前终止阻止后续 Hook 调用"""
        wrapper = HookWrapper()
        hook1 = TerminateHook()  # 优先级 0
        hook2 = MockHook(name="hook2", priority=10)  # 优先级更高

        wrapper.attach(hook1)
        wrapper.attach(hook2)

        result = wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )

        # hook2 应该先被调用（优先级 10 > 0）
        assert hook2.invoke_count == 1
        # 然后 TerminateHook 被调用（优先级 0）
        # 返回 True 表示终止

    def test_invoke_exception_handling(self):
        """测试 Hook 异常处理"""
        wrapper = HookWrapper()

        # 创建一个会抛出异常的 Hook
        class ExceptionHook(BaseHook):
            @property
            def name(self):
                return "exception_hook"

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                raise ValueError("Test exception")

        hook = ExceptionHook()
        wrapper.attach(hook)

        # invoke 应该捕获异常并继续
        result = wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=lambda: {},
            f_get_static_data=lambda: {},
            f_get_sensor_data=lambda: {},
            f_get_core_state=lambda: {},
            f_get_derived_state=lambda: {},
            f_set_core_state=None,
            f_set_action=None,
        )

        # 即使 Hook 抛出异常，invoke 也应该返回 False（不终止）
        assert result is False

    def test_invoke_with_all_parameters(self):
        """测试 invoke 传递所有参数"""
        wrapper = HookWrapper()
        hook = MockHook()
        wrapper.attach(hook)

        # 创建模拟函数
        f_get_action = lambda: {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        f_get_static_data = lambda: {'key': 'value'}
        f_get_sensor_data = lambda: {}
        f_get_core_state = lambda: {'time': 0.0}
        f_get_derived_state = lambda: {}
        f_set_core_state = lambda state: None
        f_set_action = lambda action: None

        wrapper.invoke(
            invoke_type=InvokeType.POST_ACTION_STEP,
            f_get_action=f_get_action,
            f_get_static_data=f_get_static_data,
            f_get_sensor_data=f_get_sensor_data,
            f_get_core_state=f_get_core_state,
            f_get_derived_state=f_get_derived_state,
            f_set_core_state=f_set_core_state,
            f_set_action=f_set_action,
        )

        # Hook 应该被调用
        assert hook.invoke_count == 1
