"""
Hook 抽象类接口

Hook 模式：在仿真的特定时间点被调用，可以修改仿真状态的 Hook 接口。

与 Observer 的区别：
- Observer: 只读观察，不能修改状态
- Hook: 可以修改核心状态、动作等，返回终止标志

Hook 可以访问和修改：
- 动作 (Action): 读取和修改当前动作指令
- 核心状态 (Core State): 读取和修改广义坐标 q 和广义速度 q̇（可选）
- 静态数据 (Static Data): 只读，场景配置等
- 传感器数据 (Sensor Data): 只读，传感器读数
- 衍生状态 (Derived State): 只读，接触力等
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
from enum import Enum


class InvokeType(Enum):
    """
    Hook 调用时机类型

    对应仿真生命周期中的6个关键时间点：

    Hook 调用时序：
        PRE_EPISODE: Episode 开始前
        POST_EPISODE: Episode 结束后
        PRE_ACTION_STEP: 在 simulator.set_action 之后（为了在 Hook 中能够拿到最新的 Action），执行具体的物理步之前
        POST_ACTION_STEP: 此动作步所有的物理步结束后
        PRE_PHY_STEP: 物理步前
        POST_PHY_STEP: 物理步后
    """
    # Episode 级别钩子
    PRE_EPISODE = "pre_episode"       # 每个 Episode 开始前调用
    POST_EPISODE = "post_episode"     # 每个 Episode 结束后调用

    # Action Step 级别钩子
    PRE_ACTION_STEP = "pre_action_step"   # 在 set_action 之后，物理步循环之前调用（可访问/修改最新 Action）
    POST_ACTION_STEP = "post_action_step" # 所有物理步结束后调用（终止判定、观测构建）

    # Physics Step 级别钩子
    PRE_PHY_STEP = "pre_phy_step"     # 每个物理仿真步前调用（施加扰动等）
    POST_PHY_STEP = "post_phy_step"   # 每个物理仿真步后调用（约束执行等）


class BaseHook(ABC):
    """
    Hook 抽象基类

    Hook 在仿真的特定时间点被调用，可以访问和修改仿真状态。
    用于实现扰动、约束、终止判定、控制模式切换等功能。

    设计理念：
    - Hook 可以修改仿真状态（核心状态、动作等）
    - Hook 返回终止标志，可以提前终止 Episode
    - Hook 之间可能存在依赖关系，需要注意调用顺序
    - 如果某个 Hook 不应该修改状态，相应的 setter 函数会是 None
    """

    @property
    def name(self) -> str:
        """
        Hook 名称

        Returns:
            Hook 名称，用于日志和调试
        """
        return "base_hook"

    @property
    def priority(self) -> int:
        """
        Hook 优先级

        Returns:
            优先级，数值越大越先执行（默认为 0）
        """
        return 0

    @abstractmethod
    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_action: Callable[[], Dict[str, Any]],
        f_get_static_data: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_set_core_state: Optional[Callable[[Dict[str, Any]], None]],
        f_set_action: Optional[Callable[[Dict[str, Any]], None]],
    ) -> bool:
        """
        在指定时间点调用 Hook

        Args:
            invoke_type: 调用时机类型（6种之一）
            f_get_action: 获取当前动作的函数
                返回格式: {'robot_a': np.ndarray, 'robot_b': np.ndarray}
            f_get_static_data: 获取静态数据的函数
                返回格式参考 OpenSimulator.get_static_data()
            f_get_sensor_data: 获取传感器数据的函数
                返回格式参考 OpenSimulator.get_sensor_data()
            f_get_core_state: 获取核心状态的函数
                返回格式参考 OpenSimulator.get_core_state()
            f_get_derived_state: 获取衍生状态的函数
                返回格式参考 OpenSimulator.get_derived_state()
            f_set_core_state: 设置核心状态的函数（可选，可能为 None）
                如果为 None，表示当前时机不允许修改核心状态
                参数格式参考 OpenSimulator.set_core_state()
            f_set_action: 设置动作的函数（可选，可能为 None）
                如果为 None，表示当前时机不允许修改动作
                参数格式: {'robot_a': np.ndarray, 'robot_b': np.ndarray}

        Returns:
            bool: 终止标志
            - True: 表示应该终止当前 Episode
            - False: 继续正常执行

        注意：
        - 在使用 setter 函数前，应该检查其是否为 None
        - 修改核心状态后，物理引擎会自动更新缓存
        - 终止标志会立即生效，中断当前流程
        """
        pass

    # 可选的生命周期钩子

    def on_attach(self) -> None:
        """
        Hook 被附加到仿真器时调用

        用于初始化 Hook 状态
        """
        pass

    def on_detach(self) -> None:
        """
        Hook 从仿真器分离时调用

        用于清理资源、保存数据等
        """
        pass


class HookWrapper:
    """
    Hook 包装器

    用于管理多个 Hook，在指定时机调用所有已注册的 Hook。
    支持 Hook 优先级、时机过滤和提前终止机制。
    """

    def __init__(self):
        """初始化 Hook 包装器"""
        # 存储: (hook, priority, invoke_types)
        self._hooks: list = []

    def attach(
        self,
        hook: BaseHook,
        priority: Optional[int] = None,
        invoke_types: Optional[list[InvokeType]] = None,
    ) -> None:
        """
        附加 Hook

        Args:
            hook: Hook 实例
            priority: 优先级，数值越大越先执行（默认为 0）
                     如果为 None，则使用 hook.priority 的值
            invoke_types: 调用时机列表，只在这些时机调用此 Hook
                        如果为 None，则在所有时机都调用
        """
        # 检查是否已附加
        hook_key = id(hook)
        if hook_key in [id(h) for h, _, _ in self._hooks]:
            return

        # 如果没有指定优先级，使用 hook 的默认优先级
        if priority is None:
            priority = hook.priority

        # 如果没有指定时机，默认所有时机
        if invoke_types is None:
            invoke_types = list(InvokeType)

        # 按优先级插入
        inserted = False
        for i, (existing_hook, existing_priority, _) in enumerate(self._hooks):
            if priority > existing_priority:
                self._hooks.insert(i, (hook, priority, invoke_types))
                inserted = True
                break
        if not inserted:
            self._hooks.append((hook, priority, invoke_types))

        hook.on_attach()

    def detach(self, hook: BaseHook) -> None:
        """
        分离 Hook

        Args:
            hook: Hook 实例
        """
        hook_key = id(hook)
        for i, (existing_hook, _, _) in enumerate(self._hooks):
            if id(existing_hook) == hook_key:
                self._hooks.pop(i)
                hook.on_detach()
                break

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_action: Callable[[], Dict[str, Any]],
        f_get_static_data: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_set_core_state: Optional[Callable[[Dict[str, Any]], None]],
        f_set_action: Optional[Callable[[Dict[str, Any]], None]],
    ) -> bool:
        """
        调用所有已注册的 Hook（仅调用注册了此时机的 Hook）

        按优先级顺序调用 Hook，如果任何 Hook 返回 True，
        则立即终止并返回 True。

        Args:
            invoke_type: 调用时机类型
            f_get_action: 获取当前动作的函数
            f_get_static_data: 获取静态数据的函数
            f_get_sensor_data: 获取传感器数据的函数
            f_get_core_state: 获取核心状态的函数
            f_get_derived_state: 获取衍生状态的函数
            f_set_core_state: 设置核心状态的函数（可选）
            f_set_action: 设置动作的函数（可选）

        Returns:
            bool: 终止标志
            - 如果任何 Hook 返回 True，则返回 True
            - 否则返回 False
        """
        for hook, _, invoke_types in self._hooks:
            # 只调用注册了此时机的 Hook
            if invoke_type not in invoke_types:
                continue

            try:
                terminate = hook.invoke(
                    invoke_type,
                    f_get_action,
                    f_get_static_data,
                    f_get_sensor_data,
                    f_get_core_state,
                    f_get_derived_state,
                    f_set_core_state,
                    f_set_action,
                )
                if terminate:
                    return True
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Hook '{hook.name}' failed at {invoke_type.value}: {e}"
                )
        return False

    def clear(self) -> None:
        """清除所有 Hook"""
        for hook, _, _ in self._hooks[:]:
            self.detach(hook)

    @property
    def hooks(self) -> list:
        """获取所有 Hook 列表"""
        return [hook for hook, _, _ in self._hooks]
