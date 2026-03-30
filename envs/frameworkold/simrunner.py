"""
SimRunner - 仿真运行器

框架类，整合 OpenSimulator 和 Hooks 形成完整的仿真循环。

设计理念：
- 这是一个纯工具架构类，不负责实现 Gym 接口
- 没有 Reward 概念（Reward 由 Hook 实现）
- step() 无返回值，所有数据处理通过 Hook 实现
- 每个 Hook 注明其 Invoke 时机

Hook 调用时序：
```
Episode:
  PRE_EPISODE     → 重置 Hook
  └─ Action Loop:
       simulator.set_action(action)  # 设置动作
       PRE_ACTION_STEP  → 动作 Hook（在 set_action 之后，可访问/修改最新 Action）
       └─ Physics Loop (N 次):
            PRE_PHY_STEP    → 扰动 Hook
            physical_step()
            POST_PHY_STEP   → 约束 Hook
       POST_ACTION_STEP → 终止判定/观测构建 Hook（所有物理步结束后）
  POST_EPISODE    → 清理 Hook
```
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np

from .open_simulator import OpenSimulator
from .base_hook import BaseHook, HookWrapper, InvokeType


class SimRunner:
    """
    仿真运行器

    整合 OpenSimulator 和 Hook，提供完整的仿真循环功能。

    这不是 Gym 环境，不提供 reward，也不实现 Gym 接口。
    所有数据处理（观测、奖励、终止等）都通过 Hook 实现。

    Hook 调用时序：
    - PRE_EPISODE: Episode 开始前
    - POST_EPISODE: Episode 结束后
    - PRE_ACTION_STEP: 在simulator.set_action之后（为了在Hook中能够拿到最新的Action）， 执行具体的物理步之前。
    - POST_ACTION_STEP: 此动作步所有的物理步结束后
    - PRE_PHY_STEP: 物理步前
    - POST_PHY_STEP: 物理步后
    """

    def __init__(
        self,
        simulator: OpenSimulator,
        phy_steps_per_action: int = 25,
        video_fps: int = 30,
        video_frame_receiver: Optional[callable] = None,
    ):
        """
        初始化仿真运行器

        Args:
            simulator: OpenSimulator 实例
            phy_steps_per_action: 每个动作步执行的物理步数
            video_fps: 视频帧率
            video_frame_receiver: 接收视频帧的回调方法，如果为 None 则使用内部缓冲区
        """
        self.simulator = simulator
        self.phy_steps_per_action = phy_steps_per_action

        # Hook 管理器
        self._hook_wrapper = HookWrapper()

        # 视频录制
        self._video_fps = video_fps
        self._video_frame_receiver = video_frame_receiver
        self._physics_step_count = 0
        self._video_sample_interval = 1

        # 状态
        self._is_episode_active = False

        # 缓存数据
        self._cached_static_data: Optional[Dict[str, Any]] = None

    # ==================== Hook 管理 ====================

    @property
    def hooks(self) -> List[BaseHook]:
        """获取所有已附加的 Hook"""
        return self._hook_wrapper.hooks

    def attach_hook(
        self,
        hook: BaseHook,
        priority: int = 0,
        invoke_types: Optional[list[InvokeType]] = None,
    ) -> None:
        """
        附加 Hook

        Args:
            hook: Hook 实例
            priority: 优先级，数值越大越先执行
            invoke_types: 调用时机列表，只在这些时机调用此 Hook
                        如果为 None，则在所有时机都调用
        """
        self._hook_wrapper.attach(hook, priority, invoke_types)

    def detach_hook(self, hook: BaseHook) -> None:
        """
        分离 Hook

        Args:
            hook: Hook 实例
        """
        self._hook_wrapper.detach(hook)

    def clear_hooks(self) -> None:
        """清除所有 Hook"""
        self._hook_wrapper.clear()

    # ==================== 仿真循环 ====================

    def reset(self) -> None:
        """
        重置仿真

        调用流程：
        1. 重置仿真器状态
        2. 调用 PRE_EPISODE Hook（用于重置状态）
        3. 如果 Hook 返回终止，调用 POST_EPISODE Hook

        无返回值，所有数据通过 Hook 处理。
        """
        self._is_episode_active = True
        self._physics_step_count = 0

        # 重置仿真器
        self.simulator.reset()

        # 获取静态数据
        self._cached_static_data = self.simulator.get_static_data()

        # 计算视频采样间隔
        physical_frequency = self.simulator.get_physical_frequency()
        if self._video_fps > 0:
            self._video_sample_interval = max(1, int(round(physical_frequency / self._video_fps)))
        else:
            self._video_sample_interval = 1

        # 调用 PRE_EPISODE Hook（重置状态）
        terminate = self._invoke_hooks(InvokeType.PRE_EPISODE)

        if terminate:
            self._is_episode_active = False
            self._invoke_hooks(InvokeType.POST_EPISODE)

    def step(self, action: Dict[str, Any]) -> None:
        """
        执行一个动作步

        调用流程：
        1. 设置动作到仿真器（set_action）
        2. PRE_ACTION_STEP Hook（在 set_action 之后，可访问/修改最新 Action）
        3. 执行 N 个物理步：
           - PRE_PHY_STEP Hook（施加扰动）
           - physical_step()
           - POST_PHY_STEP Hook（执行约束）
        4. POST_ACTION_STEP Hook（所有物理步结束后，终止判定、观测构建）

        无返回值，所有数据处理通过 Hook 实现。

        Args:
            action: 动作字典 {'robot_a': np.ndarray, 'robot_b': np.ndarray}
        """
        if not self._is_episode_active:
            return

        # 1. 设置动作到仿真器（供 PRE_ACTION_STEP Hook 访问）
        self.simulator.set_action(action)

        # 2. 调用 PRE_ACTION_STEP Hook（在 set_action 之后，可访问/修改最新 Action）
        terminate = self._invoke_hooks(InvokeType.PRE_ACTION_STEP)
        if terminate:
            self._is_episode_active = False
            self._invoke_hooks(InvokeType.POST_EPISODE)
            return

        # 3. 执行多个物理步
        for _ in range(self.phy_steps_per_action):
            # PRE_PHY_STEP Hook（施加扰动）
            terminate = self._invoke_hooks(InvokeType.PRE_PHY_STEP)
            if terminate:
                self._is_episode_active = False
                self._invoke_hooks(InvokeType.POST_EPISODE)
                return

            # 物理步进（动作已在循环前设置，PD 控制在 set_action 中计算）
            self.simulator.physical_step()
            self._physics_step_count += 1

            # POST_PHY_STEP Hook（执行约束）
            terminate = self._invoke_hooks(InvokeType.POST_PHY_STEP)
            if terminate:
                self._is_episode_active = False
                self._invoke_hooks(InvokeType.POST_EPISODE)
                return

            # 视频帧采集（在 POST_phy_step 之后）
            if self._video_frame_receiver is not None and self._physics_step_count % self._video_sample_interval == 0:
                frame = self.simulator.get_broadcastview_image()
                self._video_frame_receiver(frame)

        # 4. 调用 POST_ACTION_STEP Hook（所有物理步结束后，终止判定、观测构建）
        terminate = self._invoke_hooks(InvokeType.POST_ACTION_STEP)
        if terminate:
            self._is_episode_active = False
            self._invoke_hooks(InvokeType.POST_EPISODE)

    def close(self) -> None:
        """
        关闭仿真运行器

        分离所有 Hook 并关闭仿真器。
        """
        self._hook_wrapper.clear()
        self._is_episode_active = False
        if hasattr(self.simulator, 'close'):
            self.simulator.close()

    # ==================== 内部方法 ====================

    def _invoke_hooks(self, invoke_type: InvokeType) -> bool:
        """
        调用所有 Hook

        根据调用时机决定哪些 setter 可用：
        - PRE_ACTION_STEP: f_set_action 可用
        - PRE_PHY_STEP / POST_PHY_STEP: f_set_core_state 可用

        Args:
            invoke_type: 调用时机

        Returns:
            bool: 是否应该终止
        """
        # 根据调用时机决定哪些 setter 可用
        f_set_action = self._f_set_action if invoke_type == InvokeType.PRE_ACTION_STEP else None
        f_set_core_state = self._f_set_core_state if invoke_type in [
            InvokeType.PRE_ACTION_STEP,
            InvokeType.PRE_PHY_STEP,
            InvokeType.POST_PHY_STEP,
        ] else None

        return self._hook_wrapper.invoke(
            invoke_type=invoke_type,
            f_get_action=self._f_get_action,
            f_get_static_data=self._f_get_static_data,
            f_get_sensor_data=self._f_get_sensor_data,
            f_get_core_state=self._f_get_core_state,
            f_get_derived_state=self._f_get_derived_state,
            f_set_core_state=f_set_core_state,
            f_set_action=f_set_action,
        )

    # ==================== 数据获取函数（供 Hook 使用）====================

    def _f_get_action(self) -> Dict[str, Any]:
        """获取当前动作（直接从 simulator 获取）"""
        if hasattr(self.simulator, 'get_action'):
            return self.simulator.get_action()
        # 兼容没有 get_action 方法的 simulator
        return {"robot_a": np.zeros(21), "robot_b": np.zeros(21)}

    def _f_get_static_data(self) -> Dict[str, Any]:
        """获取静态数据"""
        if self._cached_static_data is None:
            self._cached_static_data = self.simulator.get_static_data()
        return self._cached_static_data

    def _f_get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据"""
        return self.simulator.get_sensor_data()

    def _f_get_core_state(self) -> Dict[str, Any]:
        """获取核心状态"""
        return self.simulator.get_core_state()

    def _f_get_derived_state(self) -> Dict[str, Any]:
        """获取衍生状态"""
        return self.simulator.get_derived_state()

    def _f_set_core_state(self, state: Dict[str, Any]) -> None:
        """设置核心状态"""
        self.simulator.set_core_state(state)

    def _f_set_action(self, action: Dict[str, Any]) -> None:
        """设置动作"""
        # 立即应用到仿真器（Hook 修改动作后需要重新设置）
        self.simulator.set_action(action)

    # ==================== 便捷访问方法 ====================

    def get_core_state(self) -> Dict[str, Any]:
        """获取核心状态（可读可写）"""
        return self._f_get_core_state()

    def set_core_state(self, state: Dict[str, Any]) -> None:
        """设置核心状态"""
        self._f_set_core_state(state)

    def get_derived_state(self) -> Dict[str, Any]:
        """获取衍生状态（只读）"""
        return self._f_get_derived_state()

    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据（只读）"""
        return self._f_get_sensor_data()

    def get_static_data(self) -> Dict[str, Any]:
        """获取静态数据"""
        return self._f_get_static_data()

    def get_broadcastview_image(self) -> np.ndarray:
        """获取广播视角图像"""
        return self.simulator.get_broadcastview_image()

    # ==================== 属性 ====================

    @property
    def video_fps(self) -> int:
        """获取视频帧率"""
        return self._video_fps

    @video_fps.setter
    def video_fps(self, value: int) -> None:
        """设置视频帧率"""
        self._video_fps = max(1, int(value))

    @property
    def video_enabled(self) -> bool:
        """视频录制是否启用"""
        return self._video_frame_receiver is not None

    @video_enabled.setter
    def video_enabled(self, value: bool) -> None:
        """设置视频录制开关（已弃用，请使用 video_frame_receiver 参数）"""
        # 此 setter 保留用于向后兼容，但不实际修改行为
        pass

    @property
    def is_episode_active(self) -> bool:
        """Episode 是否活跃"""
        return self._is_episode_active
