from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class IDataAccessor(ABC):
    """
    数据访问器契约（只读）。
    用于获取仿真过程中的各类状态数据。
    """
    @abstractmethod
    def get_static_data(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_core_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_derived_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_action(self) -> Dict[str, Any]:
        """获取当前正在执行的动作（如果有）"""
        pass

    @abstractmethod
    def get_broadcastview_image(self) -> Any:
        """获取广播视角图像（渲染输出）"""
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """获取当前两个机器人的观测向量。

        Returns:
            {"robot_a": <obs_a>, "robot_b": <obs_b>}
            每个 value 是 policy 可直接消费的 ndarray 或容器。
        """
        pass


class IDataMutator(ABC):
    """
    数据操作器契约（可写）。
    用于修改仿真的核心状态或控制动作。
    """
    @abstractmethod
    def set_core_state(self, state: Dict[str, Any]) -> None:
        """
        设置核心状态。
        实现时必须处理底层物理引擎的缓存刷新（如正向运动学、碰撞缓存等）。
        """
        pass

    @abstractmethod
    def set_action(self, action: Dict[str, Any]) -> None:
        pass

    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a"
    ) -> None:
        """
        对指定 body 施加外力和/或外力矩

        Args:
            body_name: body 名称（如 'head', 'torso', 'hand_right'）
            force: 3D 力向量 [fx, fy, fz] (牛顿)
            torque: 可选的 3D 力矩向量 [tx, ty, tz] (牛顿·米)
            robot_id: 机器人 ID ('robot_a' 或 'robot_b')

        Note:
            默认实现为空，子类可以选择性实现以支持外部扰动功能。
            这个方法建议在 on_pre_phy_step 钩子中被调用，用于在物理步前施加外力。
        """
        pass  # 默认实现为空，子类可选实现


class BaseSimulator(IDataAccessor, IDataMutator):
    """
    底层物理仿真器的抽象契约。
    扩展了数据访问（Accessor）和操作（Mutator）能力，并提供生命周期与步进控制。
    """
    @abstractmethod
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> None:
        """重置底层物理引擎状态"""
        pass

    @abstractmethod
    def physical_step(self) -> None:
        """执行一个最细粒度的物理仿真步。"""
        pass

    @abstractmethod
    def get_physical_frequency(self) -> float:
        """获取物理仿真的运行频率（Hz）。"""
        pass

    def close(self) -> None:
        """释放资源"""
        pass
