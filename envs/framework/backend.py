from abc import ABC, abstractmethod
from typing import Any, Dict

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
