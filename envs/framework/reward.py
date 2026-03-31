import abc
from typing import Dict, Any

class BaseRewardFunction(abc.ABC):
    """
    奖励计算抽象基类 (Reward Function)。
    
    定位：将底层沙盒输出的客观指标（metrics）和事件（events）转化为强化学习算法需要的主观标量奖励（reward）。
    这是一个纯函数式的接口，不应包含或修改环境的状态。
    """
    
    def reset(self) -> None:
        """
        在每个 episode 开始时调用，用于重置内部状态（如时间步衰减、阶段性课程状态等）。
        大部分简单的马尔可夫奖励函数不需要实现此方法。
        """
        pass

    @abc.abstractmethod
    def compute_reward(
        self, 
        agent_id: str, 
        prev_info: Dict[str, Any], 
        curr_info: Dict[str, Any]
    ) -> float:
        """
        计算特定智能体在当前步的增量奖励。
        
        Args:
            agent_id: 需要计算奖励的智能体ID（例如 'robot_a' 或 'robot_b'）
            prev_info: 上一步的环境 info 字典（至少包含 'metrics'）
            curr_info: 当前步的环境 info 字典（至少包含 'metrics' 和 'events'）
            
        Returns:
            计算得到的标量奖励 (float)
        """
        pass


class NullRewardFunction(BaseRewardFunction):
    """
    默认的空奖励函数，始终返回 0.0。
    用于不需要环境提供奖励（如纯模仿学习）的场景。
    """
    def compute_reward(
        self, 
        agent_id: str, 
        prev_info: Dict[str, Any], 
        curr_info: Dict[str, Any]
    ) -> float:
        return 0.0
