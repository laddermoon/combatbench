from typing import Any, Dict, List, Optional
from .backend import BaseSimulator, IDataAccessor, IDataMutator

class TerminationReason:
    """定义常见的终止原因"""
    TIMEOUT = "timeout"
    KO = "ko"
    FOUL = "foul"
    OUT_OF_BOUNDS = "out_of_bounds"
    CUSTOM = "custom"


class SimContext:
    """
    仿真引擎的统一上下文（黑板模式 Blackboard）。
    
    职责：
    1. 提供给插件的数据访问器 (accessor) 和 数据操作器 (mutator)。
       如果 mutator 为 None，表示当前 Hook 时机不允许修改数据。
    2. 承载跨插件流转的派生指标 (metrics)、事件 (events) 和控制流信号。
    """
    def __init__(self, simulator: BaseSimulator):
        self._simulator = simulator
        
        # 内部时序状态
        self.episode_step: int = 0
        self.physics_step: int = 0

        # 派生黑板
        self.metrics: Dict[str, Any] = {}
        self.events: List[Any] = []
        self.termination_proposals: List[str] = []

        # 挂载底层引擎作为默认的 accessor
        self.accessor: IDataAccessor = simulator
        # 默认不暴露 mutator，引擎会在特定 Hook 调用前临时赋予
        self.mutator: Optional[IDataMutator] = None

    def request_termination(self, reason: str = TerminationReason.CUSTOM) -> None:
        """提出终止请求"""
        self.termination_proposals.append(reason)

    @property
    def is_terminated(self) -> bool:
        """判断是否已经收到终止请求"""
        return len(self.termination_proposals) > 0

    def clear_episode_state(self) -> None:
        """在 Episode 开始前清理历史状态"""
        self.episode_step = 0
        self.physics_step = 0
        self.metrics.clear()
        self.events.clear()
        self.termination_proposals.clear()

    # --- 引擎控制权限的辅助方法 ---
    def _grant_mutator(self) -> None:
        self.mutator = self._simulator

    def _revoke_mutator(self) -> None:
        self.mutator = None
