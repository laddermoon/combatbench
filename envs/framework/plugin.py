from typing import Any, Dict, List, Optional
from .context import SimContext

class BasePlugin:
    """
    仿真扩展插件（Simulation Plugin）。
    
    职责：
    1. 在仿真引擎的特定生命周期节点注入自定义逻辑。
    2. 通过 ctx.accessor 读取数据。
    3. 通过 ctx.mutator 修改数据（如果当前生命周期被允许，否则为 None）。
    """
    
    @property
    def name(self) -> str:
        return "unnamed_plugin"
        
    @property
    def priority(self) -> int:
        return 0

    @property
    def require_mutator(self) -> bool:
        """
        是否申请数据操作权限。
        如果为 False，即使处于允许修改的生命周期（如 on_pre_phy_step），
        传给此插件的 ctx.mutator 依然会是 None。遵循最小权限原则。
        """
        return False

    # ==========================================
    # 生命周期钩子 (Lifecycle Hooks)
    # ==========================================

    def on_pre_episode(self, ctx: SimContext) -> None:
        """
        [时机]: 在一个新的 Episode 开始前，环境刚刚被重置时。
        [职责]: 初始化状态（Resetter）、清空历史统计。
        [权限]: 读写（ctx.mutator 可用）。
        """
        pass

    def on_pre_action_step(self, ctx: SimContext) -> None:
        """
        [时机]: 接收到外部动作（Action），但还未拆分并下发给物理引擎之前。
        [职责]: 控制模式映射（Control Modes）、动作空间映射、动作限幅。
        [权限]: 读写（ctx.mutator 可用，可修改动作）。
        """
        pass

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """
        [时机]: 在每一次极细粒度的物理仿真步（Physical Step）执行之前。
        [职责]: 注入外部扰动（Disturbances）。
        [权限]: 读写（ctx.mutator 可用，可修改外部力）。
        """
        pass

    def on_post_phy_step(self, ctx: SimContext) -> None:
        """
        [时机]: 在每一次极细粒度的物理仿真步执行之后，且状态已更新。
        [职责]: 刚性状态约束（Constraints），高频数据收集。
        [权限]: 读写（ctx.mutator 可用，可强行覆盖状态实现投影）。
        """
        pass

    def on_post_action_step(self, ctx: SimContext) -> None:
        """
        [时机]: 一个 Action Step 对应的所有物理步都执行完毕之后。
        [职责]: 指标聚合（Metrics）、犯规判定（Terminations）、计算奖励。
        [权限]: 只读（ctx.mutator 为 None）。
        """
        pass

    def on_post_episode(self, ctx: SimContext) -> None:
        """
        [时机]: Episode 确定终止之后。
        [职责]: 整局维度的日志汇总、数据上报。
        [权限]: 只读（ctx.mutator 为 None）。
        """
        pass

    # ==========================================
    # 管理钩子 (Management Hooks)
    # ==========================================

    def on_attach(self) -> None:
        pass

    def on_detach(self) -> None:
        pass
