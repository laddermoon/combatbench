"""
CombatBench - 多智能体格斗仿真环境
"""

# 新框架模块
from .envs import (
    BaseSimulator,
    IDataAccessor,
    IDataMutator,
    ReadOnlySimContext,
    SimContext,
    TerminationReason,
    BasePlugin,
    BaseRuntimeUnit,
    BaseObserverPlugin,
    EnvRuntime,
    TimeoutPlugin,
    VideoRecorderPlugin,
    RoundRunner,
    MatchResult,
    MatchRunner,
    # Humanoid21
    MujocoCombatSimulator,
    Humanoid21Observer,
    make_env,
)

__all__ = [
    # 框架核心
    "BaseSimulator",
    "IDataAccessor",
    "IDataMutator",
    "ReadOnlySimContext",
    "SimContext",
    "TerminationReason",
    "BasePlugin",
    "BaseRuntimeUnit",
    "BaseObserverPlugin",
    "EnvRuntime",
    "TimeoutPlugin",
    "VideoRecorderPlugin",
    "RoundRunner",
    "MatchResult",
    "MatchRunner",
    # Humanoid21
    "MujocoCombatSimulator",
    "Humanoid21Observer",
    "make_env",
]
