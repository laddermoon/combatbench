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
    # TODO: RoundRunner / MatchResult / MatchRunner are pending migration to
    # the refactored EpisodeRunner (legacy EpisodeResult / RolloutConfig
    # removed). Re-add once the migration lands.
    # Humanoid21
    MujocoCombatSimulator,
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
    # Humanoid21
    "MujocoCombatSimulator",
]
