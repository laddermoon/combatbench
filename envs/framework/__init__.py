from .backend import BaseSimulator, IDataAccessor, IDataMutator
from .context import ReadOnlySimContext, SimContext, TerminationReason
from .plugin import BasePlugin
from .observer_plugin import (
    OBSERVER_DISPATCHER_PRIORITY,
    BaseObserverPlugin,
    BaseRuntimeUnit,
    CompositeObserver,
)
from .recorder import BaseFrameRecorder, EpisodeBufferRecorder, PostActionRecorder
from .replay import (
    ReplayError,
    ReplayExhaustedError,
    ReplayReadOnlyError,
    ReplaySimulator,
)
from .env_runtime import EnvRuntime
from .common_plugins import TimeoutPlugin, VideoRecorderPlugin
from .policy import Policy, call_policy, coerce_action
from .episode_runner import AGENT_IDS, EpisodeRunner
from .rollout_batch import RolloutBatch
# TODO: parallel_runner / round_runner / match_runner still reference
# legacy symbols (EpisodeResult, RolloutConfig, AgentTrajectory, StepContext,
# default_bindings, default_reward_extractor, _derive_batch_seeds) that were
# removed in the EpisodeRunner refactor. They are not re-exported here so
# that ``import combatbench.envs.framework`` keeps working; they will be
# brought back to life when those modules are migrated.

__all__ = [
    "BaseSimulator",
    "IDataAccessor",
    "IDataMutator",
    "ReadOnlySimContext",
    "SimContext",
    "TerminationReason",
    "BasePlugin",
    "OBSERVER_DISPATCHER_PRIORITY",
    "BaseRuntimeUnit",
    "BaseObserverPlugin",
    "CompositeObserver",
    "BaseFrameRecorder",
    "EpisodeBufferRecorder",
    "PostActionRecorder",
    "ReplaySimulator",
    "ReplayError",
    "ReplayExhaustedError",
    "ReplayReadOnlyError",
    "EnvRuntime",
    "TimeoutPlugin",
    "VideoRecorderPlugin",
    "EpisodeRunner",
    "Policy",
    "call_policy",
    "coerce_action",
    "RolloutBatch",
    "AGENT_IDS",
]
