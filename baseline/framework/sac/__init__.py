"""SAC framework — independent off-policy training implementation.

The first-version contract uses validated ``sac_transition_v2`` slices,
FIFO uniform replay, explicit sample/source identity, and SAC-specific
metrics/debug contracts.

See ``PLAN.md`` for the full design rationale and ``DECISIONS.md`` for
the implementation decision log.
"""
from __future__ import annotations

from .actor import SACActor
from .collected_episode import CollectedEpisode
from .collection import SACBehaviorSpec, SACFactSpec, SACJob
from .checkpoint import (
    SACCheckpointBundle,
    SACCheckpointError,
    load_checkpoint_bundle,
    load_model_only,
    save_checkpoint_bundle,
)
from .clocks import SACClockState
from .collection_rollouter import SACCollectionError, SACParallelRollouter
from .debugkit import (
    SACDumpError,
    find_sample,
    recompute_dump,
    summarize_dump,
)
from .experiment import (
    CommonParamsSAC,
    DataSource,
    ExperimentSAC,
    ReplayPlan,
    SACParams,
    SACRewardChannel,
)
from .metrics import MetricEvent, SACMetricsWriter, load_events
from .networks import MultiHeadQCritic, QTrunkGroup
from .replay import SACReplayBuffer, SACReplayError
from .s01_actor import S01Actor, S01RuntimePolicy
from .tn_actor import ARCH_SPECS as TN_ARCH_SPECS, TNActor, TNRuntimePolicy
from .trainer import SACTrainerError, sac_update, sac_update_v2
from .transition import (
    SAC_TRANSITION_SCHEMA,
    SACTransitionSlice,
    build_agent_transition_slice,
    validate_transition_slice,
)

__all__ = [
    "CommonParamsSAC",
    "CollectedEpisode",
    "SACActor",
    "MetricEvent",
    "S01Actor",
    "S01RuntimePolicy",
    "SACBehaviorSpec",
    "SACCheckpointBundle",
    "SACCheckpointError",
    "SACClockState",
    "SACCollectionError",
    "SACDumpError",
    "SACFactSpec",
    "SACJob",
    "SACMetricsWriter",
    "SACParallelRollouter",
    "SACReplayBuffer",
    "SACReplayError",
    "DataSource",
    "ExperimentSAC",
    "MultiHeadQCritic",
    "QTrunkGroup",
    "ReplayPlan",
    "SACParams",
    "SACRewardChannel",
    "SACTrainerError",
    "SACTransitionSlice",
    "SAC_TRANSITION_SCHEMA",
    "TN_ARCH_SPECS",
    "TNActor",
    "TNRuntimePolicy",
    "build_agent_transition_slice",
    "find_sample",
    "load_checkpoint_bundle",
    "load_events",
    "load_model_only",
    "sac_update",
    "sac_update_v2",
    "recompute_dump",
    "save_checkpoint_bundle",
    "summarize_dump",
    "validate_transition_slice",
]
