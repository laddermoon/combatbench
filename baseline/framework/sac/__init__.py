"""SAC framework — independent off-policy training implementation.

The first-version contract uses validated ``sac_transition_v1`` slices,
FIFO uniform replay, explicit sample/source identity, and SAC-specific
metrics/debug contracts.

See ``PLAN.md`` for the full design rationale and ``DECISIONS.md`` for
the implementation decision log.
"""
from __future__ import annotations

from .actor import SACActor, SACActorNotImplementedError
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
from .experiment import (
    CommonParamsSAC,
    DataSource,
    ExperimentSAC,
    ReplayPlan,
    SACParams,
    SACRewardChannel,
    TrajectorySlice,
)
from .metrics import MetricEvent, SACMetricsWriter, load_events
from .networks import MultiHeadQCritic, QTrunkGroup
from .replay import SACReplayBuffer, SACReplayError, TaggedReplay
from .trainer import sac_update_v2
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
    "SACActorNotImplementedError",
    "MetricEvent",
    "SACBehaviorSpec",
    "SACCheckpointBundle",
    "SACCheckpointError",
    "SACClockState",
    "SACCollectionError",
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
    "SACTransitionSlice",
    "SAC_TRANSITION_SCHEMA",
    "TaggedReplay",
    "TrajectorySlice",
    "build_agent_transition_slice",
    "load_checkpoint_bundle",
    "load_events",
    "load_model_only",
    "sac_update_v2",
    "save_checkpoint_bundle",
    "validate_transition_slice",
]
