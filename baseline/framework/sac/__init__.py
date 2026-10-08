"""SAC V2 framework — off-policy training with tagged replay.

This package implements a SAC framework designed from the ground up to
exploit off-policy capabilities: tagged replay buffer, multi-head Q
critics, per-channel n-step returns, and action-gradient normalization.

See ``PLAN.md`` for the full design rationale and ``DECISIONS.md`` for
the implementation decision log.
"""
from __future__ import annotations

from .actor import SACActor, SACActorNotImplementedError
from .collected_episode import CollectedEpisode
from .collection import SACBehaviorSpec, SACFactSpec, SACJob
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
from .networks import MultiHeadQCritic, QTrunkGroup
from .replay import TaggedReplay
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
    "SACBehaviorSpec",
    "SACCollectionError",
    "SACFactSpec",
    "SACJob",
    "SACParallelRollouter",
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
    "sac_update_v2",
    "validate_transition_slice",
]
