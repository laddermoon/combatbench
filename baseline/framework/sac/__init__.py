"""SAC V2 framework — off-policy training with tagged replay.

This package implements a SAC framework designed from the ground up to
exploit off-policy capabilities: tagged replay buffer, multi-head Q
critics, per-channel n-step returns, and action-gradient normalization.

See ``PLAN.md`` for the full design rationale and ``DECISIONS.md`` for
the implementation decision log.
"""
from __future__ import annotations

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

__all__ = [
    "CommonParamsSAC",
    "DataSource",
    "ExperimentSAC",
    "MultiHeadQCritic",
    "QTrunkGroup",
    "ReplayPlan",
    "SACParams",
    "SACRewardChannel",
    "TaggedReplay",
    "TrajectorySlice",
    "sac_update_v2",
]
