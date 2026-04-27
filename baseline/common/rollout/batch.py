"""Compatibility shim: re-export :class:`RolloutBatch` from the framework.

The canonical definition lives in :mod:`envs.framework.rollout_batch` —
the framework owns this dataclass because it is the natural frozen view
of an :class:`envs.framework.AgentTrajectory` and is consumed by every
algorithm package. This module is kept around so existing imports
(``from baseline.common.rollout import RolloutBatch`` /
``from baseline.common.rollout.batch import RolloutBatch``) keep working.
"""
from envs.framework.rollout_batch import RolloutBatch

__all__ = ["RolloutBatch"]
