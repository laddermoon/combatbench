"""Rollout-side building blocks for on-policy baselines.

See ``baseline/framework/rollout/DESIGN.md`` for the full design.
"""

from .episode import Episode, blueprint_hash
from .episode_collection import EpisodeCollection
from .episode_recorder import EpisodeRecorder
from .job import EiSpec, Job
from .observer_utils import (
    coerce_per_step,
    extract_per_step_field,
    extract_per_step_scalar,
)
from .parallel_rollouter import ParallelRollouter

__all__ = [
    "Episode",
    "EpisodeCollection",
    "EpisodeRecorder",
    "EiSpec",
    "Job",
    "ParallelRollouter",
    "blueprint_hash",
    "coerce_per_step",
    "extract_per_step_field",
    "extract_per_step_scalar",
]
