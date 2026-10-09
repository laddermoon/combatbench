"""Rollout-side building blocks for on-policy baselines.

See ``baseline/framework/rollout/README.md`` for the module overview.
"""

from .episode import Episode, blueprint_hash
from .episode_collection import EpisodeCollection
from .episode_recorder import EpisodeRecorder
from .exploratory_policy import ExploratoryPolicy, SamplingPolicy
from .job import EfSpec, Job, ReferenceSpec, SamplingSpec
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
    "EfSpec",
    "ExploratoryPolicy",
    "Job",
    "ParallelRollouter",
    "ReferenceSpec",
    "SamplingPolicy",
    "SamplingSpec",
    "blueprint_hash",
    "coerce_per_step",
    "extract_per_step_field",
    "extract_per_step_scalar",
]
