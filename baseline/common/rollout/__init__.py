"""Rollout-side building blocks for on-policy baselines.

See ``baseline/DESIGN.md`` §3.3 for the scope of this package.
"""

from .batch import RolloutBatch

__all__ = ["RolloutBatch"]
