"""Running-stats normalization for observations and rewards / returns.

See ``baseline/DESIGN.md`` §3.5.
"""

from .running_mean_std import RunningMeanStd
from .normalizers import ObservationNormalizer, ReturnNormalizer

__all__ = ["RunningMeanStd", "ObservationNormalizer", "ReturnNormalizer"]
