"""Policy evaluation: episode-level statistics with bootstrap confidence intervals.

See ``baseline/DESIGN.md`` §3.7.
"""

from .evaluator import (
    EvalReport,
    MetricStats,
    PolicyEvaluator,
    bootstrap_ci,
    head_to_head_winrate,
)

__all__ = [
    "PolicyEvaluator",
    "EvalReport",
    "MetricStats",
    "bootstrap_ci",
    "head_to_head_winrate",
]
