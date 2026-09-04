"""Standup with TruncatedNormalPolicy (action-space truncated normal).

Same reward/truncation/eval logic as Standup, but uses the
TruncatedNormalPolicy instead of the baseline TanhGaussianMLPPolicy.

Key differences from the baseline:
- Distribution is defined directly on [-1, 1] (truncated normal),
  no pre-tanh / tanh-transform indirection.
- Uncertainty U = 1/(2×peak) replaces normalized entropy:
  natural [0,1], 0 = deterministic, 1 = uniform.
- explore_intensity scales σ directly (piecewise-linear: 1/3 to 3×).
- No checkpoint compatibility with TanhGaussianMLPPolicy.

See DESIGN_truncated_normal.md for full design rationale.
"""
from __future__ import annotations

from .exp_standup import Standup


class StandupTruncNorm(Standup):

    name = "standup_truncnorm"
    actor_blueprint = "init_policy_truncated_normal.yaml"

    # TruncatedNormalPolicy has no log_std_min/log_std_max attributes
    # (no pre-tanh entropy normalization). The base class build_actor
    # guards with hasattr, so these are simply ignored.


EXPERIMENT_CLASS = StandupTruncNorm
