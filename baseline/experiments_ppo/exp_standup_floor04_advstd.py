"""A/B variant of `standup_floor04`: advantage normalization without
batch-mean centering.

  standup_floor04        : per-channel adv normalized as (A−μ)/σ
                           (z-score — frames near the batch mean can
                           flip sign, flipping their surrogate gradient)
  standup_floor04_advstd : per-channel adv normalized as A/σ — the raw
                           advantage sign of every frame is preserved
                           (this run)

Same seed=42 and all other params as `standup_floor04`.  A/B pair
against train_standup_floor04_ppo_20260920_164819 to test whether
z-score centering disperses the per-frame gradient signal.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04AdvStd(StandupFloor04):
    name = "standup_floor04_advstd"

    adv_norm: str = "std"


EXPERIMENT_CLASS = StandupFloor04AdvStd
