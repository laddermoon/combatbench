"""Rollover ablation — Method 1: Delta reward.

r_t = φ(t) - φ(t-1)

Pure potential difference, no gamma.  Telescoping sum = φ(T) - φ(0).
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverDeltaExperiment(RolloverBase):
    name = "rollover_delta"
    reward_mode = "delta"


EXPERIMENT = RolloverDeltaExperiment()
