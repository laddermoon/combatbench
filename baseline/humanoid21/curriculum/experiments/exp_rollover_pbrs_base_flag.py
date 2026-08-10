"""Rollover ablation — Method 3A: PBRS + base reward (flag-based one-time bonus).

r_t = [γ·φ(t) - φ(t-1)] + base_r_t

Base reward:
  - First time φ crosses 0.97 (up): +1.0 (once per episode, flag-gated)
  - Each step φ ≥ 0.97: +0.01 (maintain reward, cancels PBRS maintenance tax)
  - Otherwise: 0

Validates that PBRS works well when a proper base reward exists.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverPBRSBaseFlagExperiment(RolloverBase):
    name = "rollover_pbrs_base_flag"
    reward_mode = "pbrs_base_flag"


EXPERIMENT = RolloverPBRSBaseFlagExperiment()
