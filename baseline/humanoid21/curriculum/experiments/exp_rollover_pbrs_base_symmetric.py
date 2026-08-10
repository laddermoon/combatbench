"""Rollover ablation — Method 3B: PBRS + base reward (symmetric ±1.0).

r_t = [γ·φ(t) - φ(t-1)] + base_r_t

Base reward:
  - Each time φ crosses 0.97 upward: +1.0
  - Each time φ crosses 0.97 downward: -1.0
  - Each step φ ≥ 0.97: +0.01 (maintain reward)

Symmetric penalty prevents re-entry farming. Higher reward variance
may cause PPO instability — this is the expected trade-off vs 3A.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverPBRSBaseSymmetricExperiment(RolloverBase):
    name = "rollover_pbrs_base_symmetric"
    reward_mode = "pbrs_base_symmetric"


EXPERIMENT = RolloverPBRSBaseSymmetricExperiment()
