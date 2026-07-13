"""Rollover ablation — Method 2: PBRS reward.

r_t = γ·φ(t) - φ(t-1)

Standard Potential-Based Reward Shaping (Ng et al. 1999).
With no base reward, total shaped return = -(1-γ)·Σφ(t), which is
negative for any non-trivial trajectory.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverPBRSExperiment(RolloverBase):
    name = "rollover_pbrs"
    reward_mode = "pbrs"


EXPERIMENT = RolloverPBRSExperiment()
