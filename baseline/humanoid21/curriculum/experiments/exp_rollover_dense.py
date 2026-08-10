"""Rollover ablation — Method 4: Dense potential reward.

r_t = (1-γ)·φ(t)

Direct dense reward proportional to the potential.  Its discounted return
(1-γ)·Σγ^t·φ(t) equals Delta's Abel-summed discounted return, so it should
induce the same optimal policy and learning curve as Delta.

Purpose (report §7.1): verify that "Delta reward shaping" is not really
shaping at all, but merely dense potential reward in disguise.  If this
experiment's curves overlap with Delta's, that hypothesis is confirmed.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverDenseExperiment(RolloverBase):
    name = "rollover_dense"
    reward_mode = "dense_potential"


EXPERIMENT = RolloverDenseExperiment()
