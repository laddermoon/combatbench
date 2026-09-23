"""Dose variant of `standup_floor04_ef`: explore_factor = 0.3.

σ_eff = σ × exp(0.3·ln3) ≈ σ×1.39 — the milder end of the ef dose
scan (ef=0.5 confirmed positive on s42/s1/s2).  See
exp_standup_floor04_ef.py for the mechanism and injection point.
"""
from __future__ import annotations

from .exp_standup_floor04_ef import StandupFloor04EF


class StandupFloor04EF03(StandupFloor04EF):
    name = "standup_floor04_ef03"

    EF: float = 0.3  # σ_eff = σ × exp(0.3·ln3) ≈ σ×1.39


EXPERIMENT_CLASS = StandupFloor04EF03
