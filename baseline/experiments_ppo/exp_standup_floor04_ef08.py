"""Dose variant of `standup_floor04_ef`: explore_factor = 0.8.

σ_eff = σ × exp(0.8·ln3) ≈ σ×2.41 — the aggressive end of the ef
dose scan.  Tests whether the ef=0.5 gain is still rising at wider
exploration or whether dilution/saturation kicks in.
"""
from __future__ import annotations

from .exp_standup_floor04_ef import StandupFloor04EF


class StandupFloor04EF08(StandupFloor04EF):
    name = "standup_floor04_ef08"

    EF: float = 0.8  # σ_eff = σ × exp(0.8·ln3) ≈ σ×2.41


EXPERIMENT_CLASS = StandupFloor04EF08
