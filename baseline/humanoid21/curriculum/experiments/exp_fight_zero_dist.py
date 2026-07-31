"""Fight-Zero Distance branch: only r_distance + r_gate.

Verifies that the robot can learn to maintain striking distance from the
opponent, without any damage rewards.  Inherits everything from
FightZeroConfig and only overrides reward keys, weights, and extraction.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.exp_fight_zero import FightZeroConfig


class FightZeroDistConfig(FightZeroConfig):
    """Distance-only branch of fight_zero."""

    name = "fight_zero_dist"

    reward_keys = ("r_distance", "r_gate")
    gammas = {
        "r_distance": 0.99,
        "r_gate": 0.99,
    }

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0, 1.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._fight_ratio = float(eval_metrics.get("fight_ratio", 0.0))
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        return (1.0, 1.0)

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        full = super().extract_rewards(episode)
        return {
            "r_distance": full["r_distance"],
            "r_gate": full["r_gate"],
        }


EXPERIMENT = FightZeroDistConfig()
