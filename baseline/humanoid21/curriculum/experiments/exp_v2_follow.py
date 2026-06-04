"""V2 experiment: 6-reward curriculum (r_fall, r_cross, r_damage, r_hold, r_radial, r_tangential).

Translates the reward extraction and weight scheduling from
``train_curriculum_v2.py`` into the ExperimentConfig interface.

Uses follow_opponent's ``compute_approach_rewards`` for r_radial/r_tangential
(trainer-side post-processing from recorded position trajectories).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import (
    _extract_per_step_field,
    _extract_per_step_scalar,
)


class V2FollowConfig(ExperimentConfig):
    """V2: 6-reward curriculum using follow_opponent approach signal."""

    name = "v2_follow"
    reward_keys = (
        "r_fall", "r_cross", "r_damage",
        "r_hold", "r_radial", "r_tangential",
    )
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_damage": 0.80,
        "r_hold": 0.98,
        "r_radial": 0.97,
        "r_tangential": 0.95,
    }
    env_blueprint = "curriculum_env_v2.yaml"
    ppo_overrides: Dict[str, Any] = {}

    # Terminal fall penalty (set by training loop before buffer construction).
    terminal_fall_penalty: float = 1.0

    # Stateful scheduler
    _phase: str = "balance"
    _consecutive_pass: int = 0

    def initial_weights(self) -> Tuple[float, ...]:
        return (3.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        len_ratio = eval_metrics.get("mean_length", 0.0) / 200.0
        if len_ratio < 0.98:
            self._phase = "balance"
            self._consecutive_pass = 0
            return (3.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        elif eval_metrics.get("in_zone", 0.0) < 0.5:
            self._phase = "approach"
            self._consecutive_pass += 1
            return (2.0, 1.0, 0.0, 0.5, 0.5, 0.0)
        else:
            self._phase = "combat"
            self._consecutive_pass += 1
            return (2.0, 1.0, 1.0, 0.5, 0.5, 0.0)

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        fell = "imbalance" in termination_proposals
        r_fall = np.zeros(T, dtype=np.float32)
        if fell and self.terminal_fall_penalty > 0.0:
            r_fall[-1] = -float(self.terminal_fall_penalty)

        r_cross = _extract_per_step_scalar(observer_outputs, "cross_support", T)
        r_damage = _extract_per_step_scalar(observer_outputs, "damage", T)

        # r_hold from in_zone_hold observer
        r_hold = _extract_per_step_field(observer_outputs, "in_zone_hold", "reward", T)
        if r_hold is None:
            r_hold = np.zeros(T, dtype=np.float32)

        # r_radial / r_tangential: trainer-side post-processing
        # from approach_velocity observer's recorded positions
        from baseline.humanoid21.rewards.follow_opponent import compute_approach_rewards

        self_x = _extract_per_step_field(observer_outputs, "approach_velocity", "self_x", T)
        self_y = _extract_per_step_field(observer_outputs, "approach_velocity", "self_y", T)
        opp_x = _extract_per_step_field(observer_outputs, "approach_velocity", "opp_x", T)
        opp_y = _extract_per_step_field(observer_outputs, "approach_velocity", "opp_y", T)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T, dtype=np.float32)
            r_tangential = np.zeros(T, dtype=np.float32)
        else:
            self_xy = np.stack([self_x, self_y], axis=1)
            opp_xy = np.stack([opp_x, opp_y], axis=1)
            r_radial, r_tangential = compute_approach_rewards(
                self_xy, opp_xy,
                debug=False,
            )

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_damage": r_damage,
            "r_hold": r_hold,
            "r_radial": r_radial,
            "r_tangential": r_tangential,
        }

    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, float]:
        hold_in_zone = _extract_per_step_field(
            observer_outputs, "in_zone_hold", "in_zone", T
        )
        in_zone = 0.0
        if hold_in_zone is not None:
            in_zone = float(np.any(hold_in_zone > 0.5))
        return {"in_zone": in_zone}

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "phase": self._phase,
            "consecutive_pass": self._consecutive_pass,
        }

    def scheduler_state(self) -> dict:
        return {
            "phase": self._phase,
            "consecutive_pass": self._consecutive_pass,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._phase = state.get("phase", "balance")
        self._consecutive_pass = state.get("consecutive_pass", 0)


# Singleton instance for the registry
EXPERIMENT = V2FollowConfig()
