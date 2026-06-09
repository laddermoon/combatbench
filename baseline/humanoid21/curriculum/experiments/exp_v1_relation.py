"""V1 experiment: 4-reward curriculum (r_fall, r_cross, r_relation, r_damage).

Translates the reward extraction and weight scheduling from
``train_curriculum.py`` into the ExperimentConfig interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import (
    _extract_per_step_field,
    _extract_per_step_scalar,
)
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class V1RelationConfig(ExperimentConfig):
    """V1: 4-reward curriculum using opponent_relation for approach signal."""

    name = "v1_relation"
    reward_keys = ("r_fall", "r_cross", "r_relation", "r_damage")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_relation": 0.98,
        "r_damage": 0.80,
    }
    BLUEPRINT = "curriculum_env.yaml"

    def video_env_blueprint(self):
        return self._make_video_blueprint(self._env_pb())

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def build_rollout_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.eval_episodes)

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("in_zone", 0.0) > best_esum.get("in_zone", 0.0)

    def initial_weights(self) -> Tuple[float, ...]:
        return (3.0, 1.0, 0.3, 0.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        len_ratio = eval_metrics.get("mean_length", 0.0) / 200.0
        if len_ratio < 0.98:
            return (3.0, 1.0, 0.3, 0.0)
        elif eval_metrics.get("in_zone", 0.0) < 0.5:
            return (2.0, 1.0, 1.0, 0.0)
        else:
            return (2.0, 1.0, 1.0, 1.0)

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        fell = "imbalance" in termination_proposals
        r_fall = np.zeros(T, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell and penalty > 0.0:
            r_fall[-1] = -penalty

        r_cross = _extract_per_step_scalar(observer_outputs, "cross_support", T)
        # opponent_relation emits a dict {"reward": ..., "in_zone": ...}
        r_relation = _extract_per_step_field(
            observer_outputs, "opponent_relation", "reward", T
        )
        if r_relation is None:
            r_relation = _extract_per_step_scalar(
                observer_outputs, "opponent_relation", T
            )
        r_damage = _extract_per_step_scalar(observer_outputs, "damage", T)
        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_relation": r_relation,
            "r_damage": r_damage,
        }

    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, float]:
        rel_in_zone = _extract_per_step_field(
            observer_outputs, "opponent_relation", "in_zone", T
        )
        in_zone = 0.0
        if rel_in_zone is not None:
            in_zone = float(np.any(rel_in_zone > 0.5))
        return {"in_zone": in_zone}

    def scheduler_info(self) -> Dict[str, Any]:
        return {}


# Singleton instance for the registry
EXPERIMENT = V1RelationConfig()
