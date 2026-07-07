"""Stage 1 (r2): V1 smooth potential, from scratch.

Initial standup training with weak reward signal.
pot=1.0, entropy=1e-3, lr=3e-4, log_std_min=-2.5
"""
from __future__ import annotations
from typing import Any, Dict

from baseline.humanoid21.curriculum.experiments.standup_repro_base import StandupReproBase


class StandupReproS1(StandupReproBase):
    name = "standup_repro_s1"
    BLUEPRINT = "standup_repro_v1_env.yaml"

    learning_rate = 3e-4
    critic_learning_rate = 3e-4
    entropy_coef = 1e-3
    log_std_min = -2.5

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 1.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 0.0,
        "time_penalty": 0.0,
        "wall_penalty": 0.0,
        "stage5_per_step_bonus": 0.0,
        "curriculum_phase": 0,
        "height_thresholds": [0.5, 0.3, 0.15],
        "phase_transition_success_rate": 0.5,
        "phase_transition_eval_count": 5,
    }
    custom_config = DEFAULT_CUSTOM_CONFIG

    TERMINATION_PARAMS: Dict[str, Any] = {
        "success_height": 0.75,
        "success_uprightness": 0.85,
        "success_hold_steps": 10,
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }


EXPERIMENT = StandupReproS1()
