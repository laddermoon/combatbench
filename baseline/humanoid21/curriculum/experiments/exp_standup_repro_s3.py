"""Stage 3 (r4): V1 smooth potential, entropy off, LR up.

Resume from S2 checkpoint. Disable entropy for precise control.
lr=5e-4, entropy=0.0, h_reward=5.0, term=50.0
"""
from __future__ import annotations
from typing import Any, Dict

from baseline.humanoid21.curriculum.experiments.standup_repro_base import StandupReproBase


class StandupReproS3(StandupReproBase):
    name = "standup_repro_s3"
    BLUEPRINT = "standup_repro_v1_env.yaml"

    learning_rate = 5e-4
    critic_learning_rate = 5e-4
    entropy_coef = 0.0
    log_std_min = -4.0

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 5.0,
        "height_reward_scale": 5.0,
        "terminal_success_bonus": 50.0,
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


EXPERIMENT = StandupReproS3()
