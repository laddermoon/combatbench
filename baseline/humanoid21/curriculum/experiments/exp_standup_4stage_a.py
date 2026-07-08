"""Phase A — exploration stage of 4-stage standup training.

High entropy (1e-3) to explore roll-over, hand+foot support, and push-up paths.
Thresholds from original S8/S9 final working config.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_4stage_base import Standup4StageBase


class Standup4StageA(Standup4StageBase):
    name = "standup_4stage_a"
    BLUEPRINT = "standup_4stage_env.yaml"

    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    minibatch_size: int = 4096

    DEFAULT_CUSTOM_CONFIG = {
        "max_steps": 400,
        "potential_reward_scale": 1.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 100.0,
        "time_penalty": -0.01,
        "wall_penalty": -0.05,
        "stage4_per_step_bonus": 0.1,
    }
    custom_config = DEFAULT_CUSTOM_CONFIG

    TERMINATION_PARAMS = {
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }


EXPERIMENT = Standup4StageA()
