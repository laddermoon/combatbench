"""Phase B — precise execution stage of 4-stage standup training.

Entropy disabled (0.0) for precise control, triggering confidence positive
feedback: less noise → better rollout quality → higher critic EV → stronger
advantage signal → breakthrough (same mechanism as original S3).

Resume from Phase A checkpoint via --resume-from.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_4stage_base import Standup4StageBase


class Standup4StageB(Standup4StageBase):
    name = "standup_4stage_b"
    BLUEPRINT = "standup_4stage_env.yaml"

    log_std_min: float = -4.0
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_coef: float = 0.0
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


EXPERIMENT = Standup4StageB()
