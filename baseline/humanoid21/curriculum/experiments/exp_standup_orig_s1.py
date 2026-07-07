"""Stage 1 (r2) — initial standup_v2 training.

Original commit: 93bab5f
- V1 rewarder (StandupPotentialRewarder)
- pot=1.0, no height reward, no terminal bonus
- LR=3e-4, entropy=1e-3, log_std_min=-2.5
- minibatch=4096
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_orig_base import StandupOrigBase


class StandupOrigS1(StandupOrigBase):
    name = "standup_orig_s1"
    BLUEPRINT = "standup_orig_v1_env.yaml"

    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    minibatch_size: int = 4096

    DEFAULT_CUSTOM_CONFIG = {
        "max_steps": 400,
        "potential_reward_scale": 1.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 0.0,
        "time_penalty": 0.0,
        "wall_penalty": 0.0,
        "stage5_per_step_bonus": 0.0,
    }
    custom_config = DEFAULT_CUSTOM_CONFIG

    TERMINATION_PARAMS = {
        "success_height": 0.75,
        "success_uprightness": 0.85,
        "success_hold_steps": 10,
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }


EXPERIMENT = StandupOrigS1()
