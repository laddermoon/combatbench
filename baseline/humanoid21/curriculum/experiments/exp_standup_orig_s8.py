"""Stage 8 (r9/r10) — velocity-gated sustainable standing, higher terminal bonus.

Original commit: 415b0cc
- V2 rewarder (from 415b0cc, velocity-gated Stage 5)
- pot=10.0, h_reward=0.0, term=100.0, time_penalty=0.0
- LR=5e-4, entropy=0.0, log_std_min=-4.0
- Lowered success thresholds: height=0.60, uprightness=0.70, hold=50
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_orig_base import StandupOrigBase


class StandupOrigS8(StandupOrigBase):
    name = "standup_orig_s8"
    BLUEPRINT = "standup_orig_v2_r10_env.yaml"

    log_std_min: float = -4.0
    learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    entropy_coef: float = 0.0
    minibatch_size: int = 4096

    DEFAULT_CUSTOM_CONFIG = {
        "max_steps": 400,
        "potential_reward_scale": 10.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 100.0,
        "time_penalty": 0.0,
        "wall_penalty": 0.0,
        "stage5_per_step_bonus": 0.0,
    }
    custom_config = DEFAULT_CUSTOM_CONFIG

    TERMINATION_PARAMS = {
        "success_height": 0.60,
        "success_uprightness": 0.70,
        "success_hold_steps": 50,
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }


EXPERIMENT = StandupOrigS8()
