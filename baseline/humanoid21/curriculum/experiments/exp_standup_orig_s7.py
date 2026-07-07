"""Stage 7 (r8) — increase potential scale to 10, reduce time penalty.

Original commit: 2e06734
- V2 rewarder (from f598dfa, no wall detection yet)
- pot=10.0, h_reward=0.0, term=50.0, time_penalty=-0.005
- LR=5e-4, entropy=0.0, log_std_min=-4.0
- Same termination as V1
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_orig_base import StandupOrigBase


class StandupOrigS7(StandupOrigBase):
    name = "standup_orig_s7"
    BLUEPRINT = "standup_orig_v2_r7_env.yaml"

    log_std_min: float = -4.0
    learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    entropy_coef: float = 0.0
    minibatch_size: int = 4096

    DEFAULT_CUSTOM_CONFIG = {
        "max_steps": 400,
        "potential_reward_scale": 10.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 50.0,
        "time_penalty": -0.005,
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


EXPERIMENT = StandupOrigS7()
