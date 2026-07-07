"""Stage 2 (r3) — amplify reward signal, lower entropy.

Original commit: 1594b36
- V1 rewarder
- pot=5.0, h_reward=2.0, term=10.0
- LR=3e-4, entropy=5e-4, log_std_min=-4.0
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_orig_base import StandupOrigBase


class StandupOrigS2(StandupOrigBase):
    name = "standup_orig_s2"
    BLUEPRINT = "standup_orig_v1_env.yaml"

    log_std_min: float = -4.0
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_coef: float = 5e-4
    minibatch_size: int = 4096

    DEFAULT_CUSTOM_CONFIG = {
        "max_steps": 400,
        "potential_reward_scale": 5.0,
        "height_reward_scale": 2.0,
        "terminal_success_bonus": 10.0,
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


EXPERIMENT = StandupOrigS2()
