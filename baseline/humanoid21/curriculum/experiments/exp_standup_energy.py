"""Energy-based standup experiment — single phase.

PPO with entropy exploration.  Uses StandupEnergyRewarder (orbital energy
potential) + StandupEnergyTerminationPlugin (success = no non-foot ground
contact for ~2 seconds).

Usage::

    PYTHONPATH=. python3 baseline/framework/train.py --experiment standup_energy --algo ppo
    PYTHONPATH=. python3 baseline/framework/train.py --experiment standup_energy --algo ppo --smoke
    PYTHONPATH=. python3 baseline/framework/train.py --experiment standup_energy --algo ppo --background
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standup_energy_base import StandupEnergyBase


class StandupEnergyExperiment(StandupEnergyBase):
    name = "standup_energy"
    BLUEPRINT = "standup_energy_env.yaml"

    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    minibatch_size: int = 4096

    DEFAULT_CUSTOM_CONFIG = {
        "max_steps": 200,
        "potential_reward_scale": 1.0,
        "terminal_success_bonus": 100.0,
        "time_penalty": 0.0,
    }
    custom_config = DEFAULT_CUSTOM_CONFIG


EXPERIMENT = StandupEnergyExperiment()
