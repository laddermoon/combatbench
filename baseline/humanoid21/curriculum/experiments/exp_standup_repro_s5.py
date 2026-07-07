"""Stage 5 (r7): Switch to V2 gapped potential function.

Resume from S3 checkpoint (u435 in original history). V2 introduces
transition gaps between stages. Stage 3→4 gap = 0.10 (0.55→0.65).
No vel_gate, no wall_aware yet. pot=5.0, time_penalty=-0.01
"""
from __future__ import annotations
from typing import Any, Dict

from baseline.humanoid21.curriculum.experiments.standup_repro_base import StandupReproBase


class StandupReproS5(StandupReproBase):
    name = "standup_repro_s5"
    BLUEPRINT = "standup_repro_v2_env.yaml"

    learning_rate = 5e-4
    critic_learning_rate = 5e-4
    entropy_coef = 0.0
    log_std_min = -4.0

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 5.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 50.0,
        "time_penalty": -0.01,
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

    REWARDER_PARAMS: Dict[str, Any] = {
        "vel_gate": False,
        "wall_aware": False,
        "s5_h_thr": 0.75,
        "s5_u_thr": 0.85,
        "s5_base": 0.85,
        "s5_range": 0.15,
        "s5_h_range": 0.15,
        "s5_u_range": 0.15,
        "s5_v_power": 1,
    }


EXPERIMENT = StandupReproS5()
