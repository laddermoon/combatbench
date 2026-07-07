"""Stage 9 (r12-r14): V2 gapped potential, wall awareness + Stage 5 bonus.

Resume from S8 checkpoint. Final stage with all anti-exploit features.
wall_aware=True, wall_penalty=-0.05, stage5_per_step_bonus=0.1
This matches the final r14 configuration that produced the successful model.
"""
from __future__ import annotations
from typing import Any, Dict

from baseline.humanoid21.curriculum.experiments.standup_repro_base import StandupReproBase


class StandupReproS9(StandupReproBase):
    name = "standup_repro_s9"
    BLUEPRINT = "standup_repro_v2_env.yaml"

    learning_rate = 5e-4
    critic_learning_rate = 5e-4
    entropy_coef = 0.0
    log_std_min = -4.0

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 10.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 100.0,
        "time_penalty": 0.0,
        "wall_penalty": -0.05,
        "stage5_per_step_bonus": 0.1,
        "curriculum_phase": 0,
        "height_thresholds": [0.5, 0.3, 0.15],
        "phase_transition_success_rate": 0.5,
        "phase_transition_eval_count": 5,
    }
    custom_config = DEFAULT_CUSTOM_CONFIG

    TERMINATION_PARAMS: Dict[str, Any] = {
        "success_height": 0.60,
        "success_uprightness": 0.70,
        "success_hold_steps": 50,
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }

    REWARDER_PARAMS: Dict[str, Any] = {
        "vel_gate": True,
        "wall_aware": True,
        "s5_h_thr": 0.60,
        "s5_u_thr": 0.70,
        "s5_base": 0.90,
        "s5_range": 0.10,
        "s5_h_range": 0.20,
        "s5_u_range": 0.20,
        "s5_v_power": 3,
    }


EXPERIMENT = StandupReproS9()
