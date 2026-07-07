"""Stage 8 (r10): V2 gapped potential, anti jump-up exploit.

Resume from S7 checkpoint. Require sustainable standing.
hold=50, vel_gate=True, term=100, time_penalty=0, v_power=3
Stage 4.5b: high-velocity standing potential [0.80, 0.85]
"""
from __future__ import annotations
from typing import Any, Dict

from baseline.humanoid21.curriculum.experiments.standup_repro_base import StandupReproBase


class StandupReproS8(StandupReproBase):
    name = "standup_repro_s8"
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
        "wall_penalty": 0.0,
        "stage5_per_step_bonus": 0.0,
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
        "wall_aware": False,
        "s5_h_thr": 0.60,
        "s5_u_thr": 0.70,
        "s5_base": 0.90,
        "s5_range": 0.10,
        "s5_h_range": 0.20,
        "s5_u_range": 0.20,
        "s5_v_power": 3,
    }


EXPERIMENT = StandupReproS8()
