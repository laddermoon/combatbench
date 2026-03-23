from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AttackerRewardConfig:
    damage_reward_scale: float = 1.0
    damage_received_penalty_scale: float = 0.05
    hit_reward_scale: float = 0.35
    approach_reward_scale: float = 0.8
    close_distance_reward_scale: float = 0.08
    close_distance_threshold: float = 1.25
    retreat_penalty_scale: float = 0.35
    facing_reward_scale: float = 0.05
    facing_delta_reward_scale: float = 0.05
    upright_reward_scale: float = 0.03
    upright_delta_reward_scale: float = 0.05
    tilt_penalty_scale: float = 0.1
    action_magnitude_reward_scale: float = 0.02
    action_delta_reward_scale: float = 0.05
    inactivity_penalty: float = 0.02
    inactivity_action_threshold: float = 0.03
    inactivity_delta_threshold: float = 0.02
    win_bonus: float = 2.0
    loss_penalty: float = 0.5


@dataclass
class DistanceStageRewardConfig:
    target_distance: float = 0.4
    target_tolerance: float = 0.08
    distance_error_penalty_scale: float = 1.4
    distance_progress_reward_scale: float = 2.5
    distance_regress_penalty_scale: float = 0.8
    distance_record_reward_scale: float = 0.8
    target_band_reward_scale: float = 0.25
    first_target_arrival_bonus_scale: float = 2.5
    overshoot_penalty_scale: float = 1.0
    facing_reward_scale: float = 0.03
    upright_reward_scale: float = 0.04
    upright_delta_reward_scale: float = 0.03
    tilt_penalty_scale: float = 0.08
    action_magnitude_penalty_scale: float = 0.01
    action_delta_penalty_scale: float = 0.03


REWARD_TERM_KEYS = (
    "damage_dealt",
    "damage_received_penalty",
    "hit_reward",
    "approach_reward",
    "close_distance_reward",
    "retreat_penalty",
    "facing_reward",
    "facing_delta_reward",
    "upright_reward",
    "upright_delta_reward",
    "tilt_penalty",
    "action_magnitude_reward",
    "action_delta_reward",
    "inactivity_penalty",
    "win_bonus",
    "loss_penalty",
    "distance_error_penalty",
    "distance_progress_reward",
    "distance_regress_penalty",
    "distance_record_reward",
    "target_band_reward",
    "first_target_arrival_bonus",
    "overshoot_penalty",
    "action_magnitude_penalty",
    "action_delta_penalty",
)


def zero_reward_terms() -> Dict[str, float]:
    return {key: 0.0 for key in REWARD_TERM_KEYS}


def compute_attacker_reward(
    metrics: Dict[str, float],
    config: Optional[AttackerRewardConfig] = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = AttackerRewardConfig() if config is None else config
    terms = zero_reward_terms()
    horizontal_distance = float(metrics.get("horizontal_distance", 0.0))
    horizontal_distance_delta = float(metrics.get("horizontal_distance_delta", 0.0))
    uprightness = float(metrics.get("uprightness", 1.0))
    uprightness_delta = float(metrics.get("uprightness_delta", 0.0))

    terms["damage_dealt"] = cfg.damage_reward_scale * float(metrics.get("damage_dealt", 0.0))
    terms["damage_received_penalty"] = -cfg.damage_received_penalty_scale * float(metrics.get("damage_received", 0.0))
    terms["hit_reward"] = cfg.hit_reward_scale * float(metrics.get("hits_dealt", 0.0))
    terms["approach_reward"] = cfg.approach_reward_scale * max(0.0, horizontal_distance_delta)
    terms["close_distance_reward"] = cfg.close_distance_reward_scale * max(0.0, cfg.close_distance_threshold - horizontal_distance)
    terms["retreat_penalty"] = -cfg.retreat_penalty_scale * max(0.0, -horizontal_distance_delta)
    terms["facing_reward"] = cfg.facing_reward_scale * max(0.0, float(metrics.get("facing_opponent", 0.0)))
    terms["facing_delta_reward"] = cfg.facing_delta_reward_scale * max(0.0, float(metrics.get("facing_delta", 0.0)))
    terms["upright_reward"] = cfg.upright_reward_scale * max(0.0, uprightness)
    terms["upright_delta_reward"] = cfg.upright_delta_reward_scale * max(0.0, uprightness_delta)
    terms["tilt_penalty"] = -cfg.tilt_penalty_scale * max(0.0, 1.0 - uprightness)
    terms["action_magnitude_reward"] = cfg.action_magnitude_reward_scale * float(metrics.get("action_magnitude", 0.0))
    terms["action_delta_reward"] = cfg.action_delta_reward_scale * float(metrics.get("action_delta", 0.0))

    is_inactive = (
        float(metrics.get("action_magnitude", 0.0)) < cfg.inactivity_action_threshold
        and float(metrics.get("action_delta", 0.0)) < cfg.inactivity_delta_threshold
        and float(metrics.get("damage_dealt", 0.0)) <= 0.0
        and float(metrics.get("hits_dealt", 0.0)) <= 0.0
    )
    if is_inactive:
        terms["inactivity_penalty"] = -cfg.inactivity_penalty

    if float(metrics.get("win", 0.0)) > 0.0:
        terms["win_bonus"] = cfg.win_bonus
    if float(metrics.get("loss", 0.0)) > 0.0:
        terms["loss_penalty"] = -cfg.loss_penalty

    reward = float(sum(terms.values()))
    return reward, terms


def compute_distance_stage_reward(
    metrics: Dict[str, float],
    config: Optional[DistanceStageRewardConfig] = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = DistanceStageRewardConfig() if config is None else config
    terms = zero_reward_terms()
    distance_error = float(metrics.get("distance_error", 0.0))
    distance_error_delta = float(metrics.get("distance_error_delta", 0.0))
    best_distance_error_delta = float(metrics.get("best_distance_error_delta", 0.0))
    within_target_band = float(metrics.get("within_target_band", 0.0))
    first_target_reached = float(metrics.get("first_target_reached", 0.0))
    horizontal_distance = float(metrics.get("horizontal_distance", 0.0))
    uprightness = float(metrics.get("uprightness", 1.0))
    uprightness_delta = float(metrics.get("uprightness_delta", 0.0))
    current_step = max(0.0, float(metrics.get("current_step", 0.0)))
    max_steps = max(1.0, float(metrics.get("max_steps", 1.0)))
    lower_target_bound = cfg.target_distance - cfg.target_tolerance

    terms["distance_error_penalty"] = -cfg.distance_error_penalty_scale * distance_error
    terms["distance_progress_reward"] = cfg.distance_progress_reward_scale * max(0.0, distance_error_delta)
    terms["distance_regress_penalty"] = -cfg.distance_regress_penalty_scale * max(0.0, -distance_error_delta)
    terms["distance_record_reward"] = cfg.distance_record_reward_scale * max(0.0, best_distance_error_delta)
    terms["target_band_reward"] = cfg.target_band_reward_scale * within_target_band
    terms["first_target_arrival_bonus"] = (
        cfg.first_target_arrival_bonus_scale * max(0.0, 1.0 - current_step / max_steps) * first_target_reached
    )
    terms["overshoot_penalty"] = -cfg.overshoot_penalty_scale * max(0.0, lower_target_bound - horizontal_distance)
    terms["facing_reward"] = cfg.facing_reward_scale * max(0.0, float(metrics.get("facing_opponent", 0.0)))
    terms["upright_reward"] = cfg.upright_reward_scale * max(0.0, uprightness)
    terms["upright_delta_reward"] = cfg.upright_delta_reward_scale * max(0.0, uprightness_delta)
    terms["tilt_penalty"] = -cfg.tilt_penalty_scale * max(0.0, 1.0 - uprightness)
    terms["action_magnitude_penalty"] = -cfg.action_magnitude_penalty_scale * float(metrics.get("action_magnitude", 0.0))
    terms["action_delta_penalty"] = -cfg.action_delta_penalty_scale * float(metrics.get("action_delta", 0.0))

    reward = float(sum(terms.values()))
    return reward, terms
