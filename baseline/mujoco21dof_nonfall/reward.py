from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AttackerRewardConfig:
    damage_reward_scale: float = 1.0
    damage_received_penalty_scale: float = 0.05
    hit_reward_scale: float = 0.25
    approach_reward_scale: float = 0.5
    facing_reward_scale: float = 0.05
    facing_delta_reward_scale: float = 0.05
    action_magnitude_reward_scale: float = 0.02
    action_delta_reward_scale: float = 0.03
    inactivity_penalty: float = 0.02
    inactivity_action_threshold: float = 0.03
    inactivity_delta_threshold: float = 0.02
    win_bonus: float = 2.0
    loss_penalty: float = 0.5


REWARD_TERM_KEYS = (
    "damage_dealt",
    "damage_received_penalty",
    "hit_reward",
    "approach_reward",
    "facing_reward",
    "facing_delta_reward",
    "action_magnitude_reward",
    "action_delta_reward",
    "inactivity_penalty",
    "win_bonus",
    "loss_penalty",
)


def zero_reward_terms() -> Dict[str, float]:
    return {key: 0.0 for key in REWARD_TERM_KEYS}


def compute_attacker_reward(
    metrics: Dict[str, float],
    config: Optional[AttackerRewardConfig] = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = AttackerRewardConfig() if config is None else config
    terms = zero_reward_terms()

    terms["damage_dealt"] = cfg.damage_reward_scale * float(metrics.get("damage_dealt", 0.0))
    terms["damage_received_penalty"] = -cfg.damage_received_penalty_scale * float(metrics.get("damage_received", 0.0))
    terms["hit_reward"] = cfg.hit_reward_scale * float(metrics.get("hits_dealt", 0.0))
    terms["approach_reward"] = cfg.approach_reward_scale * max(0.0, float(metrics.get("horizontal_distance_delta", 0.0)))
    terms["facing_reward"] = cfg.facing_reward_scale * max(0.0, float(metrics.get("facing_opponent", 0.0)))
    terms["facing_delta_reward"] = cfg.facing_delta_reward_scale * max(0.0, float(metrics.get("facing_delta", 0.0)))
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
