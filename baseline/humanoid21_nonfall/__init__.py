"""
Humanoid21 Non-Fall Baseline

基于新框架 (envs/framework + envs/humanoid21) 的训练 baseline，
包含 GRPO 算法所需的奖励计算、对手策略、Gym 适配器和训练入口。
"""

# Reward Config
from .reward_config import (
    AttackerRewardConfig,
    DistanceStageRewardConfig,
    REWARD_TERM_KEYS,
    compute_attacker_reward,
    compute_distance_stage_reward,
    compute_distance_stage_curriculum_returns,
    zero_reward_terms,
)

# Rewarder
from .rewarder import Humanoid21Rewarder

# Opponents
from .opponents import (
    BaseCombatPolicy,
    StandingCombatPolicy,
    RandomCombatPolicy,
    ScriptedActiveCombatPolicy,
    make_opponent_policy,
)

# Gym Adapter
from .gym_adapter import SingleAgentAttackerEnv

# GRPO Algorithm
from .grpo import (
    GRPOModelConfig,
    GRPOActionPenaltyConfig,
    GRPOActor,
    GRPORolloutCollector,
    optimize_grpo,
    evaluate_grpo_actor,
    save_grpo_checkpoint,
    load_grpo_checkpoint,
    resolve_device,
)

__all__ = [
    # Reward Config
    "AttackerRewardConfig",
    "DistanceStageRewardConfig",
    "REWARD_TERM_KEYS",
    "compute_attacker_reward",
    "compute_distance_stage_reward",
    "compute_distance_stage_curriculum_returns",
    "zero_reward_terms",
    # Rewarder
    "Humanoid21Rewarder",
    # Opponents
    "BaseCombatPolicy",
    "StandingCombatPolicy",
    "RandomCombatPolicy",
    "ScriptedActiveCombatPolicy",
    "make_opponent_policy",
    # Gym Adapter
    "SingleAgentAttackerEnv",
    # GRPO Algorithm
    "GRPOModelConfig",
    "GRPOActionPenaltyConfig",
    "GRPOActor",
    "GRPORolloutCollector",
    "optimize_grpo",
    "evaluate_grpo_actor",
    "save_grpo_checkpoint",
    "load_grpo_checkpoint",
    "resolve_device",
]
