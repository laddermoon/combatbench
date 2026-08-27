"""PPO framework — on-policy training with trajectory-based reward channels.

This package contains the PPO-only training pipeline:
- ``experiment`` — ``ExperimentPPO`` ABC + ``CommonParams`` / ``PPOParams``
- ``trajectory`` — ``Trajectory``, ``RewardChannel``, ``ChannelData``
- ``trainer`` — ``PPOBuffer`` + ``ppo_update``
- ``loop`` — ``train_ppo`` training loop + checkpoint helpers
"""
from __future__ import annotations

from .experiment import (
    ActorEval,
    CommonParams,
    ExperimentPPO,
    ExplorationSpec,
    PPOParams,
    TrainablePolicy,
)

__all__ = [
    "ActorEval",
    "CommonParams",
    "ExperimentPPO",
    "ExplorationSpec",
    "PPOParams",
    "TrainablePolicy",
]
