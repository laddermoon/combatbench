"""Curriculum learning framework — generic over reward schemes."""
from __future__ import annotations

from .config import ExperimentConfig
from .training_loop import train
from .ppo_trainer import set_seed

__all__ = [
    "ExperimentConfig",
    "train",
    "set_seed",
]
