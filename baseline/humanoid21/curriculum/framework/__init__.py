"""Curriculum learning framework — generic over reward schemes."""
from __future__ import annotations

from .config import ExperimentConfig
from .training_loop import CurriculumConfig, train
from .ppo_trainer import set_seed

__all__ = [
    "ExperimentConfig",
    "CurriculumConfig",
    "train",
    "set_seed",
]
