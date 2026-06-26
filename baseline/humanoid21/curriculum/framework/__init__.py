"""Curriculum learning framework — generic over reward schemes."""
from __future__ import annotations

from .experiment import Experiment, FrameworkParams, TrainablePolicy
from .training_loop import train
from .ppo_trainer import set_seed

__all__ = [
    "Experiment",
    "FrameworkParams",
    "TrainablePolicy",
    "train",
    "set_seed",
]
