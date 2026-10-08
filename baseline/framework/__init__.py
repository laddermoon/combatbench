"""Training framework package.

Algorithm implementations are imported lazily so importing one framework
namespace does not silently pull in the other.
"""
from __future__ import annotations

from typing import Any


_PPO_EXPORTS = {
    "CommonParams",
    "ExperimentPPO",
    "PPOParams",
    "TrainablePolicy",
}


def __getattr__(name: str) -> Any:
    if name == "CriticMLP":
        from .critic_mlp import CriticMLP
        return CriticMLP
    if name in _PPO_EXPORTS:
        from . import ppo
        return getattr(ppo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CommonParams",
    "CriticMLP",
    "ExperimentPPO",
    "PPOParams",
    "TrainablePolicy",
]
