"""PPO experiment registry — auto-discovers ``exp_*.py`` files in this directory.

Each experiment file should export ``EXPERIMENT_CLASS: type[ExperimentPPO]``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Type

from baseline.framework.ppo import ExperimentPPO

_REGISTRY: Dict[str, Type[ExperimentPPO]] = {}


def _discover() -> None:
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.glob("exp_*.py")):
        mod = importlib.import_module(f".{f.stem}", package=__package__)
        exp_cls = getattr(mod, "EXPERIMENT_CLASS", None)
        if exp_cls is not None:
            _REGISTRY[exp_cls.name] = exp_cls


def get_ppo_experiment(name: str, **kwargs) -> ExperimentPPO:
    """Instantiate a PPO experiment by name with optional constructor kwargs."""
    if not _REGISTRY:
        _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown PPO experiment {name!r}. Available: {list_ppo_experiments()}"
        )
    return _REGISTRY[name](**kwargs)


def list_ppo_experiments() -> List[str]:
    """Return sorted list of available PPO experiment names."""
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.keys())
