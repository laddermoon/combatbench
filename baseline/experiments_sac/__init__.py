"""SAC experiment registry — auto-discovers ``exp_sac_*.py`` files.

Each experiment file should export ``EXPERIMENT_CLASS: type[ExperimentSAC]``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Type

from baseline.framework.sac.experiment import ExperimentSAC

_REGISTRY: Dict[str, Type[ExperimentSAC]] = {}


def _discover() -> None:
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.glob("exp_sac_*.py")):
        mod = importlib.import_module(f".{f.stem}", package=__package__)
        exp_cls = getattr(mod, "EXPERIMENT_CLASS", None)
        if exp_cls is not None:
            _REGISTRY[exp_cls.name] = exp_cls


def get_sac_experiment(name: str, **kwargs) -> ExperimentSAC:
    if not _REGISTRY:
        _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown SAC experiment {name!r}. "
            f"Available: {list_sac_experiments()}"
        )
    return _REGISTRY[name](**kwargs)


def list_sac_experiments() -> List[str]:
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.keys())
