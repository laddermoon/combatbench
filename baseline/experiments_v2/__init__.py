"""V2 experiment registry — auto-discovers ``exp_*.py`` files in this directory.

Each experiment file should export ``EXPERIMENT_CLASS: type[ExperimentV2]``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Type

from baseline.framework.experiment_v2 import ExperimentV2

_REGISTRY: Dict[str, Type[ExperimentV2]] = {}


def _discover() -> None:
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.glob("exp_*.py")):
        mod = importlib.import_module(f".{f.stem}", package=__package__)
        exp_cls = getattr(mod, "EXPERIMENT_CLASS", None)
        if exp_cls is not None:
            _REGISTRY[exp_cls.name] = exp_cls


def get_v2_experiment(name: str, **kwargs) -> ExperimentV2:
    """Instantiate a V2 experiment by name with optional constructor kwargs."""
    if not _REGISTRY:
        _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown V2 experiment {name!r}. Available: {list_v2_experiments()}"
        )
    return _REGISTRY[name](**kwargs)


def list_v2_experiments() -> List[str]:
    """Return sorted list of available V2 experiment names."""
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.keys())
