"""Experiment registry — auto-discovers ``exp_*.py`` files in this directory.

Each experiment file should export ``EXPERIMENT: Experiment``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List

from baseline.framework.experiment import Experiment as _Experiment

Experiment = _Experiment  # backward-compatible alias

_REGISTRY: Dict[str, Experiment] = {}


def _discover() -> None:
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.glob("exp_*.py")):
        mod = importlib.import_module(f".{f.stem}", package=__package__)
        exp = getattr(mod, "EXPERIMENT", None)
        if exp is not None:
            _REGISTRY[exp.name] = exp


def get_experiment(name: str) -> Experiment:
    """Retrieve an experiment config by name."""
    if not _REGISTRY:
        _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown experiment {name!r}. Available: {list_experiments()}"
        )
    return _REGISTRY[name]


def list_experiments() -> List[str]:
    """Return sorted list of available experiment names."""
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.keys())
