"""V2 experiment registry — auto-discovers ``exp_*.py`` files in this directory.

Each experiment file should export ``EXPERIMENT: ExperimentV2``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List

from baseline.framework.experiment_v2 import ExperimentV2

_REGISTRY: Dict[str, ExperimentV2] = {}


def _discover() -> None:
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.glob("exp_*.py")):
        mod = importlib.import_module(f".{f.stem}", package=__package__)
        exp = getattr(mod, "EXPERIMENT", None)
        if exp is not None:
            _REGISTRY[exp.name] = exp


def get_v2_experiment(name: str) -> ExperimentV2:
    """Retrieve a V2 experiment config by name."""
    if not _REGISTRY:
        _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown V2 experiment {name!r}. Available: {list_v2_experiments()}"
        )
    return _REGISTRY[name]


def list_v2_experiments() -> List[str]:
    """Return sorted list of available V2 experiment names."""
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.keys())
