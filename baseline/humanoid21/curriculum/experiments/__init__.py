"""Experiment registry — auto-discovers ``exp_*.py`` files in this directory.

Each experiment file should export ``EXPERIMENT: Experiment``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List

try:
    from baseline.framework.experiment import Experiment as _Experiment
except ImportError:
    # 旧统一框架已移除（experiments_ppo/sac 注册表接管）；
    # Experiment 仅作类型注解/别名，注解在 future-import 下惰性，
    # 此处容忍缺失以免阻断 exp_* 模块的直接导入。
    _Experiment = object

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
