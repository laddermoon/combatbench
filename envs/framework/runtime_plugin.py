"""Backward-compat shim.

This module was renamed to :mod:`envs.framework.observer_plugin`; the symbols
live there now. Existing call sites (``env_runtime``, several tests, some
baselines) still import from ``envs.framework.runtime_plugin``, so this
module re-exports the same names until those imports are migrated.

Do not add new symbols here — put them in ``observer_plugin`` instead.
"""
from .observer_plugin import (  # noqa: F401
    BaseObserverPlugin,
    BaseRuntimeUnit,
    _ObserverDispatcherPlugin,
)

__all__ = [
    "BaseObserverPlugin",
    "BaseRuntimeUnit",
    "_ObserverDispatcherPlugin",
]
