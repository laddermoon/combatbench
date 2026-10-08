"""Runtime loading and invocation of SAC pre-action fact providers."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Tuple

from .collection import SACFactSpec


class PreActionFactProvider(Protocol):
    """Compute one named fact before ``EnvRuntime.step()``.

    Implementations receive the runtime's read-only data accessor and must
    return a finite scalar/ndarray for their declared agent.
    """

    def compute(self, accessor: Any, agent_id: str) -> Any:
        ...


def _load_class(path: str) -> type:
    module_name, sep, class_name = path.partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(
            f"SAC fact provider must use 'module.path:ClassName', got {path!r}"
        )
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"SAC fact provider {path!r} not found")
    return cls


def build_fact_providers(
    specs: Iterable[SACFactSpec],
) -> Dict[str, List[Tuple[str, PreActionFactProvider]]]:
    out: Dict[str, List[Tuple[str, PreActionFactProvider]]] = {}
    for spec in specs:
        cls = _load_class(spec.provider)
        provider = cls(**dict(spec.config))
        if not hasattr(provider, "compute"):
            raise TypeError(
                f"SAC fact provider {spec.provider!r} has no compute() method"
            )
        out.setdefault(str(spec.agent_id), []).append((str(spec.name), provider))
    return out


__all__ = ["PreActionFactProvider", "build_fact_providers"]
