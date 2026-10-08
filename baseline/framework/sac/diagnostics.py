"""Bounded in-memory diagnostics for SAC update ticks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

from .clocks import SACClockState


@dataclass
class SACTickRing:
    """Recent update-tick snapshots, bounded by ``capacity``."""

    capacity: int = 4096
    _items: Deque[Dict[str, Any]] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        if int(self.capacity) <= 0:
            raise ValueError(f"tick ring capacity must be > 0, got {self.capacity}")
        self._items = deque(maxlen=int(self.capacity))

    def append(
        self,
        *,
        clocks: SACClockState,
        metrics: Mapping[str, Any],
        sample_ids: Iterable[int] = (),
    ) -> Dict[str, Any]:
        item = {
            "clocks": clocks.snapshot(),
            "metrics": {str(k): float(v) for k, v in metrics.items()},
            "sample_ids": [int(v) for v in sample_ids],
        }
        self._items.append(item)
        return item

    def latest(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._items)
        if n is None:
            return items
        return items[-int(n):]

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["SACTickRing"]
