"""Task-specific pre-action fact providers for SAC experiments."""
from __future__ import annotations

from typing import Any

import numpy as np


class HeightPhiPreActionProvider:
    """Compute ``phi = uprightness * (height / standing_height)`` pre-action.

    This intentionally mirrors ``HeightPhiObserver`` semantics while using the
    accessor state that exists immediately before ``runtime.step()``.
    """

    def __init__(self, standing_height: float = 1.28) -> None:
        self.standing_height = float(standing_height)

    def compute(self, accessor: Any, agent_id: str) -> float:
        core_state = accessor.get_core_state()[agent_id]
        derived_state = accessor.get_derived_state([agent_id])[agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        return uprightness * (height / self.standing_height)


__all__ = ["HeightPhiPreActionProvider"]
