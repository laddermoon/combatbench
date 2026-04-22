"""No-op combat policy.

Emits a zero action every step. Useful as a minimal baseline, for
debugging environment wiring, and as a default "do nothing" opponent.

Conforms to the canonical :class:`envs.framework.policy.Policy` contract:
``act`` returns an explicit ``np.ndarray`` (the framework forbids
``None`` at this layer — there is no "hold previous action" sentinel).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from policy.base import BaseCombatPolicy


class NoOpActionPolicy(BaseCombatPolicy):
    """Always returns a zero action of shape ``(ACTION_DIM,)``."""

    def act(self, observation: Any) -> np.ndarray:
        return np.zeros(self.ACTION_DIM, dtype=np.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
