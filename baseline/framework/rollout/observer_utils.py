"""Per-step observer-output extraction helpers.

These utilities work directly on the stacked ``observer_outputs`` dict
of an :class:`~baseline.framework.rollout.episode.Episode`.  They are
pure data-extraction functions — no PPO / critic / optimizer logic —
and are shared by both the v1 and v2 experiment/training paths.

Moved here from ``baseline/framework/ppo_trainer.py`` so that
experiments do not need to depend on the (legacy) PPO trainer module
just to read observer fields.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core coercion
# ---------------------------------------------------------------------------

def coerce_per_step(values: Any, expected_len: int) -> np.ndarray:
    """Coerce a raw observer leaf into a ``(T,)`` float32 array of length ``expected_len``.

    Raises ``ValueError`` if the observer output length does not match the
    expected episode length — a length mismatch indicates a bug in the
    observer (e.g. wrong stacking cadence) and must not be silently
    papered over with interpolation.
    """
    if values is None:
        return np.zeros(expected_len, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(
            f"Observer output length {arr.shape[0]} != expected episode "
            f"length {expected_len}. This indicates a timestep misalignment "
            f"bug in the observer; reward interpolation is intentionally "
            f"disabled to surface the problem."
        )
    return arr


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_per_step_scalar(
    observer_outputs: Any,
    observer_name: str,
    expected_len: int,
) -> np.ndarray:
    """Pull a ``(T,)`` float32 reward signal from stacked observer outputs.

    If the observer emits a dict (e.g. ``{"reward": ..., "in_zone": ...}``),
    the first value is used. Use :func:`extract_per_step_field` to read a
    specific named field.
    """
    node = observer_outputs.get(observer_name)
    if node is None:
        return np.zeros(expected_len, dtype=np.float32)
    values = next(iter(node.values())) if isinstance(node, dict) else node
    return coerce_per_step(values, expected_len)


def extract_per_step_field(
    observer_outputs: Any,
    observer_name: str,
    field: str,
    expected_len: int,
) -> Optional[np.ndarray]:
    """Pull a specific named field from a dict-valued observer output.

    Returns ``None`` if the observer is absent or not a dict.
    """
    node = observer_outputs.get(observer_name)
    if not isinstance(node, dict) or field not in node:
        return None
    return coerce_per_step(node[field], expected_len)
