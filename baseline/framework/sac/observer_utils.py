"""SAC-local per-step observer extraction helpers.

Copied from the neutral observer utility logic so SAC experiment modules
can be imported without loading the shared rollout package initializer.
Missing required values fail loudly; optional fields are the caller's
responsibility to check.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def coerce_per_step(values: Any, expected_len: int) -> np.ndarray:
    if values is None:
        raise ValueError("Required observer output is missing")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(
            f"Observer output length {arr.shape[0]} != expected episode "
            f"length {expected_len}."
        )
    return arr


def extract_per_step_scalar(
    observer_outputs: Any,
    observer_name: str,
    expected_len: int,
) -> np.ndarray:
    node = observer_outputs.get(observer_name)
    if node is None:
        raise KeyError(f"Missing required observer output {observer_name!r}")
    values = next(iter(node.values())) if isinstance(node, dict) else node
    return coerce_per_step(values, expected_len)


def extract_per_step_field(
    observer_outputs: Any,
    observer_name: str,
    field: str,
    expected_len: int,
) -> Optional[np.ndarray]:
    node = observer_outputs.get(observer_name)
    if not isinstance(node, dict) or field not in node:
        return None
    return coerce_per_step(node[field], expected_len)
