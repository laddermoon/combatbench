"""``--where`` expression parser for frame filtering.

S3: Simple field-comparison syntax (no Python eval).  Supports
``<field_path> <op> <value>`` joined by ``&&``.

Syntax::

    combine.aw_normed.r_left_foot < 0
    gae.advantages.r_potential > 0.01
    buffer.explore_factor > 0 && combine.aw_normed.r_potential < 0

Field path format: ``<stage>.<name>`` or ``<stage>.<name>.<channel>``.
The parser splits on ``.`` and looks up the array in a dict of
``{stage: {name: array}}`` (loaded from replay .npz files).

Operators: ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``.

See ``DEBUG_GUIDE.md`` §3.4 ``frame --where``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One comparison: ``field_path op value``."""
    field_path: str  # e.g. "combine.aw_normed.r_left_foot"
    op: str           # one of <, >, <=, >=, ==, !=
    value: float

    def evaluate(self, arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """Return boolean mask for this condition.

        Args:
            arrays: ``{field_path: array}`` — the caller must flatten
                the stage/name/channel hierarchy into dotted keys.
        """
        arr = arrays.get(self.field_path)
        if arr is None:
            raise KeyError(
                f"--where: field {self.field_path!r} not found. "
                f"Available: {sorted(arrays.keys())[:20]}..."
            )
        arr = np.asarray(arr)
        if arr.ndim == 0:
            # Scalar — broadcast to length-1.
            arr = arr.reshape(1)
        v = self.value
        if self.op == "<":
            return arr < v
        elif self.op == ">":
            return arr > v
        elif self.op == "<=":
            return arr <= v
        elif self.op == ">=":
            return arr >= v
        elif self.op == "==":
            return arr == v
        elif self.op == "!=":
            return arr != v
        else:
            raise ValueError(f"unknown operator {self.op!r}")


@dataclass
class WhereClause:
    """Parsed ``--where`` expression: one or more Conditions ANDed together."""
    conditions: List[Condition] = field(default_factory=list)
    raw_expr: str = ""

    def match(self, arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """Return boolean mask of frames matching ALL conditions."""
        if not self.conditions:
            # No conditions → match all.
            # Need at least one array to determine length.
            for arr in arrays.values():
                arr = np.asarray(arr)
                if arr.ndim >= 1:
                    return np.ones(len(arr), dtype=bool)
            return np.array([], dtype=bool)

        masks = []
        for cond in self.conditions:
            masks.append(cond.evaluate(arrays))
        # All conditions must match (AND).
        # Broadcast: all arrays should have the same length (total frames).
        result = masks[0]
        for m in masks[1:]:
            result = result & m
        return result


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Operator pattern — order matters: two-char ops before one-char.
_OPS = ["<=", ">=", "==", "!=", "<", ">"]
_OP_PATTERN = "|".join(re.escape(op) for op in _OPS)

# One condition: field_path op value
_CONDITION_RE = re.compile(
    r"\s*([\w.]+)\s*(" + _OP_PATTERN + r")\s*(-?[\d.eE+-]+)\s*"
)

# Split on && to get individual conditions.
_AND_SPLIT_RE = re.compile(r"\s*&&\s*")


def parse(expr: str) -> WhereClause:
    """Parse a ``--where`` expression into a :class:`WhereClause`.

    Raises ``ValueError`` on syntax errors.
    """
    expr = expr.strip()
    if not expr:
        return WhereClause(conditions=[], raw_expr="")

    parts = _AND_SPLIT_RE.split(expr)
    conditions: List[Condition] = []
    for part in parts:
        m = _CONDITION_RE.match(part)
        if not m:
            raise ValueError(
                f"--where: cannot parse {part!r}. "
                f"Expected: <field_path> <op> <number> "
                f"(e.g. combine.aw_normed.r_left_foot < 0)"
            )
        field_path = m.group(1)
        op = m.group(2)
        try:
            value = float(m.group(3))
        except ValueError:
            raise ValueError(
                f"--where: cannot parse value {m.group(3)!r} as a number"
            )
        conditions.append(Condition(field_path=field_path, op=op, value=value))

    return WhereClause(conditions=conditions, raw_expr=expr)


# ---------------------------------------------------------------------------
# Array flattening helper
# ---------------------------------------------------------------------------

def flatten_npz(
    stage_data: Dict[str, Any],
    stage_name: str,
) -> Dict[str, np.ndarray]:
    """Flatten a stage's .npz data into ``{stage.name: array}`` keys.

    For per-channel arrays stored as ``name.channel`` (e.g.
    ``aw_normed.r_left_foot``), the key becomes
    ``stage.name.channel`` (e.g. ``combine.aw_normed.r_left_foot``).

    For global arrays stored as ``name`` (e.g. ``combined_adv``), the
    key becomes ``stage.name`` (e.g. ``combine.combined_adv``).
    """
    out: Dict[str, np.ndarray] = {}
    for key, arr in stage_data.items():
        out[f"{stage_name}.{key}"] = arr
    return out


def build_frame_arrays(
    replay_dir: "Path",
) -> Dict[str, np.ndarray]:
    """Load all replay .npz files and flatten into ``{dotted.key: array}``.

    This is the dict consumed by :meth:`WhereClause.match` and by
    :func:`frame.inspect_frame`.
    """
    from pathlib import Path
    replay_dir = Path(replay_dir)
    arrays: Dict[str, np.ndarray] = {}
    for stage in ("buffer", "gae", "combine", "update"):
        npz_path = replay_dir / f"{stage}.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            arrays.update(flatten_npz(dict(data), stage))
    # debug_arrays.npz — experiment-owned, no stage prefix.
    da_path = replay_dir / "debug_arrays.npz"
    if da_path.exists():
        data = np.load(da_path, allow_pickle=True)
        for key, arr in data.items():
            arrays[f"debug.{key}"] = arr
    return arrays


__all__ = [
    "Condition",
    "WhereClause",
    "parse",
    "flatten_npz",
    "build_frame_arrays",
]
