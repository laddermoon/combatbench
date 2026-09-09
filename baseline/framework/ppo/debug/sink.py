"""DebugSink — write-only per-frame data capture for offline replay.

S2: ``ppo_update`` records named arrays at fixed *stages* into a sink.
The sink is **write-only** from the trainer's perspective — its only
effect is to persist data; it must not change algorithm branches or
consume RNG (P1/P3 from ``DESIGN_debug_system.md``).

Stage/name contract (renaming breaks downstream tools — keep stable):

``buffer``
    ``old_log_prob``, ``explore_factor``, ``floor_weight``,
    ``sample_weights``, ``frame_ids`` (may be None)
``gae``
    per channel ``values.<k>``, ``advantages.<k>``, ``returns.<k>``,
    ``bootstrap_value.<k>`` (per-segment array), ``active_mask.<k>``
``combine``
    per channel ``aw_frame.<k>``, ``aw_normed.<k>``, ``conf.<k>``,
    ``normed_adv.<k>``, ``contribution.<k>``, ``norm_mask.<k>``;
    global ``aw_l1_sum``, ``combined_adv``
``update``
    per minibatch (keyed by ``epoch:mb``): ``ratio``, ``clip_mask``,
    ``policy_loss``, ``floor_loss``, ``grad_norm``;
    when ``include_full_grad=True``: ``full_grad`` (flat 1-D tensor) +
    ``full_grad_param_names`` (list) for epoch 0, mb 0 only

See ``DESIGN_debug_system.md`` §3.2.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


# Fixed stage names — the contract between ppo_update and downstream tools.
STAGES: Tuple[str, ...] = ("buffer", "gae", "combine", "update")


class DebugSink(Protocol):
    """Write-only sink for per-frame debug data.

    The trainer never reads from the sink.  ``record`` accepts numpy
    arrays, scalars, or None (None is silently dropped so callers can
    pass optional values without branching).
    """

    def record(self, stage: str, name: str, value: Any) -> None:
        """Record a named value under a stage."""
        ...

    def record_minibatch(
        self, epoch: int, mb: int, name: str, value: Any,
    ) -> None:
        """Record a per-minibatch value under the ``update`` stage."""
        ...

    def close(self) -> None:
        """Flush to disk.  No further records expected."""
        ...


def _coerce(value: Any) -> Optional[np.ndarray]:
    """Normalize a value to a numpy array (or None / scalar as float array).

    None → None (dropped).  Tensors are detached + cpu'd.  Scalars
    become 0-D float arrays.  Lists/tuples become arrays.
    """
    if value is None:
        return None
    # Torch tensor — detach without graph pollution.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    arr = np.asarray(value, dtype=None if isinstance(value, (list, tuple)) else None)
    # Don't force float on integer masks — preserve dtype for masks.
    return arr


class NpzSink:
    """In-memory accumulator; ``close()`` writes one ``.npz`` per stage.

    Per-minibatch records (``record_minibatch``) are stored under the
    ``update`` stage as a dict keyed by ``f"{epoch:03d}:{mb:03d}"``,
    each value being a sub-dict of name → array.  On close, the update
    stage is written as a single ``.npz`` with one key per
    ``f"{epoch:03d}:{mb:03d}.{name}"`` (flattened) plus the raw arrays
    concatenated where unambiguous.

    For simplicity and robustness, minibatch arrays of the same name
    across minibatches are also stacked into a single array under
    ``{name}`` when all minibatches have the same shape; otherwise
    only the flattened per-minibatch keys are written.
    """

    def __init__(self, out_dir: Union[str, Path]):
        self.out_dir = Path(out_dir)
        # stage -> {name: array}
        self._records: Dict[str, Dict[str, Any]] = {s: {} for s in STAGES}
        # (epoch, mb) -> {name: array}  (lives under "update" stage)
        self._minibatches: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._closed = False

    def record(self, stage: str, name: str, value: Any) -> None:
        if self._closed:
            raise RuntimeError("NpzSink closed; cannot record")
        if stage not in self._records:
            raise ValueError(
                f"unknown stage {stage!r}; expected one of {STAGES}"
            )
        arr = _coerce(value)
        if arr is None:
            return
        self._records[stage][name] = arr

    def record_minibatch(
        self, epoch: int, mb: int, name: str, value: Any,
    ) -> None:
        if self._closed:
            raise RuntimeError("NpzSink closed; cannot record")
        arr = _coerce(value)
        if arr is None:
            return
        key = (epoch, mb)
        self._minibatches.setdefault(key, {})[name] = arr

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Write per-stage .npz for non-update stages.
        for stage in STAGES:
            if stage == "update":
                continue
            data = self._records[stage]
            if not data:
                # Still write an empty file so consumers can detect presence.
                np.savez_compressed(self.out_dir / f"{stage}.npz")
                continue
            np.savez_compressed(self.out_dir / f"{stage}.npz", **data)

        # Write update stage: per-minibatch flattened + stacked where uniform.
        update_payload: Dict[str, Any] = {}
        # Group by name across minibatches in epoch:mb order.
        keys_sorted = sorted(self._minibatches.keys())
        names: List[str] = []
        for key in keys_sorted:
            for n in self._minibatches[key]:
                if n not in names:
                    names.append(n)
        for name in names:
            arrays = []
            for key in keys_sorted:
                mb_dict = self._minibatches.get(key, {})
                if name in mb_dict:
                    arrays.append(mb_dict[name])
            if not arrays:
                continue
            # Flatten per-minibatch keys.
            for key, arr in zip(keys_sorted, arrays):
                # Skip if this name doesn't exist for this mb (zip mismatch).
                if name not in self._minibatches.get(key, {}):
                    continue
                update_payload[f"{key[0]:03d}:{key[1]:03d}.{name}"] = arr
            # Try to stack — only if all same shape.
            shapes = {a.shape for a in arrays}
            if len(shapes) == 1:
                update_payload[name] = np.stack(arrays, axis=0)
        np.savez_compressed(
            self.out_dir / "update.npz", **update_payload,
        )

    # ------------------------------------------------------------------
    # Introspection (for tests)
    # ------------------------------------------------------------------
    def stages_recorded(self) -> Dict[str, List[str]]:
        """Return {stage: [name, ...]} for non-empty stages (test helper)."""
        out: Dict[str, List[str]] = {}
        for stage in STAGES:
            if stage == "update":
                if self._minibatches:
                    names = set()
                    for d in self._minibatches.values():
                        names.update(d.keys())
                    out[stage] = sorted(names)
                continue
            if self._records[stage]:
                out[stage] = sorted(self._records[stage].keys())
        return out

    @property
    def out_path(self) -> Path:
        return self.out_dir


__all__ = ["DebugSink", "NpzSink", "STAGES"]
