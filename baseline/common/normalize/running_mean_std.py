"""``RunningMeanStd``: numerically-stable streaming mean / variance.

Implements the Chan / Welford parallel algorithm (combine-from-batch
formulation), which is the standard pattern used by SB3 / CleanRL /
IsaacGym for PPO observation normalization. The merge form is
preferable to one-sample-at-a-time Welford because each PPO iteration
sees a *batch* of samples and we want to update with that whole batch
in O(1) ops on the mean / variance tensors.

Numerical notes
---------------
* ``count`` is initialized to ``epsilon`` (default 1e-4) rather than 0,
  so the first ``normalize`` call after construction does not divide by
  zero. The cost is a small bias on the first ~100 samples; SB3 uses
  the same trick.
* Internal state is kept in ``float64`` to limit drift over millions of
  PPO updates. The ``normalize`` output respects the input dtype.
* ``var`` is the *population* variance (no Bessel's correction). This
  matches SB3 / CleanRL conventions; do not "fix" it without coordinating
  with whatever consumer expects population statistics.

State shape
-----------
``shape`` is the per-element shape of one sample. ``update`` accepts
inputs of shape ``(batch, *shape)`` (preferred) or ``shape`` itself
(treated as ``batch=1``). Higher-rank inputs are not auto-flattened —
the caller is responsible for collapsing time/episode axes first
(``rewards.reshape(-1)`` for example).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class RunningMeanStd:
    """Streaming mean / variance with batch-merge updates."""

    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        *,
        epsilon: float = 1e-4,
    ) -> None:
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self.mean: np.ndarray = np.zeros(self.shape, dtype=np.float64)
        self.var: np.ndarray = np.ones(self.shape, dtype=np.float64)
        self.count: float = float(epsilon)
        self._epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update(self, x: np.ndarray) -> None:
        """Update statistics from a batch of samples.

        ``x.shape`` must be either ``self.shape`` (single sample) or
        ``(batch, *self.shape)``.
        """
        arr = np.asarray(x)
        if arr.shape == self.shape:
            arr = arr[np.newaxis, ...]
        elif arr.shape[1:] != self.shape:
            raise ValueError(
                f"Expected x.shape == {self.shape} or (batch, *{self.shape}); "
                f"got {arr.shape}."
            )
        if arr.shape[0] == 0:
            return
        batch_mean = arr.mean(axis=0, dtype=np.float64)
        batch_var = arr.var(axis=0, dtype=np.float64)
        batch_count = int(arr.shape[0])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Merge another distribution's (mean, var, count) into self.

        Useful when statistics have already been computed elsewhere
        (e.g. across worker processes).
        """
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    # ------------------------------------------------------------------
    # Use
    # ------------------------------------------------------------------
    def normalize(
        self,
        x: np.ndarray,
        *,
        eps: float = 1e-8,
        center: bool = True,
    ) -> np.ndarray:
        """Return ``(x - mean) / sqrt(var + eps)`` (or ``x / sqrt(var + eps)``).

        ``center=False`` is the PPO reward-normalization convention:
        divide by running std but do **not** subtract the mean (the
        sign of the reward is informative and shouldn't be shifted).
        """
        arr = np.asarray(x)
        out_dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
        std = np.sqrt(self.var + eps)
        if center:
            normalized = (arr.astype(np.float64) - self.mean) / std
        else:
            normalized = arr.astype(np.float64) / std
        return normalized.astype(out_dtype, copy=False)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    # ------------------------------------------------------------------
    # Checkpoint IO
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "shape": tuple(self.shape),
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
            "epsilon": float(self._epsilon),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        shape = tuple(state["shape"])
        if shape != self.shape:
            raise ValueError(
                f"State shape {shape} does not match self.shape {self.shape}."
            )
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
        self._epsilon = float(state.get("epsilon", self._epsilon))

    def __repr__(self) -> str:
        return (
            f"RunningMeanStd(shape={self.shape}, count={self.count:.1f}, "
            f"mean[0]={float(self.mean.flatten()[0]) if self.mean.size else 0.0:.4f}, "
            f"std[0]={float(self.std.flatten()[0]) if self.var.size else 1.0:.4f})"
        )
