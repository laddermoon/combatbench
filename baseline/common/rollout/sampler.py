"""``RolloutSampler``: variable-length episodes → fixed-size minibatches.

Two modes (see ``baseline/DESIGN.md`` §3.4):

  ``"concat"`` (default — what PPO/GRPO want)
      Concatenate every step from every episode into one (T_total, *)
      array per field. Each iteration of :meth:`__iter__` shuffles the
      step indices and yields minibatches of size ``minibatch_size``
      *steps*. Last partial minibatch is dropped by default to keep the
      gradient scale uniform; set ``drop_last=False`` to keep it.

  ``"pad"``
      Stack episodes into ``(N, T_max, *)`` per field, padded with zeros,
      with a ``mask: (N, T_max) bool`` field added automatically. Each
      iteration shuffles episode order and yields minibatches of
      ``minibatch_size`` *episodes*. Pad mode is provided for
      completeness; the canonical PPO recipe is concat. RNN/sequence
      mode is **not** supported in this MVP — DESIGN.md §8.

Independent of mode:

  - Every ``__iter__`` reshuffles, so PPO's outer ``for epoch in
    range(K_epochs):`` works without manual reseeding.
  - Per-episode arrays are validated to share their first axis length
    (``T_i``) for every field — catches "advantages of length T+1
    against actions of length T" mistakes early, in the data layer
    rather than inside the loss.
  - Tensors are produced lazily: numpy arrays live on host until a
    minibatch is yielded, then ``torch.as_tensor(..., device=device)``
    happens once.

Helper :meth:`from_batches` constructs a sampler from
``Sequence[RolloutBatch]`` plus per-episode extras (advantages /
returns) — i.e. PPO's "stitch GAE outputs onto the rollouts" step.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .batch import RolloutBatch


_MODE_CONCAT = "concat"
_MODE_PAD = "pad"


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
class RolloutSampler:
    """Yield fixed-shape minibatches over a collection of variable-length episodes.

    Parameters
    ----------
    per_episode_arrays:
        Mapping ``{field_name: list[np.ndarray]}``. The outer list is
        per-episode; each inner array has shape ``(T_i, *feature)``
        where ``T_i`` is that episode's number of steps. All fields
        must agree on ``T_i`` for every episode.
    minibatch_size:
        Steps per minibatch in concat mode; episodes per minibatch in
        pad mode.
    mode:
        ``"concat"`` or ``"pad"``.
    device:
        Where to move tensors when yielded.
    drop_last:
        Concat mode only. Drop the trailing minibatch when
        ``T_total % minibatch_size != 0``. PPO standard is True.
    seed:
        Used to seed the per-iteration shuffle. Default uses fresh
        entropy each call (so PPO's repeated epochs see different
        permutations). Pin it for tests.
    dtype:
        Default torch dtype for floating tensors. Integer / bool fields
        keep their numpy dtype.
    """

    def __init__(
        self,
        per_episode_arrays: Mapping[str, Sequence[np.ndarray]],
        *,
        minibatch_size: int,
        mode: str = _MODE_CONCAT,
        device: torch.device | str = "cpu",
        drop_last: bool = True,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if mode not in (_MODE_CONCAT, _MODE_PAD):
            raise ValueError(f"mode must be 'concat' or 'pad'; got {mode!r}")
        if minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1; got {minibatch_size}")
        if not per_episode_arrays:
            raise ValueError("per_episode_arrays is empty.")

        episode_lengths = self._validate_lengths(per_episode_arrays)
        if not episode_lengths:
            raise ValueError("Every field has zero episodes.")

        self._mode = mode
        self._minibatch_size = int(minibatch_size)
        self._device = torch.device(device)
        self._drop_last = bool(drop_last)
        self._dtype = dtype
        self._rng = np.random.default_rng(seed)

        if mode == _MODE_CONCAT:
            self._concat_arrays = {
                name: np.concatenate(list(arrs), axis=0)
                for name, arrs in per_episode_arrays.items()
            }
            self._total_steps = int(next(iter(self._concat_arrays.values())).shape[0])
        else:
            self._pad_arrays, self._mask = self._build_padded(
                per_episode_arrays, episode_lengths,
            )
            self._num_episodes = int(self._mask.shape[0])
            self._t_max = int(self._mask.shape[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def num_minibatches(self) -> int:
        if self._mode == _MODE_CONCAT:
            n = self._total_steps // self._minibatch_size
            if not self._drop_last and self._total_steps % self._minibatch_size:
                n += 1
            return max(0, n)
        # pad mode
        n = self._num_episodes // self._minibatch_size
        if not self._drop_last and self._num_episodes % self._minibatch_size:
            n += 1
        return max(0, n)

    def __len__(self) -> int:
        return self.num_minibatches

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self._mode == _MODE_CONCAT:
            return self._iter_concat()
        return self._iter_pad()

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------
    def _iter_concat(self) -> Iterator[Dict[str, torch.Tensor]]:
        order = self._rng.permutation(self._total_steps)
        mb = self._minibatch_size
        n_full = self._total_steps // mb
        end = n_full * mb if self._drop_last else self._total_steps
        for start in range(0, end, mb):
            idx = order[start : start + mb]
            yield {
                name: self._to_tensor(arr[idx])
                for name, arr in self._concat_arrays.items()
            }

    def _iter_pad(self) -> Iterator[Dict[str, torch.Tensor]]:
        order = self._rng.permutation(self._num_episodes)
        mb = self._minibatch_size
        n_full = self._num_episodes // mb
        end = n_full * mb if self._drop_last else self._num_episodes
        for start in range(0, end, mb):
            idx = order[start : start + mb]
            out: Dict[str, torch.Tensor] = {
                name: self._to_tensor(arr[idx])
                for name, arr in self._pad_arrays.items()
            }
            out["mask"] = self._to_tensor(self._mask[idx])
            yield out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if arr.dtype == np.bool_:
            return torch.as_tensor(arr, device=self._device)
        if np.issubdtype(arr.dtype, np.integer):
            return torch.as_tensor(arr, device=self._device)
        return torch.as_tensor(arr, dtype=self._dtype, device=self._device)

    @staticmethod
    def _validate_lengths(
        per_episode_arrays: Mapping[str, Sequence[np.ndarray]],
    ) -> List[int]:
        names = list(per_episode_arrays)
        # Number of episodes must agree across fields.
        n_eps_set = {len(per_episode_arrays[name]) for name in names}
        if len(n_eps_set) != 1:
            raise ValueError(
                "Inconsistent episode counts across fields: "
                f"{ {n: len(per_episode_arrays[n]) for n in names} }"
            )
        n_eps = n_eps_set.pop()
        # Per-episode T must agree across fields.
        lengths: List[int] = []
        for ep in range(n_eps):
            ep_lengths = {name: per_episode_arrays[name][ep].shape[0] for name in names}
            if len(set(ep_lengths.values())) != 1:
                raise ValueError(
                    f"Episode {ep}: inconsistent T across fields: {ep_lengths}"
                )
            lengths.append(int(next(iter(ep_lengths.values()))))
        return lengths

    @staticmethod
    def _build_padded(
        per_episode_arrays: Mapping[str, Sequence[np.ndarray]],
        episode_lengths: Sequence[int],
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        n_eps = len(episode_lengths)
        t_max = max(episode_lengths)
        padded: Dict[str, np.ndarray] = {}
        for name, arrs in per_episode_arrays.items():
            sample = arrs[0]
            shape = (n_eps, t_max) + tuple(sample.shape[1:])
            buf = np.zeros(shape, dtype=sample.dtype)
            for ep, arr in enumerate(arrs):
                buf[ep, : arr.shape[0]] = arr
            padded[name] = buf
        mask = np.zeros((n_eps, t_max), dtype=np.bool_)
        for ep, t_i in enumerate(episode_lengths):
            mask[ep, :t_i] = True
        return padded, mask

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_batches(
        cls,
        batches: Sequence[RolloutBatch],
        extras: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
        *,
        include_obs: bool = True,
        include_actions: bool = True,
        include_rewards: bool = False,
        include_log_probs: bool = True,
        include_values: bool = True,
        **sampler_kwargs: Any,
    ) -> "RolloutSampler":
        """Build a sampler from ``RolloutBatch`` list + caller-provided extras.

        ``RolloutBatch.obs`` has length ``T+1`` (initial obs + post-step
        obs). For PPO updates we want the ``T`` observations that
        *produced* the actions, so this helper slices ``obs[:-1]``
        automatically. ``final_obs`` is not included — it's only useful
        as the bootstrap input for advantage estimation, which happens
        before the sampler is built.

        ``extras`` is the place for caller-computed per-episode arrays
        like ``advantages`` / ``returns`` from a GAE pass. Each value
        must be a sequence of per-episode ``np.ndarray`` of shape
        ``(T_i, *)`` aligned with the corresponding ``RolloutBatch``.
        """
        if not batches:
            raise ValueError("from_batches needs at least one RolloutBatch.")
        per_episode: Dict[str, List[np.ndarray]] = {}

        def _push(name: str, arr: np.ndarray) -> None:
            per_episode.setdefault(name, []).append(arr)

        for b in batches:
            if include_obs:
                _push("obs", b.obs[:-1])         # drop final_obs → length T
            if include_actions:
                _push("actions", b.actions)
            if include_rewards:
                _push("rewards", b.rewards)
            if include_log_probs and b.log_probs is not None:
                _push("log_probs", b.log_probs)
            if include_values and b.values is not None:
                _push("values", b.values)

        if extras is not None:
            for name, ep_arrays in extras.items():
                if len(ep_arrays) != len(batches):
                    raise ValueError(
                        f"extras[{name!r}] has {len(ep_arrays)} episodes, "
                        f"but batches has {len(batches)}."
                    )
                for arr in ep_arrays:
                    _push(name, np.asarray(arr))

        return cls(per_episode, **sampler_kwargs)
