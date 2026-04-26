"""Process-level parallel runner for :class:`EpisodeRunner`.

``ParallelRunner`` spreads N episodes across M worker processes, each of
which constructs its **own** :class:`EnvRuntime` + policies + recorders via
a user-supplied factory. Workers are persistent (initialized once), so the
MuJoCo / plugin / observer setup cost is amortized across all episodes a
worker handles.

Why a factory, not a pre-built runner?
--------------------------------------
:class:`EnvRuntime` and most policies hold non-picklable state (MuJoCo
``MjModel``/``MjData`` handles, torch tensors on GPU, file descriptors).
Pickling a live runner across the process boundary fails or silently
shares mutable state. The factory is imported **by name** inside each
worker and builds a fresh runner from scratch — the only thing crossing
the boundary is the per-episode **seed** (int) and the returned
:class:`EpisodeResult` (pure numpy / Python primitives, picklable).

Seed determinism
----------------
``run(base_seed=X, n=N)`` derives the same N child seeds as
:meth:`EpisodeRunner.run_n_episodes`, so switching between sequential and
parallel execution **does not change which seeds are run**. Per-episode
results are bit-equal to the sequential path *if* your factory builds
deterministic runtimes (seeded simulator, seeded policies, deterministic
plugins). :class:`ParallelRunner` itself introduces no non-determinism.

Scope and non-responsibilities
------------------------------
* **GPU policies**: each worker process gets its own policy instance, so
  each is its own GPU client if the policy uses CUDA. This does NOT batch
  inference — that needs a separate ``RemotePolicy`` that proxies calls to
  a central inference server. Not in this module.
* **Vectorized physics**: MJX / Isaac / Brax batched simulators are a
  different axis of parallelism (one process, many worlds). Not covered
  here; see the ``BaseVectorizedSimulator`` TODO in ``env_runtime.py``.
* **Exception handling**: by default (``strict=True``) a failing episode
  tears down the whole batch — the Pool is terminated and the exception
  is re-raised at the caller. With ``strict=False`` failed episodes are
  replaced by ``None`` in the result list and a traceback is logged.

Example
-------
.. code-block:: python

    # Factory MUST be top-level importable (no lambdas, no closures).
    def make_runner(worker_id: int) -> EpisodeRunner:
        runtime = build_runtime(device=f"cuda:{worker_id % n_gpus}")
        return EpisodeRunner(
            runtime=runtime,
            policies={"robot_a": load_policy_a(), "robot_b": load_policy_b()},
            rollout=RolloutConfig(capture_a=True, capture_b=False),
        )

    with ParallelRunner(make_runner, num_workers=8) as pr:
        results = pr.run(n=1000, base_seed=42)

``num_workers<=1`` short-circuits the multiprocessing path entirely:
the runner is built once in-process and ``run_n_episodes`` is called
directly. Useful for debugging and unit tests.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .episode_runner import (
    EpisodeResult,
    EpisodeRunner,
    _derive_batch_seeds,
    _resolve_seed,
)


_logger = logging.getLogger("combatbench.envs.framework.parallel_runner")


# Public alias for the factory signature. The argument is the worker index
# (0..num_workers-1) so users can shard GPUs or bind to CPU cores.
RunnerFactory = Callable[[int], EpisodeRunner]


# ---------------------------------------------------------------------------
# Worker-side state
# ---------------------------------------------------------------------------
# Each worker process holds a single long-lived EpisodeRunner. It is set up
# once via ``_worker_init`` and reused for every task routed to that worker.
# Module-global is the standard multiprocessing idiom here — workers re-import
# this module on spawn, so the global is freshly initialized per process.
_WORKER_RUNNER: Optional[EpisodeRunner] = None
_WORKER_ID: int = -1


def _worker_init(factory: RunnerFactory, worker_id_counter) -> None:  # pragma: no cover - runs in child
    """Pool initializer: build this worker's :class:`EpisodeRunner` once."""
    global _WORKER_RUNNER, _WORKER_ID
    with worker_id_counter.get_lock():
        worker_id = worker_id_counter.value
        worker_id_counter.value += 1
    _WORKER_ID = worker_id
    _WORKER_RUNNER = factory(worker_id)


def _run_one(task: Tuple[int, Optional[Dict[str, Any]]]) -> EpisodeResult:  # pragma: no cover - runs in child
    """Task function: run one episode on this worker's cached runner.

    ``task`` is a ``(seed, options)`` pair; ``options`` may be ``None``.
    Packing both into a single argument keeps the multiprocessing imap
    interface unchanged.
    """
    assert _WORKER_RUNNER is not None, (
        "ParallelRunner worker was not initialized; did _worker_init run?"
    )
    seed, options = task
    return _WORKER_RUNNER.run_episode(seed=int(seed), options=options)


def _run_one_best_effort(task: Tuple[int, Optional[Dict[str, Any]]]):  # pragma: no cover - runs in child
    """Best-effort variant used when ``strict=False``: returns ``(seed, result, error_str)``
    where at most one of ``result`` / ``error_str`` is non-None."""
    seed, _options = task
    try:
        return int(seed), _run_one(task), None
    except BaseException:  # noqa: BLE001 - we rethrow/log at the boundary
        return int(seed), None, traceback.format_exc()


# ---------------------------------------------------------------------------
# ParallelRunner
# ---------------------------------------------------------------------------
class ParallelRunner(AbstractContextManager):
    """Run many episodes in parallel via a process pool.

    Parameters
    ----------
    runner_factory: picklable callable ``(worker_id: int) -> EpisodeRunner``
        that constructs a fresh runner inside each worker. MUST be a
        top-level function or a class — no lambdas, no closures over
        un-picklable state.
    num_workers: number of worker processes. ``<= 1`` selects the
        in-process fast path.
    mp_start_method: ``"spawn"`` (default — safest with MuJoCo / CUDA /
        torch), ``"forkserver"``, or ``"fork"``. Only honored when
        ``num_workers > 1``.
    strict: if True (default) a failing episode aborts the whole batch.
        If False, failed episodes come back as ``None`` in the result list
        and a traceback is logged.
    """

    def __init__(
        self,
        runner_factory: RunnerFactory,
        num_workers: int = 1,
        *,
        mp_start_method: str = "spawn",
        strict: bool = True,
    ) -> None:
        if not callable(runner_factory):
            raise TypeError(
                f"runner_factory must be callable, got {type(runner_factory).__name__}"
            )
        self._factory = runner_factory
        self._num_workers = max(1, int(num_workers))
        self._mp_start_method = mp_start_method
        self._strict = bool(strict)
        self._pool: Optional[mp.pool.Pool] = None
        self._inproc_runner: Optional[EpisodeRunner] = None
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def num_workers(self) -> int:
        return self._num_workers

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("ParallelRunner is closed.")
        if self._num_workers <= 1:
            if self._inproc_runner is None:
                self._inproc_runner = self._factory(0)
            return
        if self._pool is not None:
            return
        ctx = mp.get_context(self._mp_start_method)
        # Shared counter hands out stable worker ids across the pool.
        worker_id_counter = ctx.Value("i", 0)
        self._pool = ctx.Pool(
            processes=self._num_workers,
            initializer=_worker_init,
            initargs=(self._factory, worker_id_counter),
        )

    def close(self) -> None:
        """Terminate workers and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                self._pool.terminate()
                self._pool.join()
            self._pool = None
        if self._inproc_runner is not None:
            close_fn = getattr(self._inproc_runner, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    _logger.exception("In-process runner close() failed")
            self._inproc_runner = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        n: int,
        *,
        base_seed: Optional[int] = None,
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
    ) -> List[Optional[EpisodeResult]]:
        """Run ``n`` episodes; returns results in **seed-submission order**.

        Child seeds are derived by :class:`numpy.random.SeedSequence(base_seed)`
        exactly as :meth:`EpisodeRunner.run_n_episodes` does, so
        ``ParallelRunner(...).run(n, base_seed=X)`` and
        ``EpisodeRunner(...).run_n_episodes(n, base_seed=X)`` produce the
        same trajectories (modulo whatever non-determinism lives inside
        the simulator or policies themselves).

        ``options_fn(episode_index) -> options_dict`` is called **on the
        main process** for each of the ``n`` episodes; the resulting dict
        is shipped to workers alongside the seed and used as
        ``runtime.reset(options=...)`` for that episode. See
        ``framework/RESET.md`` §4 for what belongs in ``options``.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative; got {n}")
        if n == 0:
            return []
        seeds = _derive_seeds(base_seed, n)
        tasks = self._build_tasks(seeds, options_fn)
        self._ensure_started()

        if self._pool is None:
            # Single-process fast path.
            assert self._inproc_runner is not None
            if self._strict:
                return [
                    self._inproc_runner.run_episode(seed=int(s), options=opts)
                    for s, opts in tasks
                ]
            results: List[Optional[EpisodeResult]] = []
            for s, opts in tasks:
                try:
                    results.append(self._inproc_runner.run_episode(seed=int(s), options=opts))
                except BaseException:  # noqa: BLE001
                    _logger.exception("Episode with seed=%s failed", s)
                    results.append(None)
            return results

        # Multi-process path.
        if self._strict:
            # imap preserves order; exceptions re-raise on the main side.
            try:
                return list(self._pool.imap(_run_one, iter(tasks)))
            except BaseException:
                # Re-raise after draining: the pool is poisoned — shut it down
                # so the caller doesn't hang on __exit__.
                self._hard_kill()
                raise
        # Best-effort: never raises; None for failed episodes.
        out: List[Optional[EpisodeResult]] = [None] * len(seeds)
        seed_to_idx = {int(s): i for i, s in enumerate(seeds)}
        for seed, result, err in self._pool.imap_unordered(
            _run_one_best_effort, iter(tasks)
        ):
            if err is not None:
                _logger.error("Episode with seed=%s failed:\n%s", seed, err)
            out[seed_to_idx[seed]] = result
        return out

    def run_iter(
        self,
        n: int,
        *,
        base_seed: Optional[int] = None,
        ordered: bool = True,
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
    ) -> Iterator[Optional[EpisodeResult]]:
        """Generator variant of :meth:`run`.

        ``ordered=True`` (default) preserves seed-submission order — useful
        when you want reproducibility. ``ordered=False`` yields as-completed
        for lower wall-time when the consumer doesn't care about order
        (e.g. on-policy RL collectors that feed directly into a replay
        buffer).

        ``options_fn`` semantics match :meth:`run`.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative; got {n}")
        if n == 0:
            return
        seeds = _derive_seeds(base_seed, n)
        tasks = self._build_tasks(seeds, options_fn)
        self._ensure_started()

        if self._pool is None:
            assert self._inproc_runner is not None
            for s, opts in tasks:
                if self._strict:
                    yield self._inproc_runner.run_episode(seed=int(s), options=opts)
                else:
                    try:
                        yield self._inproc_runner.run_episode(seed=int(s), options=opts)
                    except BaseException:
                        _logger.exception("Episode with seed=%s failed", s)
                        yield None
            return

        iterable = iter(tasks)
        if self._strict:
            imap_fn = self._pool.imap if ordered else self._pool.imap_unordered
            try:
                for result in imap_fn(_run_one, iterable):
                    yield result
            except BaseException:
                self._hard_kill()
                raise
            return
        # Best-effort streaming.
        if ordered:
            out: List[Optional[EpisodeResult]] = [None] * len(seeds)
            delivered = [False] * len(seeds)
            seed_to_idx = {int(s): i for i, s in enumerate(seeds)}
            next_emit = 0
            for seed, result, err in self._pool.imap_unordered(
                _run_one_best_effort, iterable
            ):
                if err is not None:
                    _logger.error("Episode with seed=%s failed:\n%s", seed, err)
                idx = seed_to_idx[seed]
                out[idx] = result
                delivered[idx] = True
                while next_emit < len(seeds) and delivered[next_emit]:
                    yield out[next_emit]
                    next_emit += 1
        else:
            for _seed, result, err in self._pool.imap_unordered(
                _run_one_best_effort, iterable
            ):
                if err is not None:
                    _logger.error("Episode failed:\n%s", err)
                yield result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tasks(
        seeds,
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]],
    ) -> List[Tuple[int, Optional[Dict[str, Any]]]]:
        """Pair each seed with its per-episode options (computed on the main
        process so workers don't need to import ``options_fn``)."""
        if options_fn is None:
            return [(int(s), None) for s in seeds]
        return [(int(s), options_fn(i)) for i, s in enumerate(seeds)]

    def _hard_kill(self) -> None:
        """Tear down the pool hard — used on strict-mode exceptions so the
        context manager's ``__exit__`` does not hang on drained workers."""
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            finally:
                self._pool = None


# ---------------------------------------------------------------------------
# Seed derivation (shared with EpisodeRunner.run_n_episodes via
# envs.framework.episode_runner._derive_batch_seeds)
# ---------------------------------------------------------------------------
def _derive_seeds(base_seed: Optional[int], n: int) -> np.ndarray:
    """Return ``n`` per-episode ``uint32`` seeds.

    ``base_seed=None`` is resolved at entry to a concrete ``uint32`` (see
    :func:`envs.framework.episode_runner._resolve_seed`) and logged, so
    the batch is reproducible even when the caller didn't supply a seed.
    Derivation is bit-equal to :meth:`EpisodeRunner.run_n_episodes`.
    """
    resolved = _resolve_seed(base_seed)
    _logger.info("ParallelRunner: base_seed=%d, n=%d", resolved, n)
    return _derive_batch_seeds(resolved, n)
