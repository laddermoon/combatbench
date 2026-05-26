"""Lightweight parallel episode collector.

Each job is fully specified by a tuple::

    (policy_a_blueprint, policy_b_blueprint, env_blueprint, seed, options)

``robot_a`` and ``robot_b`` may use different policies.
The collector returns a flat ``List[Episode]`` in the same order as
``jobs``.

Each worker creates a fresh EnvRuntime + Policy per job — no shared
state, no caching, no memory accumulation.  Short-lived MuJoCo
instances are cheap to create and are GC'd when the function returns.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from envs.framework.blueprint import EnvBlueprint
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.policy import PolicyBlueprint

from .episode import Episode, blueprint_hash
from .episode_collection import EpisodeCollection
from .episode_recorder import EpisodeRecorder

_logger = logging.getLogger(__name__)


def _worker_init() -> None:  # pragma: no cover - runs in child
    """Pool initializer: clamp torch to single-threaded BLAS so N workers
    don't fight each other on shared thread pools."""
    try:
        import torch  # noqa: WPS433 - optional dep

        torch.set_num_threads(1)
        with __import__('contextlib').suppress(RuntimeError):
            torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _run_job(
    policy_a_bp_dict: Dict[str, Any],
    policy_b_bp_dict: Dict[str, Any],
    env_bp_dict: Dict[str, Any],
    seed: int,
    options: Optional[Dict[str, Any]] = None,
) -> Episode:
    """Run one episode: create env + policies from scratch, collect, return."""
    #import time as _time
    #_t0 = _time.perf_counter()

    env_bp = EnvBlueprint.from_dict(env_bp_dict)
    env_hash = blueprint_hash(env_bp)

    recorder = EpisodeRecorder(blueprint_hash=env_hash)
    #_t1 = _time.perf_counter()
    runtime = env_bp.build(recorders=[recorder])
    #_t2 = _time.perf_counter()
    policy_a = PolicyBlueprint.from_dict(policy_a_bp_dict).build()
    policy_b = PolicyBlueprint.from_dict(policy_b_bp_dict).build()
    #_t3 = _time.perf_counter()

    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=policy_a,
        policy_b=policy_b,
    )
    runner.run_episode(seed=seed, options=options, want_extras=True)
    #_t4 = _time.perf_counter()

    #print(
    #    f"[worker] seed={seed} init={_t3 - _t0:.3f}s"
    #    f"(env={_t2 - _t1:.3f}s policy={_t3 - _t2:.3f}s)"
    #    f" episode={_t4 - _t3:.3f}s",
    #    flush=True,
    #)
    return recorder.get_last_episode()


# ---------------------------------------------------------------------------
# ParallelRollouter
# ---------------------------------------------------------------------------
class ParallelRollouter:
    """Collect :class:`Episode`s in parallel from blueprint tuples.

    Parameters
    ----------
    num_workers:
        ``<= 1`` runs everything in the calling process.
        ``> 1`` spawns a persistent process pool.
    mp_context:
        Multiprocessing start method (default ``"spawn"``).
    """

    def __init__(
        self,
        num_workers: int = 1,
        mp_context: str = "spawn",
    ) -> None:
        self._num_workers = max(1, int(num_workers))
        self._mp_context = mp_context
        self._executor: Optional[ProcessPoolExecutor] = None

        if self._num_workers > 1:
            ctx = mp.get_context(mp_context)
            self._executor = ProcessPoolExecutor(
                max_workers=self._num_workers,
                mp_context=ctx,
                initializer=_worker_init,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(
        self,
        jobs: Sequence[
            Tuple[
                PolicyBlueprint,
                PolicyBlueprint,
                EnvBlueprint,
                int,
                Optional[Dict[str, Any]],
            ]
        ],
    ) -> List[Episode]:
        """Run all jobs and return a list of :class:`Episode`.

        ``robot_a`` and ``robot_b`` may use different policies.
        Env blueprints may also differ across jobs; there is no uniformity
        requirement.

        Parameters
        ----------
        jobs:
            ``(policy_a_blueprint, policy_b_blueprint, env_blueprint, seed,
            options)`` per episode. ``options`` is optional and forwarded
            to :meth:`EpisodeRunner.run_episode`.

        Returns
        -------
        List[Episode]
            Episodes in the same order as ``jobs``.
        """
        if not jobs:
            raise ValueError("jobs must not be empty")

        # Serialize blueprints to plain dicts for pickling into workers
        tasks = [
            (
                policy_a_bp.to_dict(),
                policy_b_bp.to_dict(),
                env_bp.to_dict(),
                int(seed),
                dict(options) if options is not None else None,
            )
            for policy_a_bp, policy_b_bp, env_bp, seed, options in jobs
        ]

        if self._num_workers <= 1:
            episodes = [_run_job(*task) for task in tasks]
        else:
            assert self._executor is not None
            policy_a_dicts, policy_b_dicts, env_dicts, seeds, options_list = zip(*tasks)
            episodes = list(
                self._executor.map(
                    _run_job,
                    policy_a_dicts,
                    policy_b_dicts,
                    env_dicts,
                    seeds,
                    options_list,
                )
            )

        return episodes

    def close(self) -> None:
        """Shut down the worker pool (idempotent)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ParallelRollouter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


__all__ = ["ParallelRollouter"]
