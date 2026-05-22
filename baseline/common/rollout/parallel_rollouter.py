"""Lightweight parallel episode collector.

Each job is fully specified by a tuple::

    (policy_a_blueprint, policy_b_blueprint, env_blueprint, seed)

``robot_a`` and ``robot_b`` may use different policies.
The collector returns a flat ``List[Episode]`` in the same order as
``jobs``.

No training-specific features (state-dict hot-reload, deterministic
switch, debug plugins, episode options, etc.) — this is intentionally
a thin batch executor.

Efficiency notes
--------------
* Worker processes cache built :class:`EnvRuntime` and :class:`Policy`
  instances keyed by blueprint hash, so identical blueprints across
  multiple seeds are reused without rebuild cost.
* ``num_workers <= 1`` short-circuits to in-process sequential execution.
* ``num_workers > 1`` uses :class:`concurrent.futures.ProcessPoolExecutor`
  with persistent worker processes (lives until :meth:`close`).
* Python 3.14 free-threading (PEP 703) would theoretically allow a
  :class:`concurrent.futures.ThreadPoolExecutor` alternative with lower
  IPC overhead, but each thread still needs its own ``EnvRuntime`` +
  ``Policy`` instances because neither MuJoCo nor PyTorch ``nn.Module``
  is thread-safe for concurrent use on a single instance.
"""
from __future__ import annotations

import json
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

# ---------------------------------------------------------------------------
# Worker-side caches (module-level → persists across tasks in one worker)
# ---------------------------------------------------------------------------
# (env_blueprint_hash -> (EnvBlueprint, EnvRuntime, EpisodeRecorder))
_worker_env_cache: Dict[str, Tuple[Any, Any, EpisodeRecorder]] = {}

# (policy_blueprint_hash -> (PolicyBlueprint, Policy))
_worker_policy_cache: Dict[str, Tuple[Any, Any]] = {}


def _build_env(env_bp_dict: Dict[str, Any]) -> Tuple[Any, EpisodeRecorder]:
    """Build or retrieve a cached EnvRuntime + EpisodeRecorder."""
    env_bp = EnvBlueprint.from_dict(env_bp_dict)
    env_hash = blueprint_hash(env_bp)
    if env_hash not in _worker_env_cache:
        recorder = EpisodeRecorder(blueprint_hash=env_hash)
        runtime = env_bp.build(recorders=[recorder])
        _worker_env_cache[env_hash] = (env_bp, runtime, recorder)
    else:
        _, runtime, recorder = _worker_env_cache[env_hash]
    return runtime, recorder


def _build_policy(policy_bp_dict: Dict[str, Any]) -> Any:
    """Build or retrieve a cached Policy."""
    policy_bp = PolicyBlueprint.from_dict(policy_bp_dict)
    # Stable hash over the full blueprint payload
    policy_hash = json.dumps(policy_bp.to_dict(), sort_keys=True)
    if policy_hash not in _worker_policy_cache:
        policy = policy_bp.build()
        _worker_policy_cache[policy_hash] = (policy_bp, policy)
    else:
        _, policy = _worker_policy_cache[policy_hash]
    return policy


def _run_job(
    policy_a_bp_dict: Dict[str, Any],
    policy_b_bp_dict: Dict[str, Any],
    env_bp_dict: Dict[str, Any],
    seed: int,
    options: Optional[Dict[str, Any]] = None,
) -> Episode:
    """Run one episode and return its :class:`Episode`."""
    runtime, recorder = _build_env(env_bp_dict)
    policy_a = _build_policy(policy_a_bp_dict)
    policy_b = _build_policy(policy_b_bp_dict)
    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=policy_a,
        policy_b=policy_b,
    )
    runner.run_episode(seed=seed, options=options, want_extras=True)
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
            for policy_a_bp, policy_b_bp, env_bp, seed, *rest in jobs
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
