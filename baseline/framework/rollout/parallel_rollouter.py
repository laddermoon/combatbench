"""Lightweight parallel episode collector.

Each job is a :class:`Job` (frozen dataclass) specifying two policy
blueprints, an env blueprint, a seed, env-only options, and per-policy
explore_intensity.

``robot_a`` and ``robot_b`` may use different policies.
The collector returns a flat ``List[Episode]`` in the same order as
``jobs``.

Workers reuse EnvRuntime + Policy instances across episodes that share
the same blueprint, avoiding repeated MuJoCo model loading and policy
deserialization.  When blueprints change (e.g. different agent_id),
the old env is torn down and a new one is created.
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from envs.framework.blueprint import EnvBlueprint
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.policy import PolicyBlueprint

from .episode import Episode, blueprint_hash
from .episode_collection import EpisodeCollection
from .episode_recorder import EpisodeRecorder
from .exploratory_policy import ExploratoryPolicy
from .job import EiSpec, Job

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
    options: Optional[Dict[str, Any]],
    ei_a: EiSpec,
    ei_b: EiSpec,
) -> Episode:
    """Run one episode: create env + policies from scratch, collect, return."""
    env_bp = EnvBlueprint.from_dict(env_bp_dict)
    env_hash = blueprint_hash(env_bp)

    recorder = EpisodeRecorder(blueprint_hash=env_hash)
    runtime = env_bp.build(recorders=[recorder])
    policy_a = PolicyBlueprint.from_dict(policy_a_bp_dict).build()
    policy_b = PolicyBlueprint.from_dict(policy_b_bp_dict).build()

    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=ExploratoryPolicy(policy_a, explore_intensity=ei_a),
        policy_b=ExploratoryPolicy(policy_b, explore_intensity=ei_b),
    )
    runner.run_episode(
        seed=seed, options=options, want_extras=True,
    )
    return recorder.get_last_episode()


def _run_job_batch(
    tasks: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int, Optional[Dict[str, Any]], EiSpec, EiSpec]],
) -> List[Episode]:
    """Run a batch of jobs, reusing EnvRuntime + Policy when blueprints match.

    Fine-grained reuse: if only the policy changed (common case — same env,
    new policy weights), only the policy is rebuilt via ``set_policy_*``.
    If only the env changed, only the runtime is rebuilt via ``set_runtime``.
    When policy_a == policy_b, a single Policy instance is built and shared.
    """
    episodes: List[Episode] = []
    runner: Optional[EpisodeRunner] = None
    recorder: Optional[EpisodeRecorder] = None
    current_env_key: Optional[str] = None
    current_pa_key: Optional[str] = None
    current_pb_key: Optional[str] = None

    for policy_a_bp_dict, policy_b_bp_dict, env_bp_dict, seed, options, ei_a, ei_b in tasks:
        env_key = json.dumps(env_bp_dict, sort_keys=True, ensure_ascii=False)
        pa_key = json.dumps(policy_a_bp_dict, sort_keys=True, ensure_ascii=False)
        pb_key = json.dumps(policy_b_bp_dict, sort_keys=True, ensure_ascii=False)
        same_policy = pa_key == pb_key

        env_changed = env_key != current_env_key
        pa_changed = pa_key != current_pa_key
        pb_changed = pb_key != current_pb_key

        if runner is None or env_changed:
            # Full (re)build — env is the expensive part.
            if runner is not None:
                runner.close()
                runner.runtime.close()

            env_bp = EnvBlueprint.from_dict(env_bp_dict)
            env_hash = blueprint_hash(env_bp)
            recorder = EpisodeRecorder(blueprint_hash=env_hash)
            runtime = env_bp.build(recorders=[recorder])
            policy_a = PolicyBlueprint.from_dict(policy_a_bp_dict).build()
            policy_b = policy_a if same_policy else PolicyBlueprint.from_dict(policy_b_bp_dict).build()
            runner = EpisodeRunner(
                runtime=runtime,
                policy_a=ExploratoryPolicy(policy_a, explore_intensity=ei_a),
                policy_b=ExploratoryPolicy(policy_b, explore_intensity=ei_b),
            )
            current_env_key = env_key
            current_pa_key = pa_key
            current_pb_key = pb_key
        else:
            # Env unchanged — only update policies that changed.
            if pa_changed:
                new_pa = PolicyBlueprint.from_dict(policy_a_bp_dict).build()
                runner.set_policy_a(ExploratoryPolicy(new_pa, explore_intensity=ei_a))
                if same_policy:
                    runner.set_policy_b(ExploratoryPolicy(new_pa, explore_intensity=ei_b))
                current_pa_key = pa_key
            if pb_changed and not same_policy:
                new_pb = PolicyBlueprint.from_dict(policy_b_bp_dict).build()
                runner.set_policy_b(ExploratoryPolicy(new_pb, explore_intensity=ei_b))
                current_pb_key = pb_key

        runner.run_episode(
            seed=seed, options=options, want_extras=True,
        )
        episodes.append(recorder.get_last_episode())

    if runner is not None:
        runner.close()
        runner.runtime.close()

    return episodes


def _run_chunk(
    indexed_tasks: List[Tuple[int, Tuple]],
) -> List[Episode]:
    """Worker entry point: extract tasks from indexed pairs and run as a batch."""
    tasks = [t for _, t in indexed_tasks]
    return _run_job_batch(tasks)


# ---------------------------------------------------------------------------
# ParallelRollouter
# ---------------------------------------------------------------------------
class ParallelRollouter:
    """Collect :class:`Episode`s in parallel from :class:`Job` objects.

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
        jobs: Sequence[Job],
    ) -> List[Episode]:
        """Run all jobs and return a list of :class:`Episode`.

        ``robot_a`` and ``robot_b`` may use different policies.
        Env blueprints may also differ across jobs; there is no uniformity
        requirement.

        Parameters
        ----------
        jobs:
            :class:`Job` instances, one per episode.

        Returns
        -------
        List[Episode]
            Episodes in the same order as ``jobs``.
        """
        if not jobs:
            raise ValueError("jobs must not be empty")

        # Serialize blueprints to plain dicts for pickling into workers.
        # explore_intensity (float or callable) is passed through as-is;
        # callables must be top-level functions to be picklable.
        tasks = [
            (
                job.policy_a_bp.to_dict(),
                job.policy_b_bp.to_dict(),
                job.env_bp.to_dict(),
                int(job.seed),
                dict(job.episode_options) if job.episode_options else None,
                job.explore_intensity_a,
                job.explore_intensity_b,
            )
            for job in jobs
        ]

        if self._num_workers <= 1:
            episodes = _run_job_batch(tasks)
        else:
            assert self._executor is not None
            # Group tasks by blueprint identity so that each group can
            # reuse a single EnvRuntime + Policy across all its episodes.
            groups: Dict[str, List[Tuple[int, Tuple]]] = {}
            for i, task in enumerate(tasks):
                key = json.dumps(
                    {"pa": task[0], "pb": task[1], "env": task[2]},
                    sort_keys=True, ensure_ascii=False,
                )
                groups.setdefault(key, []).append((i, task))

            # Split each group into chunks sized for the worker pool.
            all_chunks: List[List[Tuple[int, Tuple]]] = []
            for indexed_tasks in groups.values():
                chunk_size = max(1, math.ceil(len(indexed_tasks) / self._num_workers))
                for j in range(0, len(indexed_tasks), chunk_size):
                    all_chunks.append(indexed_tasks[j:j + chunk_size])

            chunk_results = list(self._executor.map(_run_chunk, all_chunks))

            # Reassemble episodes in original order.
            episodes: List[Optional[Episode]] = [None] * len(tasks)
            for chunk, chunk_eps in zip(all_chunks, chunk_results):
                for (orig_idx, _), ep in zip(chunk, chunk_eps):
                    episodes[orig_idx] = ep

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
