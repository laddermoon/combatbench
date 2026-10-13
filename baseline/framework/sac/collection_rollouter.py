"""SAC-owned synchronous rollouter.

This is a CPU-first, copy-adapted collection layer.  It deliberately avoids
``baseline.framework.rollout`` because that package initializer loads
PPO-specific modules.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint

from .collected_episode import CollectedEpisode, blueprint_hash, mapping_hash
from .collection import SACBehaviorSpec, SACFactSpec, SACJob
from .collection_recorder import SACEpisodeRecorder
from .collection_runner import SACEpisodeRunner
from .fact_providers import build_fact_providers


class SACCollectionError(RuntimeError):
    """Raised when any collection job fails; no partial round is returned."""


def _worker_init() -> None:  # pragma: no cover - runs in child
    """Pool initializer: clamp torch to single-threaded BLAS.

    Without this, each of N spawn workers defaults to ``nproc`` OMP
    threads (192 on the GPU box) — N×192 threads thrash the machine and
    the whole point of parallel rollout is lost.  Same measure PPO's
    parallel_rollouter takes.
    """
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _serialize_job(job: SACJob) -> Dict[str, Any]:
    return {
        "policy_a": job.policy_a_bp.to_dict(),
        "policy_b": job.policy_b_bp.to_dict(),
        "env": job.env_bp.to_dict(),
        "seed": int(job.seed),
        "episode_options": _jsonable(dict(job.episode_options)),
        "behavior_a": _jsonable(job.behavior_a.to_dict()),
        "behavior_b": _jsonable(job.behavior_b.to_dict()),
        "fact_specs": [_jsonable(spec.to_dict()) for spec in job.fact_specs],
        "run_id": str(job.run_id),
        "collection_round": int(job.collection_round),
        "job_index": int(job.job_index),
        "metadata": _jsonable(dict(job.metadata)),
    }


def _deserialize_job(data: Mapping[str, Any]) -> SACJob:
    return SACJob(
        policy_a_bp=PolicyBlueprint.from_dict(data["policy_a"]),
        policy_b_bp=PolicyBlueprint.from_dict(data["policy_b"]),
        env_bp=EnvBlueprint.from_dict(data["env"]),
        seed=int(data["seed"]),
        episode_options=dict(data.get("episode_options") or {}),
        behavior_a=SACBehaviorSpec.from_dict(data["behavior_a"]),
        behavior_b=SACBehaviorSpec.from_dict(data["behavior_b"]),
        fact_specs=tuple(
            SACFactSpec.from_dict(item)
            for item in data.get("fact_specs") or ()
        ),
        run_id=str(data.get("run_id", "")),
        collection_round=int(data.get("collection_round", 0)),
        job_index=int(data.get("job_index", 0)),
        metadata=dict(data.get("metadata") or {}),
    )


def _group_key(job: SACJob) -> str:
    payload = {
        "env": job.env_bp.to_dict(),
        "policy_a": job.policy_a_bp.to_dict(),
        "policy_b": job.policy_b_bp.to_dict(),
        "fact_specs": [spec.to_dict() for spec in job.fact_specs],
    }
    return hashlib.sha256(
        pickle.dumps(payload, protocol=5)
    ).hexdigest()


def _group_jobs(jobs: Sequence[SACJob]) -> List[List[SACJob]]:
    """Group consecutive jobs by shared env/policy/fact blueprint."""
    groups: List[List[SACJob]] = []
    group_keys: List[str] = []
    for job in jobs:
        key = _group_key(job)
        if group_keys and key == group_keys[-1]:
            groups[-1].append(job)
        else:
            groups.append([job])
            group_keys.append(key)
    return groups


def _chunk_groups(
    groups: Sequence[Sequence[SACJob]], num_workers: int
) -> List[List[SACJob]]:
    """Split each group into chunks sized for the worker pool.

    Self-play rounds give every job the same group key (identical env +
    both policy blueprints), so without chunking a whole round collapses
    onto one worker — same fix shape as PPO's parallel_rollouter group
    chunking.  Order is preserved throughout: pool.map is ordered and
    chunks keep the per-group job order.
    """
    chunks: List[List[SACJob]] = []
    for group in groups:
        chunk_size = max(1, math.ceil(len(group) / num_workers))
        for i in range(0, len(group), chunk_size):
            chunks.append(list(group[i:i + chunk_size]))
    return chunks


def _provenance_for_job(job: SACJob) -> Dict[str, Any]:
    return {
        "run_id": str(job.run_id),
        "collection_round": int(job.collection_round),
        "job_index": int(job.job_index),
        "episode_seed": int(job.seed),
        "job_key": str(job.job_key),
        "policy_fingerprints": {
            "robot_a": mapping_hash(job.policy_a_bp.to_dict()),
            "robot_b": mapping_hash(job.policy_b_bp.to_dict()),
        },
        "behavior": {
            "robot_a": job.behavior_a.to_dict(),
            "robot_b": job.behavior_b.to_dict(),
        },
        "worker_id": int(os.getpid()),
    }


def _run_job_batch(batch: Sequence[Mapping[str, Any]]) -> List[CollectedEpisode]:
    """Run an ordered batch of jobs inside one worker process."""
    jobs = [_deserialize_job(item) for item in batch]
    if not jobs:
        return []

    runtime = None
    runner = None
    recorder = None
    active_key = None
    results: List[CollectedEpisode] = []

    try:
        for job in jobs:
            key = _group_key(job)
            if runtime is None or runner is None or recorder is None or key != active_key:
                if runner is not None:
                    runner.close()
                if runtime is not None:
                    runtime.close()

                recorder = SACEpisodeRecorder(
                    blueprint_hash=blueprint_hash(job.env_bp),
                    provenance=_provenance_for_job(job),
                    require_action_extras=bool(
                        job.behavior_a.require_extras
                        or job.behavior_b.require_extras
                    ),
                    require_pre_action_facts=bool(job.fact_specs),
                )
                runtime = job.env_bp.build()
                runtime.attach_recorder(recorder)
                policy_a = job.policy_a_bp.build()
                policy_b = job.policy_b_bp.build()
                fact_providers = build_fact_providers(job.fact_specs)
                runner = SACEpisodeRunner(
                    runtime,
                    policy_a,
                    policy_b,
                    recorder=recorder,
                    fact_providers=fact_providers,
                )
                active_key = key
            else:
                recorder.set_provenance(_provenance_for_job(job))

            t0 = time.perf_counter()
            runner.run_episode(
                seed=job.seed,
                options=dict(job.episode_options),
                want_extras=True,
            )
            elapsed = time.perf_counter() - t0
            episode = recorder.get_last_episode()
            results.append(
                replace(
                    episode,
                    wall_time_s=float(elapsed),
                    episode_index=int(job.job_index),
                )
            )

        return results
    except Exception as exc:
        context = ""
        if jobs:
            context = (
                f" (job_indices={[job.job_index for job in jobs]}, "
                f"collection_round={jobs[0].collection_round})"
            )
        raise SACCollectionError(f"SAC collection worker failed{context}: {exc}") from exc
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass


class SACParallelRollouter:
    """Synchronous SAC collection across ordered jobs.

    Workers are process-isolated, but the parent blocks until every job has
    completed.  Any worker/job failure raises ``SACCollectionError`` and no
    partial episode list is returned.
    """

    def __init__(
        self,
        num_workers: int = 1,
        *,
        mp_context: str = "spawn",
    ) -> None:
        if int(num_workers) < 1:
            raise ValueError(
                f"SACParallelRollouter requires num_workers >= 1, got {num_workers}"
            )
        self.num_workers = int(num_workers)
        self.mp_context = str(mp_context)
        # Persistent pool: spawn-context worker startup costs ~1.6s each
        # (interpreter boot + torch/mujoco import on first unpickle).
        # A fresh pool per collect() would re-pay that every round and
        # nullify the parallelism entirely — the pool must live as long
        # as the rollouter itself (whole training run).
        self._pool: Optional[mp.pool.Pool] = None

    def __enter__(self) -> "SACParallelRollouter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_pool(self) -> "mp.pool.Pool":
        if self._pool is None:
            ctx = mp.get_context(self.mp_context)
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def collect(self, jobs: Iterable[SACJob]) -> List[CollectedEpisode]:
        job_list = list(jobs)
        if not job_list:
            return []
        for index, job in enumerate(job_list):
            if not isinstance(job, SACJob):
                raise TypeError(
                    f"SACParallelRollouter.collect expects SACJob, got "
                    f"{type(job).__name__} at position {index}"
                )
        job_keys = [job.job_key for job in job_list]
        if len(set(job_keys)) != len(job_keys):
            raise SACCollectionError(
                "duplicate SACJob.job_key in one collection round"
            )

        groups = _group_jobs(job_list)

        if self.num_workers == 1:
            out: List[CollectedEpisode] = []
            for group in groups:
                out.extend(_run_job_batch([_serialize_job(j) for j in group]))
            return out

        chunks = [
            [_serialize_job(j) for j in chunk]
            for chunk in _chunk_groups(groups, self.num_workers)
        ]
        batches = self._get_pool().map(_run_job_batch, chunks)
        return [episode for batch in batches for episode in batch]


__all__ = ["SACCollectionError", "SACParallelRollouter"]
