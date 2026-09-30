"""Lightweight parallel episode collector.

Each job is a :class:`Job` (frozen dataclass) specifying two policy
blueprints, an env blueprint, a seed, env-only options, and per-policy
sampling specs (``SamplingSpec`` — explore_factor + optional reference
policies + delta config).

``robot_a`` and ``robot_b`` may use different policies.
The collector returns a flat ``List[Episode]`` in the same order as
``jobs``.

Workers reuse EnvRuntime + Policy instances across episodes that share
the same blueprint, avoiding repeated MuJoCo model loading and policy
deserialization.  When blueprints change (e.g. different agent_id),
the old env is torn down and a new one is created.

**Spec-aware reuse**: a job's ``SamplingSpec`` is part of the identity
of the *wrapped* policy handed to the runner — two jobs sharing the
same policy blueprint but different specs must NOT reuse the previous
wrapper (it would silently apply the wrong explore_factor / reference).
To keep blueprint reuse cheap, the batch loop keeps the *unwrapped*
inner policy separate from its wrapper: a blueprint change rebuilds the
inner; a blueprint **or spec** change rebuilds only the wrapper.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from envs.framework.blueprint import EnvBlueprint
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.policy import PolicyBlueprint

from .episode import Episode, blueprint_hash
from .episode_collection import EpisodeCollection
from .episode_recorder import EpisodeRecorder
from .exploratory_policy import SamplingPolicy
from .job import Job, SamplingSpec
from .inference_server import InferenceServerHandle
from .remote_policy import RemoteSamplingPolicy

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


def _spec_key(spec_dict: Dict[str, Any]) -> str:
    """Stable identity key for a serialized SamplingSpec.

    ``explore_factor`` may be a callable (not JSON-serializable), so the
    key is the md5 of the pickled dict — deterministic within a process
    and sufficient for grouping/change detection inside one collect()
    call.
    """
    return hashlib.md5(pickle.dumps(spec_dict)).hexdigest()


def _is_remote_eligible(
    policy_bp_dict: Dict[str, Any],
    spec_dict: Dict[str, Any],
    stochastic: bool,
) -> bool:
    """True when this agent's wrapper should be a RemoteSamplingPolicy.

    Remote inference applies only to the stochastic rollout path and only
    to file-exported policies (``cls: "file:..."``) — the convention that
    guarantees a ``._policy.sample_action`` net on the server.  Scripted
    (``module:Class``) blueprints and the deterministic eval path keep
    the local build/wrap behavior untouched.
    """
    return bool(
        stochastic
        and spec_dict.get("_remote_addr")
        and str(policy_bp_dict.get("cls", "")).startswith("file:")
    )


def _wrap_policy(
    policy,
    spec_dict: Dict[str, Any],
    stochastic: bool,
    policy_bp_dict: Optional[Dict[str, Any]] = None,
):
    """Wrap a policy for the EpisodeRunner.

    When ``stochastic=True``, wrap in :class:`SamplingPolicy` so
    ``sample()`` is called with a per-frame SamplingContext built from
    the spec.  When ``stochastic=False``, return the policy as-is so
    ``act()`` (deterministic) is called directly — specs (including any
    reference policies) are never consumed on the eval path.

    If the spec carries ``_remote_addr`` (injected by ``collect`` in GPU
    inference mode) and the blueprint is a file export, the inner policy
    is discarded and a :class:`RemoteSamplingPolicy` shell is returned —
    the whole sampling semantics then runs batched on the inference
    server while ``EpisodeRunner`` still sees a plain Policy.
    """
    if not stochastic:
        return policy
    if policy_bp_dict is not None and _is_remote_eligible(
        policy_bp_dict, spec_dict, stochastic,
    ):
        return RemoteSamplingPolicy(policy_bp_dict, spec_dict)
    return SamplingPolicy(policy, SamplingSpec.from_dict(spec_dict))


def _run_job(
    policy_a_bp_dict: Dict[str, Any],
    policy_b_bp_dict: Dict[str, Any],
    env_bp_dict: Dict[str, Any],
    seed: int,
    options: Optional[Dict[str, Any]],
    spec_a_dict: Dict[str, Any],
    spec_b_dict: Dict[str, Any],
    stochastic: bool,
) -> Episode:
    """Run one episode: create env + policies from scratch, collect, return."""
    env_bp = EnvBlueprint.from_dict(env_bp_dict)
    env_hash = blueprint_hash(env_bp)

    recorder = EpisodeRecorder(blueprint_hash=env_hash)
    runtime = env_bp.build(recorders=[recorder])
    remote_a = _is_remote_eligible(policy_a_bp_dict, spec_a_dict, stochastic)
    remote_b = _is_remote_eligible(policy_b_bp_dict, spec_b_dict, stochastic)
    policy_a = (
        None if remote_a
        else PolicyBlueprint.from_dict(policy_a_bp_dict).build()
    )
    policy_b = (
        None if remote_b
        else PolicyBlueprint.from_dict(policy_b_bp_dict).build()
    )

    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=_wrap_policy(policy_a, spec_a_dict, stochastic, policy_a_bp_dict),
        policy_b=_wrap_policy(policy_b, spec_b_dict, stochastic, policy_b_bp_dict),
    )
    runner.run_episode(
        seed=seed, options=options, want_extras=True,
    )
    return recorder.get_last_episode()


def _run_job_batch(
    tasks: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int, Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any], bool]],
) -> List[Episode]:
    """Run a batch of jobs, reusing EnvRuntime + Policy when blueprints match.

    Fine-grained reuse: if only the policy changed (common case — same env,
    new policy weights), only the policy is rebuilt via ``set_policy_*``.
    If only the env changed, only the runtime is rebuilt via ``set_runtime``.
    When policy_a == policy_b, a single Policy instance is built and shared.

    The *wrapper* identity additionally includes the SamplingSpec: a spec
    change (different explore_factor / reference / delta config) forces a
    re-wrap even when the inner policy blueprint is unchanged.  The inner
    policy itself is only rebuilt when its blueprint changes, so spec-only
    variation stays cheap.
    """
    episodes: List[Episode] = []
    runner: Optional[EpisodeRunner] = None
    recorder: Optional[EpisodeRecorder] = None
    current_env_key: Optional[str] = None
    current_pa_key: Optional[str] = None
    current_pb_key: Optional[str] = None
    current_sa_key: Optional[str] = None
    current_sb_key: Optional[str] = None
    # Unwrapped inner policies — reused across spec-only changes.
    inner_a = None
    inner_b = None

    for policy_a_bp_dict, policy_b_bp_dict, env_bp_dict, seed, options, spec_a_dict, spec_b_dict, stochastic in tasks:
        env_key = json.dumps(env_bp_dict, sort_keys=True, ensure_ascii=False)
        pa_key = json.dumps(policy_a_bp_dict, sort_keys=True, ensure_ascii=False)
        pb_key = json.dumps(policy_b_bp_dict, sort_keys=True, ensure_ascii=False)
        sa_key = _spec_key(spec_a_dict)
        sb_key = _spec_key(spec_b_dict)
        same_policy = pa_key == pb_key

        env_changed = env_key != current_env_key
        pa_changed = pa_key != current_pa_key
        pb_changed = pb_key != current_pb_key
        sa_changed = sa_key != current_sa_key
        sb_changed = sb_key != current_sb_key

        # Remote-eligible agents skip the local inner build entirely —
        # their network lives on the inference server; the worker only
        # ships obs + noise and receives action + extras.
        remote_a = _is_remote_eligible(policy_a_bp_dict, spec_a_dict, stochastic)
        remote_b = _is_remote_eligible(policy_b_bp_dict, spec_b_dict, stochastic)

        if runner is None or env_changed:
            # Full (re)build — env is the expensive part.
            if runner is not None:
                runner.close()
                runner.runtime.close()

            env_bp = EnvBlueprint.from_dict(env_bp_dict)
            env_hash = blueprint_hash(env_bp)
            recorder = EpisodeRecorder(blueprint_hash=env_hash)
            runtime = env_bp.build(recorders=[recorder])
            inner_a = (
                None if remote_a
                else PolicyBlueprint.from_dict(policy_a_bp_dict).build()
            )
            inner_b = (
                inner_a if same_policy
                else (None if remote_b
                      else PolicyBlueprint.from_dict(policy_b_bp_dict).build())
            )
            runner = EpisodeRunner(
                runtime=runtime,
                policy_a=_wrap_policy(
                    inner_a, spec_a_dict, stochastic, policy_a_bp_dict,
                ),
                policy_b=_wrap_policy(
                    inner_b, spec_b_dict, stochastic, policy_b_bp_dict,
                ),
            )
            current_env_key = env_key
            current_pa_key = pa_key
            current_pb_key = pb_key
            current_sa_key = sa_key
            current_sb_key = sb_key
        else:
            # Env unchanged — rebuild inner policies only when their
            # blueprints changed; re-wrap when blueprint OR spec changed.
            if pa_changed and not remote_a:
                inner_a = PolicyBlueprint.from_dict(policy_a_bp_dict).build()
            if same_policy:
                # Shared inner: covers pa_changed AND transitions back
                # into self-play (A!=B → A==B leaves a stale inner_b).
                inner_b = inner_a
            elif pb_changed and not remote_b:
                inner_b = PolicyBlueprint.from_dict(policy_b_bp_dict).build()

            # Wrapper staleness: a's wrapper depends on (inner_a, spec_a);
            # b's on (inner_b, spec_b).  When policies are shared, inner_b
            # tracks inner_a, so pb_changed (a transition into/out of
            # self-play) also makes b's wrapper stale.
            if pa_changed or sa_changed:
                runner.set_policy_a(
                    _wrap_policy(
                        inner_a, spec_a_dict, stochastic, policy_a_bp_dict,
                    ),
                )
            wrap_b_stale = (
                (pa_changed or pb_changed or sb_changed)
                if same_policy
                else (pb_changed or sb_changed)
            )
            if wrap_b_stale:
                runner.set_policy_b(
                    _wrap_policy(
                        inner_b, spec_b_dict, stochastic, policy_b_bp_dict,
                    ),
                )
            current_pa_key = pa_key
            current_pb_key = pb_key
            current_sa_key = sa_key
            current_sb_key = sb_key

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
    rollout_inference:
        ``"cpu"`` (default) keeps the existing local path — workers build
        the exported policy and run ``SamplingPolicy`` in-process.
        ``"gpu"`` spawns a centralized UDS inference server on first use;
        stochastic file-export policies are then wrapped in
        :class:`RemoteSamplingPolicy` shells that block on the socket
        while the server runs the batched forward on GPU.  Environment
        stepping and all CPU work are unchanged.
    """

    def __init__(
        self,
        num_workers: int = 1,
        mp_context: str = "spawn",
        rollout_inference: str = "cpu",
    ) -> None:
        if rollout_inference not in ("cpu", "gpu"):
            raise ValueError(
                f"rollout_inference must be 'cpu' or 'gpu', "
                f"got {rollout_inference!r}"
            )
        self._num_workers = max(1, int(num_workers))
        self._mp_context = mp_context
        self._rollout_inference = rollout_inference
        self._executor: Optional[ProcessPoolExecutor] = None
        self._inference_server: Optional[InferenceServerHandle] = None

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

        # GPU inference mode: lazily spawn the shared UDS server and
        # publish its address inside every spec dict.  The worker-side
        # wrapper consumes ``_remote_addr`` (see ``_wrap_policy``); the
        # CPU path is untouched.
        remote_addr: Optional[str] = None
        if self._rollout_inference == "gpu" and any(
            job.stochastic for job in jobs
        ):
            if self._inference_server is None:
                import os

                device = os.environ.get("CB_INFER_DEVICE", "cuda")
                capacity = int(
                    os.environ.get("CB_INFER_CAPACITY", "128")
                )
                self._inference_server = InferenceServerHandle(
                    device=device, capacity=capacity,
                )
                self._inference_server.wait_ready()
            remote_addr = self._inference_server.address

        # Serialize blueprints + specs to plain dicts for pickling into
        # workers.  explore_factor callables ride along inside the spec
        # dict and must be top-level functions to be picklable.
        tasks = []
        for job in jobs:
            spec_a, spec_b = job.sampling_a, job.sampling_b
            spec_a_dict = spec_a.to_dict()
            spec_b_dict = spec_b.to_dict()
            if remote_addr is not None:
                spec_a_dict["_remote_addr"] = remote_addr
                spec_b_dict["_remote_addr"] = remote_addr
            tasks.append((
                job.policy_a_bp.to_dict(),
                job.policy_b_bp.to_dict(),
                job.env_bp.to_dict(),
                int(job.seed),
                dict(job.episode_options) if job.episode_options else None,
                spec_a_dict,
                spec_b_dict,
                job.stochastic,
            ))

        if self._num_workers <= 1:
            episodes = _run_job_batch(tasks)
        else:
            assert self._executor is not None
            # Group tasks by blueprint+spec identity so that each group
            # can reuse a single EnvRuntime + Policy across its episodes.
            groups: Dict[str, List[Tuple[int, Tuple]]] = {}
            for i, task in enumerate(tasks):
                key = json.dumps(
                    {"pa": task[0], "pb": task[1], "env": task[2]},
                    sort_keys=True, ensure_ascii=False,
                ) + "|" + _spec_key(task[5]) + "|" + _spec_key(task[6])
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
        """Shut down the worker pool and inference server (idempotent)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._inference_server is not None:
            self._inference_server.close()
            self._inference_server = None

    def __enter__(self) -> "ParallelRollouter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


__all__ = ["ParallelRollouter"]
