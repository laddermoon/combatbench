"""ParallelRunner tests.

Most assertions use ``num_workers=1`` which runs everything in-process (no
pickling, no spawn overhead, easy to debug). A dedicated multi-worker smoke
test exercises the real ``spawn`` path with 2 workers to confirm the end-to-
end factory / pool / seed plumbing.

All symbols used inside worker processes (the factory, policy classes, sim
wrappers) are defined at MODULE TOP LEVEL so ``pickle`` under ``spawn`` can
resolve them by import.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict

import numpy as np
import pytest

from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import (
    EpisodeResult,
    EpisodeRunner,
    RolloutConfig,
)
from envs.framework.parallel_runner import ParallelRunner, _derive_seeds
from envs.framework.policy import Policy
from envs.framework.observer_plugin import BaseObserverPlugin

# Reuse the module-level observer + policy + deterministic sim from the
# episode-runner tests. Those classes are top-level → picklable under spawn.
from envs.framework.tests.test_episode_runner import (
    _DeterministicSim,
    _QposObserver,
    _ScalarRewardObserver,
    _SeededPolicy,
)


# ---------------------------------------------------------------------------
# Top-level factories (must be picklable for spawn-based multiprocessing).
# ---------------------------------------------------------------------------
def _build_runtime_for_factory(max_steps: int = 3) -> EnvRuntime:
    return EnvRuntime(
        simulator=_DeterministicSim(),
        observer_plugins={
            "robot_a_obs": _QposObserver(),
            "robot_a_reward": _ScalarRewardObserver(),
            "robot_b_obs": _QposObserver(),
            "robot_b_reward": _ScalarRewardObserver(),
        },
        max_steps=max_steps,
        phy_steps_per_action=1,
    )


def make_test_runner(worker_id: int) -> EpisodeRunner:
    """Standard factory: 3-step episodes, seeded policies."""
    return EpisodeRunner(
        runtime=_build_runtime_for_factory(max_steps=3),
        policies={
            "robot_a": _SeededPolicy(f"a-{worker_id}"),
            "robot_b": _SeededPolicy(f"b-{worker_id}"),
        },
    )


def short_episode_factory(worker_id: int) -> EpisodeRunner:
    """Shorter episodes for the multi-worker smoke test — keep CI fast."""
    return EpisodeRunner(
        runtime=_build_runtime_for_factory(max_steps=2),
        policies={
            "robot_a": _SeededPolicy(f"a-{worker_id}"),
            "robot_b": _SeededPolicy(f"b-{worker_id}"),
        },
    )


# Factory whose runner always raises on the first step — used to test error
# handling paths. Defined at top level so ``spawn`` can import it.
class _ExplodingPolicy(Policy):
    def act(self, observation: Any) -> np.ndarray:
        raise RuntimeError("boom")


def exploding_factory(worker_id: int) -> EpisodeRunner:
    return EpisodeRunner(
        runtime=_build_runtime_for_factory(max_steps=3),
        policies={
            "robot_a": _ExplodingPolicy(),
            "robot_b": _SeededPolicy("b"),
        },
    )


# ---------------------------------------------------------------------------
# Top-level chunk_fns for map_chunks tests (must be picklable for spawn).
# ---------------------------------------------------------------------------
def chunk_fn_run_seeds(runner: EpisodeRunner, task: dict) -> list:
    """Run each (seed, options) item in the task; return seed list.

    Returning only seeds (not full EpisodeResults) keeps the pickle
    payload small and decouples the test from the deeply-nested
    EpisodeResult structure.
    """
    out = []
    for seed, options in task["items"]:
        result = runner.run_episode(seed=int(seed), options=options)
        out.append(result.seed)
    return out


def chunk_fn_count(runner: EpisodeRunner, task: dict) -> int:
    """Just count steps across episodes; ignores per-task options."""
    total = 0
    for seed, options in task["items"]:
        result = runner.run_episode(seed=int(seed), options=options)
        total += result.num_steps
    return total


def chunk_fn_explode(runner: EpisodeRunner, task: dict) -> int:
    raise RuntimeError("chunk_fn boom")


def chunk_fn_explode_first_only(runner: EpisodeRunner, task: dict) -> list:
    """Best-effort smoke fixture: tasks tagged with ``"explode": True``
    raise; all others run normally and return their seed list."""
    if task.get("explode"):
        raise RuntimeError("chunk_fn boom")
    return chunk_fn_run_seeds(runner, task)


# ---------------------------------------------------------------------------
# _derive_seeds parity with EpisodeRunner.run_n_episodes
# ---------------------------------------------------------------------------
def test_derive_seeds_matches_episode_runner_batch(mock_simulator):
    """ParallelRunner uses the exact same SeedSequence derivation as
    EpisodeRunner.run_n_episodes — required for results to match across
    the sequential/parallel boundary."""
    parallel_seeds = [int(s) for s in _derive_seeds(base_seed=123, n=5)]

    runtime = _build_runtime_for_factory(max_steps=1)
    runner = EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": _SeededPolicy("a"),
            "robot_b": _SeededPolicy("b"),
        },
    )
    episode_results = runner.run_n_episodes(5, base_seed=123)
    sequential_seeds = [r.seed for r in episode_results]

    assert parallel_seeds == sequential_seeds


# ---------------------------------------------------------------------------
# Single-process fast path (num_workers=1)
# ---------------------------------------------------------------------------
class TestSingleProcessFastPath:
    def test_run_produces_n_results_in_order(self):
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            results = pr.run(n=4, base_seed=42)
        assert len(results) == 4
        assert all(isinstance(r, EpisodeResult) for r in results)
        # Seeds match the sequential SeedSequence derivation exactly.
        expected = [int(s) for s in _derive_seeds(42, 4)]
        assert [r.seed for r in results] == expected

    def test_parallel_equals_sequential_bit_equal(self):
        """Single-process ParallelRunner output must match
        EpisodeRunner.run_n_episodes bit-for-bit (same seeds, same runner)."""
        runtime = _build_runtime_for_factory(max_steps=3)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a-0"),
                "robot_b": _SeededPolicy("b-0"),
            },
        )
        sequential = runner.run_n_episodes(3, base_seed=77)

        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            parallel = pr.run(n=3, base_seed=77)

        assert [r.seed for r in sequential] == [r.seed for r in parallel]
        for sr, pr_res in zip(sequential, parallel):
            for agent in ("robot_a", "robot_b"):
                # SeededPolicy is rebuilt fresh by the factory, so with the
                # same per-episode seed the action sequence matches.
                for a_seq, a_par in zip(
                    sr.trajectories[agent].actions,
                    pr_res.trajectories[agent].actions,
                ):
                    np.testing.assert_array_equal(a_seq, a_par)

    def test_run_n_zero(self):
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            assert pr.run(n=0, base_seed=1) == []

    def test_negative_n_rejected(self):
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            with pytest.raises(ValueError, match="non-negative"):
                pr.run(n=-1)

    def test_strict_true_reraises(self):
        with ParallelRunner(exploding_factory, num_workers=1, strict=True) as pr:
            with pytest.raises(RuntimeError, match="boom"):
                pr.run(n=1, base_seed=0)

    def test_strict_false_returns_none_for_failures(self):
        with ParallelRunner(exploding_factory, num_workers=1, strict=False) as pr:
            results = pr.run(n=2, base_seed=0)
        assert results == [None, None]

    def test_context_manager_closes_runner(self):
        pr = ParallelRunner(make_test_runner, num_workers=1)
        pr.run(n=1, base_seed=0)
        assert pr._inproc_runner is not None
        pr.close()
        assert pr._inproc_runner is None
        # Closed is idempotent and rejects further use.
        pr.close()
        with pytest.raises(RuntimeError, match="closed"):
            pr.run(n=1)

    def test_run_iter_ordered_yields_in_seed_order(self):
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            streamed = list(pr.run_iter(n=3, base_seed=5, ordered=True))
        expected_seeds = [int(s) for s in _derive_seeds(5, 3)]
        assert [r.seed for r in streamed] == expected_seeds

    def test_non_callable_factory_rejected(self):
        with pytest.raises(TypeError, match="callable"):
            ParallelRunner(object(), num_workers=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Multi-process path (real spawn + worker pool)
# ---------------------------------------------------------------------------
# These tests exercise the actual pickling / pool / initializer paths. They
# are slow (~1-2s of pool startup under spawn), so we keep coverage to the
# minimum needed to prove the machinery works end to end.

def test_multiprocess_run_returns_ordered_results():
    with ParallelRunner(
        short_episode_factory, num_workers=2, mp_start_method="spawn"
    ) as pr:
        results = pr.run(n=4, base_seed=101)
    assert len(results) == 4
    assert all(isinstance(r, EpisodeResult) for r in results)
    expected = [int(s) for s in _derive_seeds(101, 4)]
    assert [r.seed for r in results] == expected


def test_multiprocess_matches_single_process_results():
    """Same factory + same base_seed → same per-episode trajectories whether
    executed in one process or spread across two workers."""
    with ParallelRunner(short_episode_factory, num_workers=1) as pr:
        seq = pr.run(n=4, base_seed=314)
    with ParallelRunner(
        short_episode_factory, num_workers=2, mp_start_method="spawn"
    ) as pr:
        par = pr.run(n=4, base_seed=314)

    assert [r.seed for r in seq] == [r.seed for r in par]
    for sr, pr_res in zip(seq, par):
        for agent in ("robot_a", "robot_b"):
            for a_seq, a_par in zip(
                sr.trajectories[agent].actions,
                pr_res.trajectories[agent].actions,
            ):
                np.testing.assert_array_equal(a_seq, a_par)


def test_multiprocess_strict_false_tolerates_worker_failures():
    with ParallelRunner(
        exploding_factory,
        num_workers=2,
        mp_start_method="spawn",
        strict=False,
    ) as pr:
        results = pr.run(n=3, base_seed=0)
    assert len(results) == 3
    assert all(r is None for r in results)


# ---------------------------------------------------------------------------
# map_chunks: generic chunk dispatch (RolloutCollector etc. ride on this)
# ---------------------------------------------------------------------------
class TestMapChunksSingleProcess:
    def test_in_process_yields_chunk_fn_results_in_order(self):
        tasks = [
            {"items": [(1, None), (2, None)]},
            {"items": [(3, None)]},
            {"items": [(4, None), (5, None), (6, None)]},
        ]
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            out = list(pr.map_chunks(tasks, chunk_fn_run_seeds))
        assert out == [[1, 2], [3], [4, 5, 6]]

    def test_empty_tasks_yields_nothing(self):
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            assert list(pr.map_chunks([], chunk_fn_run_seeds)) == []

    def test_non_callable_chunk_fn_rejected(self):
        with ParallelRunner(make_test_runner, num_workers=1) as pr:
            with pytest.raises(TypeError, match="callable"):
                list(pr.map_chunks([{"items": []}], object()))  # type: ignore[arg-type]

    def test_strict_in_process_reraises_chunk_fn_error(self):
        with ParallelRunner(make_test_runner, num_workers=1, strict=True) as pr:
            with pytest.raises(RuntimeError, match="chunk_fn boom"):
                list(pr.map_chunks([{"items": [(0, None)]}], chunk_fn_explode))

    def test_non_strict_in_process_swallows_chunk_fn_error(self):
        with ParallelRunner(
            make_test_runner, num_workers=1, strict=False,
        ) as pr:
            # Failure on first task is logged and skipped; other tasks proceed.
            out = list(pr.map_chunks(
                [
                    {"items": [(1, None)], "explode": True},
                    {"items": [(2, None)]},
                ],
                chunk_fn_explode_first_only,
            ))
        # Only the second task's result comes back.
        assert out == [[2]]


def test_multiprocess_map_chunks_seed_order_preserved():
    """Multi-worker map_chunks: ordered=True yields task results in the
    same order as submission, even though workers complete out of order."""
    tasks = [{"items": [(seed, None)]} for seed in (10, 20, 30, 40)]
    with ParallelRunner(
        short_episode_factory, num_workers=2, mp_start_method="spawn",
    ) as pr:
        out = list(pr.map_chunks(tasks, chunk_fn_run_seeds))
    assert out == [[10], [20], [30], [40]]


def test_multiprocess_map_chunks_strict_reraises():
    """Worker exception in strict mode poisons the pool and re-raises so
    the context manager's __exit__ doesn't hang."""
    with ParallelRunner(
        make_test_runner, num_workers=2, mp_start_method="spawn", strict=True,
    ) as pr:
        with pytest.raises(RuntimeError, match="chunk_fn boom"):
            list(pr.map_chunks(
                [{"items": [(1, None)]}, {"items": [(2, None)]}],
                chunk_fn_explode,
            ))
