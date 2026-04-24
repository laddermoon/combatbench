"""Example 04 — 可复现地、并行地采样轨迹 (Stage 3).

面向 (Audience) : 要灌数据的训练工程师 / BC / offline RL 开发者。
阶段 (Stage)    : 训练算法之外，框架怎么让我并行、可复现地落盘数据。
学到 (Takeaway) :
  - :class:`ParallelRunner` 的 factory 契约：为什么传函数不是实例。
  - ``RolloutConfig(store_extras=True)`` 是框架**专门为 on-policy RL 留的**
    log_prob / value 回传通道。
  - :class:`BaseFrameRecorder` 自动把 ``base_seed`` 写进 manifest，为
    example 05 的回放保证可复现。

产物 (Outputs)  : examples/out/04_collect_rollouts/rec/episode_*/ + manifest
运行 (Run)      : python examples/04_collect_rollouts.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from _common import build_humanoid21_runtime, example_out_dir
from envs.framework import EpisodeRunner, ParallelRunner
from envs.framework.episode_runner import (
    AGENT_IDS,
    ObserverBinding,
    RolloutConfig,
    _derive_batch_seeds,
)
from envs.framework.policy import Policy
from envs.framework.recorder import BaseFrameRecorder


# ---------------------------------------------------------------------------
# A policy that returns extras — mimics what an on-policy RL actor looks like:
# ``act_with_extras`` returns (action, {"log_prob": ..., "value": ...}).
# ``EpisodeRunner`` picks this up automatically when
# ``RolloutConfig.store_extras=True``.
# ---------------------------------------------------------------------------
class MockActorWithExtras(Policy):
    """Deterministic 'actor': returns a noisy zero action plus fake
    ``log_prob`` / ``value`` heads. Replace with your torch ``Actor``
    and the pipeline stays identical."""

    def __init__(self, action_dim: int = 21, noise_scale: float = 0.1) -> None:
        self.action_dim = int(action_dim)
        self.noise_scale = float(noise_scale)
        self._rng = np.random.default_rng()

    def reset(self, seed=None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation):  # pragma: no cover - we always go through act_with_extras
        raise AssertionError("store_extras=True should call act_with_extras")

    def act_with_extras(self, observation) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = self._rng.normal(0.0, self.noise_scale, self.action_dim).astype(np.float32)
        # Fake RL heads — in real code these come from the actor network.
        log_prob = float(-0.5 * np.sum(action ** 2) / (self.noise_scale ** 2))
        value = float(np.sum(observation[:8]) * 0.01)  # dummy value head
        return action, {"log_prob": log_prob, "value": value}


# ---------------------------------------------------------------------------
# Factory — MUST be top-level importable so ParallelRunner workers can
# ``pickle.loads`` it under the ``spawn`` start method.
# ---------------------------------------------------------------------------
def _build_runner(worker_id: int) -> EpisodeRunner:
    runtime = build_humanoid21_runtime(match_duration=1.0)
    return EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": MockActorWithExtras(noise_scale=0.1),
            "robot_b": MockActorWithExtras(noise_scale=0.1),
        },
        observer_bindings={
            a: ObserverBinding(obs_name=f"{a}_obs", reward_name=None)
            for a in AGENT_IDS
        },
        rollout=RolloutConfig(store_extras=True),
    )


# ---------------------------------------------------------------------------
# Part 1 — sequential collect + Recorder dump to disk
# ---------------------------------------------------------------------------
def _collect_sequential_with_recorder(n_episodes: int, base_seed: int, out_dir: Path):
    runner = _build_runner(worker_id=0)
    # Attach the recorder so every episode lands in ``out_dir/episode_NNNNN/``.
    # base_seed is published on ``runtime.ctx.base_seed`` by EpisodeRunner and
    # automatically makes its way into ``manifest.json`` (see SEED.md §4).
    recorder = BaseFrameRecorder(
        output_dir=out_dir,
        save_image=False,          # skip images to keep demo fast
        save_core_state=True,      # needed by ReplaySimulator in example 05
        save_derived_state=True,   # humanoid21 observer reads derived_state
        save_observer_outputs=True,  # so example 05 can diff replay vs online
        save_static_data=True,
        quiet=True,
    )
    runner.runtime.attach_recorder(recorder)

    t0 = time.perf_counter()
    results = runner.run_n_episodes(n_episodes, base_seed=base_seed)
    dt = time.perf_counter() - t0
    return results, dt


def _collect_parallel(n_episodes: int, num_workers: int, base_seed: int):
    """Same N episodes, same base_seed — just spread across workers."""
    t0 = time.perf_counter()
    with ParallelRunner(_build_runner, num_workers=num_workers) as pr:
        results = pr.run(n=n_episodes, base_seed=base_seed)
    dt = time.perf_counter() - t0
    return results, dt


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Example 04 — 可复现、可并行的 rollout 采集")
    print("=" * 70)

    out_dir = example_out_dir("04_collect_rollouts")
    rec_dir = out_dir / "rec"
    if rec_dir.exists():
        # Start each run from a clean slate so ``episode_NNNNN`` numbering
        # doesn't keep drifting on repeat invocations.
        import shutil
        shutil.rmtree(rec_dir)

    N_EPISODES = 4
    BASE_SEED = 20260424

    # --- Part 1: sequential + recorder ---
    print(f"\n[1] Sequential run, N={N_EPISODES}, base_seed={BASE_SEED}, recording to {rec_dir.name}/ ...")
    seq_results, seq_dt = _collect_sequential_with_recorder(N_EPISODES, BASE_SEED, rec_dir)
    print(f"    wall-clock = {seq_dt:.2f}s  ({seq_dt / N_EPISODES:.2f}s/episode)")

    # Show the extras bucket actually got captured — this is the training-ready
    # log_prob/value channel.
    ep0 = seq_results[0]
    extras_list_a = ep0.trajectories["robot_a"].extras
    print(f"    extras captured per step: keys={sorted(extras_list_a[0].keys())}")
    print(f"    example step 0 extras_a : {extras_list_a[0]}")

    # Show the manifest got base_seed — this is how replay (example 05) works.
    manifest0 = json.loads((rec_dir / "episode_00000" / "manifest.json").read_text())
    print(f"    episode_00000 manifest.base_seed = {manifest0['base_seed']}  "
          f"(matches EpisodeResult.seed = {ep0.seed})")

    # --- Part 2: parallel speed ---
    # Try 2 workers. On a tiny 4-episode batch the MuJoCo import overhead in
    # fresh workers often dominates, so don't be surprised if speedup<1. The
    # important thing is **seed parity**: same base_seed → same per-episode
    # seeds no matter where the episode ran.
    print(f"\n[2] Parallel run, N={N_EPISODES}, num_workers=2, base_seed={BASE_SEED} ...")
    par_results, par_dt = _collect_parallel(N_EPISODES, num_workers=2, base_seed=BASE_SEED)
    print(f"    wall-clock = {par_dt:.2f}s  ({par_dt / N_EPISODES:.2f}s/episode)")

    # --- Part 3: seed parity proof ---
    expected_seeds = [int(s) for s in _derive_batch_seeds(BASE_SEED, N_EPISODES)]
    seq_seeds = [r.seed for r in seq_results]
    par_seeds = [r.seed for r in par_results]
    print("\n[3] Seed parity check:")
    print(f"    expected (from _derive_batch_seeds): {expected_seeds}")
    print(f"    sequential EpisodeRunner seeds     : {seq_seeds}")
    print(f"    ParallelRunner seeds               : {par_seeds}")
    assert seq_seeds == expected_seeds == par_seeds, "Seed derivation drifted!"
    print("    ✓ 同 base_seed 下，sequential 和 parallel 跑的是完全相同的 episode 序列。")

    print(f"\n产物就绪：{rec_dir}")
    print("  - 每个 episode 一个子目录，含 manifest.json + static.json + 每步 state 快照")
    print("  - 下一步：examples/05_replay_and_inspect.py 读这个目录做回放")


if __name__ == "__main__":
    main()
