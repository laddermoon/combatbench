"""Example 04 — 可复现地采样轨迹 (Stage 3).

面向 (Audience) : 要灌数据的训练工程师 / BC / offline RL 开发者。
阶段 (Stage)    : 训练算法之外，框架怎么让我可复现地落盘数据。
学到 (Takeaway) :
  - :class:`EpisodeRunner` 的循环使用。
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
from envs.framework import EpisodeRunner
from envs.framework.policy import Policy
from envs.framework.recorder import BaseFrameRecorder


# ---------------------------------------------------------------------------
# A policy that returns extras — mimics what an on-policy RL actor looks like:
# ``act_with_extras`` returns (action, {"log_prob": ..., "value": ...}).
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
# Factory — builds an EpisodeRunner with recorder
# ---------------------------------------------------------------------------
def _build_runner(worker_id: int, recorder: BaseFrameRecorder) -> EpisodeRunner:
    runtime = build_humanoid21_runtime(match_duration=1.0)
    runtime.attach_recorder(recorder)
    return EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": MockActorWithExtras(noise_scale=0.1),
            "robot_b": MockActorWithExtras(noise_scale=0.1),
        },
    )


# ---------------------------------------------------------------------------
# Part 1 — sequential collect + Recorder dump to disk
# ---------------------------------------------------------------------------
def _collect_sequential_with_recorder(n_episodes: int, base_seed: int, out_dir: Path):
    """Collect episodes sequentially and save to disk."""
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
        save_sensor_data=False,    # skip to save disk space
        save_action_extras=True,   # capture log_prob / value from policy
    )

    runner = _build_runner(worker_id=0, recorder=recorder)

    start = time.time()
    for ep in range(n_episodes):
        seed = base_seed + ep
        runner.run_episode(seed=seed)
        print(f"  Episode {ep} (seed={seed}) complete")
    elapsed = time.time() - start

    # Load and return the index
    index_path = out_dir / "index.json"
    if not index_path.exists():
        print(f"  Warning: {index_path} not found, returning empty index")
        return {"episodes": []}, elapsed

    index = json.loads(index_path.read_text(encoding="utf-8"))
    return index, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Example 04 — 可复现地采样轨迹 (sequential)")
    print("=" * 70)

    out_dir = example_out_dir("04_collect_rollouts")
    rec_dir = out_dir / "rec"
    rec_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = 4
    base_seed = 42

    print(f"\n[Collecting {n_episodes} episodes, base_seed={base_seed}]\n")
    index, elapsed = _collect_sequential_with_recorder(n_episodes, base_seed, rec_dir)

    print(f"\n[Collection complete]")
    print(f"  Elapsed time: {elapsed:.1f}s ({elapsed / n_episodes:.2f}s per episode)")
    print(f"  Output dir: {rec_dir}")
    print(f"  Episodes recorded: {len(index.get('episodes', []))}")

    print("\nDone. 下一步：examples/05_replay_and_inspect.py 看怎么回放与检查数据。")


if __name__ == "__main__":
    main()
