"""Example 08 — 新 rollout 模块: EpisodeRecorder + ParallelRollouter (Stage 3+).

面向 (Audience) : 要用新的 strongly-typed Episode / EpisodeCollection 灌数据的
                  训练工程师 / eval 开发者。
阶段 (Stage)    : 在 framework EpisodeRunner 之上，使用 baseline 提供的
                  EpisodeRecorder 和 ParallelRollouter 做可复现、可落盘的
                  rollout 采集。
学到 (Takeaway) :
  - EpisodeRecorder 如何挂在 EnvRuntime 上， episode 结束后产出
    :class:`Episode` (numpy-only, 可序列化)。
  - ParallelRollouter 的 factory 契约：为什么传函数而不是实例。
  - Episode / EpisodeCollection 的 save/load 目录结构。

产物 (Outputs)  : examples/out/08_rollout_collection/
                    ├── sequential/
                    │   ├── episode_00000.npz
                    │   ├── episode_00000.json
                    │   └── ...
                    └── parallel_collection/
                        ├── collection.json
                        ├── blueprint.yaml
                        └── episodes/
                            └── ...

运行 (Run)      : python examples/08_rollout_collection.py
                  python examples/08_rollout_collection.py --blueprint envs/humanoid21/rule_blueprint.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Make ``combatbench.<pkg>`` imports work when running the file directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.framework.blueprint import EnvBlueprint
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.policy import Policy
from policy.random.policy import RandomCombatPolicy

from baseline.common.rollout import (
    Episode,
    EpisodeCollection,
    EpisodeRecorder,
    ParallelRollouter,
    blueprint_hash,
)


def _example_out_dir(name: str) -> Path:
    """Return (and create) ``examples/out/<name>/`` for writing artifacts."""
    d = Path(__file__).resolve().parent / "out" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Policy factories — MUST be top-level importable so multiprocessing spawn
# can pickle them into worker processes.
# ---------------------------------------------------------------------------
def _robot_a_factory() -> Policy:
    return RandomCombatPolicy(scale=0.5)


def _robot_b_factory() -> Policy:
    return RandomCombatPolicy(scale=0.3)


# ---------------------------------------------------------------------------
# Part 1 — Sequential episode with EpisodeRecorder, save single Episode
# ---------------------------------------------------------------------------
def _demo_sequential(blueprint: EnvBlueprint, out_dir: Path, n_episodes: int = 3) -> None:
    """Run a few episodes sequentially, record each as an :class:`Episode`,
    and save to ``out_dir/sequential/``.
    """
    seq_dir = out_dir / "sequential"
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)

    # EpisodeRecorder needs the blueprint hash for provenance.
    bp_hash = blueprint_hash(blueprint)

    # Build runtime with the recorder attached.
    recorder = EpisodeRecorder(
        blueprint_hash=bp_hash,
        observer_names_to_keep=None,  # keep all observer outputs
    )
    runtime = blueprint.build(recorders=[recorder])

    runner = EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": _robot_a_factory(),
            "robot_b": _robot_b_factory(),
        },
    )

    print(f"\n[1] Sequential record — {n_episodes} episodes ...")
    t0 = time.perf_counter()
    for ep_idx in range(n_episodes):
        seed = 1000 + ep_idx
        runner.run_episode(seed=seed)
        episode = recorder.get_last_episode()

        stem = seq_dir / f"episode_{ep_idx:05d}"
        episode.save(stem)
        print(
            f"    episode_{ep_idx:05d}: seed={seed}  frames={episode.num_frames}  "
            f"terminated={episode.is_terminated}  saved to {stem.name}.{{npz,json}}"
        )
    dt = time.perf_counter() - t0
    print(f"    wall-clock = {dt:.2f}s  ({dt / n_episodes:.2f}s/episode)")

    # --- quick sanity check: load back the first episode ---
    loaded = Episode.load(seq_dir / "episode_00000")
    print(f"\n[1a] Load-back check: episode_00000 num_frames={loaded.num_frames}")
    for agent in sorted(loaded.observations.keys()):
        obs = loaded.observations[agent]
        act = loaded.actions[agent]
        print(f"      {agent}: obs.shape={obs.shape}  action.shape={act.shape}")

    runtime.close()


# ---------------------------------------------------------------------------
# Part 2 — Parallel collection with ParallelRollouter, save EpisodeCollection
# ---------------------------------------------------------------------------
def _demo_parallel(
    blueprint: EnvBlueprint,
    out_dir: Path,
    n_episodes: int = 8,
    num_workers: int = 2,
) -> None:
    """Collect episodes in parallel and persist an :class:`EpisodeCollection`."""
    coll_dir = out_dir / "parallel_collection"
    if coll_dir.exists():
        shutil.rmtree(coll_dir)

    print(f"\n[2] Parallel collect — N={n_episodes}, workers={num_workers} ...")
    t0 = time.perf_counter()

    # For num_workers > 1, factories must be picklable (top-level functions).
    # num_workers <= 1 falls back to in-process execution.
    with ParallelRollouter(
        blueprint=blueprint,
        policy_factories={
            "robot_a": _robot_a_factory,
            "robot_b": _robot_b_factory,
        },
        num_workers=num_workers,
        observer_names_to_keep=None,
        deterministic=False,
    ) as rollouter:
        seeds = [2000 + i for i in range(n_episodes)]
        collection = rollouter.collect(seeds=seeds)

    dt = time.perf_counter() - t0
    print(f"    wall-clock = {dt:.2f}s  ({dt / n_episodes:.2f}s/episode)")
    print(f"    episodes   = {len(collection)}")
    print(f"    total_frames = {collection.total_frames}")

    # --- save collection ---
    collection.save(coll_dir)
    print(f"    saved to   = {coll_dir}")

    # --- load back and verify ---
    loaded = EpisodeCollection.load(coll_dir)
    print(f"\n[2a] Load-back check: {len(loaded)} episodes, total_frames={loaded.total_frames}")
    assert len(loaded) == len(collection)
    assert loaded.blueprint_hash == collection.blueprint_hash
    print("    ✓ EpisodeCollection round-trip OK")

    # --- split by termination ---
    terminated, truncated = loaded.split_by_termination()
    print(f"    terminated={len(terminated)}  truncated={len(truncated)}")

    # --- stack a field across all episodes ---
    if "robot_a" in loaded[0].observations:
        stacked = loaded.stack_field(lambda e: e.observations["robot_a"])
        print(f"    stacked obs['robot_a'] shape = {stacked.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout collection example")
    parser.add_argument(
        "--blueprint",
        type=str,
        default="envs/humanoid21/rule_blueprint.yaml",
        help="Path to the EnvBlueprint YAML (default: envs/humanoid21/rule_blueprint.yaml)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=8,
        help="Total episodes to collect in the parallel demo (default: 8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers for ParallelRollouter (default: 2)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    print("=" * 70)
    print("Example 08 — Rollout Collection: EpisodeRecorder + ParallelRollouter")
    print("=" * 70)

    blueprint_path = Path(args.blueprint)
    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint not found: {blueprint_path}")

    print(f"\nLoading blueprint: {blueprint_path}")
    blueprint = EnvBlueprint.load(str(blueprint_path))
    print(f"  version={blueprint.version}  max_steps={blueprint.max_steps}")

    out_dir = _example_out_dir("08_rollout_collection")
    print(f"  output dir = {out_dir}")

    # --- Part 1: sequential ---
    _demo_sequential(blueprint, out_dir, n_episodes=3)

    # --- Part 2: parallel ---
    _demo_parallel(
        blueprint,
        out_dir,
        n_episodes=args.num_episodes,
        num_workers=args.workers,
    )

    print("\nDone. 产物:")
    print(f"  {out_dir / 'sequential'}        — 单 episode .npz + .json")
    print(f"  {out_dir / 'parallel_collection'} — EpisodeCollection 目录")


if __name__ == "__main__":
    main()
