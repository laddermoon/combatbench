"""Example 09 — 使用 RoundRunner + EpisodeRecorder 做单 episode 录制.

面向 (Audience) : 想在最简 API (RoundRunner) 上挂接 EpisodeRecorder、
                  落盘单条 episode 并做回读验证的开发者。
阶段 (Stage)    : 只需一个 RoundRunner、两个 policy 实例、一个 recorder，
                  即可跑出可序列化的 :class:`Episode`。
学到 (Takeaway) :
  - RoundRunner 自动管理 runtime 生命周期，policy 直接传实例即可。
  - EpisodeRecorder 作为 PostActionRecorder 在 runtime 的每个 step 后
    自动缓存帧数据；episode 结束后调用 ``recorder.get_last_episode()``。
  - Episode.save / Episode.load 的磁盘格式 (npz + json)。

产物 (Outputs)  : examples/out/09_episode_recorder_round_runner/
                    ├── episode_00000.npz
                    ├── episode_00000.json
                    └── ...

运行 (Run)      : python examples/09_episode_recorder_round_runner.py
                  python examples/09_episode_recorder_round_runner.py --blueprint envs/humanoid21/rule_blueprint.yaml --episodes 5
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
from envs.framework.round_runner import RoundRunner
from policy.random.policy import RandomCombatPolicy

from baseline.common.rollout import Episode, EpisodeRecorder, blueprint_hash


def _example_out_dir(name: str) -> Path:
    """Return (and create) ``examples/out/<name>/`` for writing artifacts."""
    d = Path(__file__).resolve().parent / "out" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
def _run_and_record(
    blueprint: EnvBlueprint,
    out_dir: Path,
    n_episodes: int = 3,
) -> None:
    """Run *n_episodes* via RoundRunner, record each as an :class:`Episode`."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bp_hash = blueprint_hash(blueprint)
    recorder = EpisodeRecorder(
        blueprint_hash=bp_hash,
        observer_names_to_keep=None,
    )

    policy_a = RandomCombatPolicy(scale=0.5)
    policy_b = RandomCombatPolicy(scale=0.3)

    print(f"\n[1] RoundRunner + EpisodeRecorder — {n_episodes} episodes ...")
    t0 = time.perf_counter()

    with RoundRunner(
        blueprint=blueprint,
        policy_a=policy_a,
        policy_b=policy_b,
        recorders=[recorder],
    ) as runner:
        for ep_idx in range(n_episodes):
            seed = 3000 + ep_idx
            result = runner.run(seed=seed)

            episode = recorder.get_last_episode()
            stem = out_dir / f"episode_{ep_idx:05d}"
            episode.save(stem)

            print(
                f"    episode_{ep_idx:05d}: seed={seed}  steps={result['steps']}  "
                f"terminated={episode.is_terminated}  saved to {stem.name}.{{npz,json}}"
            )

    dt = time.perf_counter() - t0
    print(f"    wall-clock = {dt:.2f}s  ({dt / n_episodes:.2f}s/episode)")

    # --- quick sanity check: load back the first episode ---
    loaded = Episode.load(out_dir / "episode_00000")
    print(f"\n[1a] Load-back check: episode_00000 num_frames={loaded.num_frames}")
    for agent in sorted(loaded.observations.keys()):
        obs = loaded.observations[agent]
        act = loaded.actions[agent]
        print(f"      {agent}: obs.shape={obs.shape}  action.shape={act.shape}")

    # --- verify blueprint hash round-trip ---
    assert loaded.blueprint_hash == bp_hash
    print(f"    ✓ blueprint_hash round-trip OK ({bp_hash[:16]}...)")


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EpisodeRecorder via RoundRunner example",
    )
    parser.add_argument(
        "--blueprint",
        type=str,
        default="envs/humanoid21/rule_blueprint.yaml",
        help="Path to the EnvBlueprint YAML (default: envs/humanoid21/rule_blueprint.yaml)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 70)
    print("Example 09 — EpisodeRecorder via RoundRunner")
    print("=" * 70)

    blueprint_path = Path(args.blueprint)
    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint not found: {blueprint_path}")

    print(f"\nLoading blueprint: {blueprint_path}")
    blueprint = EnvBlueprint.load(str(blueprint_path))
    print(f"  version={blueprint.version}  max_steps={blueprint.max_steps}")

    out_dir = _example_out_dir("09_episode_recorder_round_runner")
    print(f"  output dir = {out_dir}")

    _run_and_record(blueprint, out_dir, n_episodes=args.episodes)

    print("\nDone. 产物:")
    print(f"  {out_dir}  — 单 episode .npz + .json")


if __name__ == "__main__":
    main()
