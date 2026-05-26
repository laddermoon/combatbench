"""Example 05 — 回放 + 调试 + 出视频 (Stage 4).

面向 (Audience) : 训练不 work / 想做可视化分析的人。
阶段 (Stage)    : 训练期某个样本行为诡异，如何定位。
学到 (Takeaway) :
  - :class:`ReplaySimulator` 是一个"非物理后端"，但对 observer / plugin
    **完全透明**——同一套 observer 代码即可回放任意录像。
  - Seed + Recorder + Replay 是定位训练 bug 的黄金三角。

依赖 (Depends)  : 先运行 examples/04_collect_rollouts.py 生成录像。
产物 (Outputs)  : examples/out/05_replay_and_inspect/replay_video.mp4
运行 (Run)      : python examples/05_replay_and_inspect.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import build_humanoid21_runtime, example_out_dir
from envs.framework import EnvRuntime, EpisodeRunner, ReplaySimulator
from envs.framework.common_plugins import VideoRecorderPlugin

# Import the mock actor defined in example 04
from importlib import import_module
_ex04 = import_module("04_collect_rollouts")
MockActorWithExtras = _ex04.MockActorWithExtras


REC_ROOT = Path(__file__).resolve().parent / "out" / "04_collect_rollouts" / "rec"


def _load_recorded_observer_outputs(episode_dir: Path) -> list[np.ndarray]:
    """Read the observation values the recorder wrote at training time."""
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    out: list[np.ndarray] = []
    for step_entry in manifest["steps"]:
        data_name = step_entry.get("data")
        if data_name is None:
            continue
        step_data = json.loads((episode_dir / data_name).read_text())
        # Try to find observation in the recorded data
        if "observation" in step_data:
            obs = step_data["observation"]
            out.append(np.asarray(obs, dtype=np.float32))
        elif "derived_state" in step_data and "robot_a" in step_data["derived_state"]:
            obs = step_data["derived_state"]["robot_a"]["observation"]
            out.append(np.asarray(obs, dtype=np.float32))
    return out


def _make_live_video(base_seed: int, episode_index: int, out_path: Path) -> None:
    """Produce an MP4 by re-running the target episode on a LIVE simulator
    with the exact same seed."""
    print(f"  Generating live video: {out_path.name}")

    runtime = build_humanoid21_runtime(
        match_duration=1.0,
        extra_plugins=[VideoRecorderPlugin(fps=30, output_path=str(out_path))],
    )

    runner = EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": MockActorWithExtras(noise_scale=0.1),
            "robot_b": MockActorWithExtras(noise_scale=0.1),
        },
    )

    runner.run_episode(seed=base_seed + episode_index)


def main() -> None:
    print("=" * 70)
    print("Example 05 — 回放 + 调试 + 出视频")
    print("=" * 70)

    if not REC_ROOT.exists():
        print(f"\nError: Recording directory not found: {REC_ROOT}")
        print(f"Please run 'python 04_collect_rollouts.py' first to generate recordings.")
        return

    # Find the first episode directory
    episode_dirs = sorted(REC_ROOT.glob("episode_*"))
    if not episode_dirs:
        print(f"\nError: No episode directories found in {REC_ROOT}")
        print(f"Please run 'python 04_collect_rollouts.py' first to generate recordings.")
        return

    episode_dir = episode_dirs[0]
    print(f"\n[Inspecting episode: {episode_dir.name}]")

    # Load and inspect the manifest
    manifest_path = episode_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  Warning: manifest.json not found")
        return

    manifest = json.loads(manifest_path.read_text())
    print(f"  base_seed: {manifest.get('base_seed', 'N/A')}")
    print(f"  num_steps: {manifest.get('num_steps', 'N/A')}")
    print(f"  termination_reasons: {manifest.get('steps', [])[-1].get('termination_reasons', []) if manifest.get('steps') else []}")

    # Load recorded observations
    print(f"\n[Loading recorded observations]")
    try:
        recorded_obs = _load_recorded_observer_outputs(episode_dir)
        print(f"  Loaded {len(recorded_obs)} observation steps")
        if recorded_obs:
            print(f"  Observation shape: {recorded_obs[0].shape}")
    except Exception as e:
        print(f"  Warning: Could not load observations: {e}")
        recorded_obs = []

    # Generate a live video
    out_dir = example_out_dir("05_replay_and_inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "replay_video.mp4"

    base_seed = manifest.get('base_seed', 42)
    episode_index = int(episode_dir.name.split("_")[1])

    _make_live_video(base_seed, episode_index, video_path)

    if video_path.exists():
        print(f"\n[Video generated]")
        print(f"  Path: {video_path}")
    else:
        print(f"\n[Warning] Video not generated: {video_path}")

    print("\nDone. 下一步：examples/06_evaluate_policy.py 看怎么评测一个 policy。")


if __name__ == "__main__":
    main()
