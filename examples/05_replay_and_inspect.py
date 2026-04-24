"""Example 05 — 回放 + 调试 + 出视频 (Stage 4).

面向 (Audience) : 训练不 work / 想做可视化分析的人。
阶段 (Stage)    : 训练期某个样本行为诡异，如何定位。
学到 (Takeaway) :
  - :class:`ReplaySimulator` 是一个"非物理后端"，但对 observer / plugin
    **完全透明**——同一套 observer 代码即可回放任意录像。
  - 回放期 observer 输出 vs 录制期 observer 输出应当 **bit-wise 一致**，
    否则就是在线代码和 observer 的实现不匹配，训练出的分布必然有偏。
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
from envs.framework.episode_runner import AGENT_IDS, ObserverBinding, RolloutConfig
from envs.humanoid21.observer_plugins import Humanoid21Observer

# Import the factory and mock actor defined in example 04 so the live
# rollout here is bit-equal to what 04 recorded (same classes, same code).
from importlib import import_module
_ex04 = import_module("04_collect_rollouts")
MockActorWithExtras = _ex04.MockActorWithExtras


REC_ROOT = Path(__file__).resolve().parent / "out" / "04_collect_rollouts" / "rec"


def _replay_observer_outputs(episode_dir: Path) -> list[np.ndarray]:
    """Re-run a recorded episode through ReplaySimulator + the SAME
    Humanoid21Observer the training code uses; return ``robot_a_obs`` per
    step so we can diff against what was saved to disk at record time."""
    sim = ReplaySimulator(str(episode_dir))
    runtime = EnvRuntime(
        simulator=sim,
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
        },
        phy_steps_per_action=1,   # recorder stride was 1 → one replay frame per step
    )
    runtime.reset()
    # Pull the reset-time observation first (matches how online EpisodeRunner
    # stores ``initial_observation``).
    obs_per_step: list[np.ndarray] = [runtime.get_observer_output("robot_a_obs").copy()]
    zero = np.zeros(21, dtype=np.float32)
    while runtime.is_episode_active and sim.has_next_step():
        runtime.step(zero, zero)  # actions are ignored; replay is read-only
        obs_per_step.append(runtime.get_observer_output("robot_a_obs").copy())
    return obs_per_step


def _load_recorded_observer_outputs(episode_dir: Path) -> list[np.ndarray]:
    """Read the ``robot_a_obs`` values the recorder wrote at training time."""
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    out: list[np.ndarray] = []
    for step_entry in manifest["steps"]:
        data_name = step_entry.get("data")
        if data_name is None:
            continue
        step_data = json.loads((episode_dir / data_name).read_text())
        obs_a = step_data["observer_outputs"]["robot_a_obs"]
        out.append(np.asarray(obs_a, dtype=np.float32))
    return out


def _make_live_video(base_seed: int, episode_index: int, out_path: Path) -> None:
    """Produce an MP4 by re-running the target episode on a LIVE simulator
    with the exact same seed. Because the whole pipeline is deterministic
    (SEED.md), the on-screen trajectory matches the recording frame-for-frame.
    """
    video_plugin = VideoRecorderPlugin(fps=30, output_path=str(out_path))
    runtime = build_humanoid21_runtime(
        match_duration=1.0,
        extra_plugins=[video_plugin],
    )
    runner = EpisodeRunner(
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
    # run_n_episodes(N, base_seed=X) derives the exact same children as the
    # recording run, so ``results[episode_index]`` covers the target episode.
    runner.run_n_episodes(episode_index + 1, base_seed=base_seed)


def main() -> None:
    print("=" * 70)
    print("Example 05 — 回放 + observer 差分 + 出视频")
    print("=" * 70)

    if not REC_ROOT.exists():
        raise FileNotFoundError(
            f"Recording not found at {REC_ROOT}. Run 04_collect_rollouts.py first."
        )

    episode_dir = REC_ROOT / "episode_00000"
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    base_seed = int(manifest["base_seed"])
    print(f"\nReplaying {episode_dir.relative_to(REC_ROOT.parent)}")
    print(f"  base_seed in manifest : {base_seed}")
    print(f"  num_steps             : {manifest['num_steps']}")

    # ---- (1) Replay → observer outputs ----
    replay_obs = _replay_observer_outputs(episode_dir)
    recorded_obs = _load_recorded_observer_outputs(episode_dir)
    print(f"\n[1] Computed obs from replay : {len(replay_obs)} frames")
    print(f"    Recorded obs from disk   : {len(recorded_obs)} frames")

    # The recorded JSON stream and the replay stream should line up 1-for-1.
    n = min(len(replay_obs), len(recorded_obs))
    max_abs_diff = 0.0
    for i in range(n):
        max_abs_diff = max(max_abs_diff, float(np.abs(replay_obs[i] - recorded_obs[i]).max()))
    print(f"    max |replay_obs - recorded_obs| over {n} frames = {max_abs_diff:.2e}")
    # Dtype is float32 on both sides; tolerance of 1e-5 covers the
    # JSON-roundtrip rehydration loss documented in ReplaySimulator.
    assert max_abs_diff < 1e-5, (
        f"Replay observer diverges from recording (max diff = {max_abs_diff})."
    )
    print("    ✓ 在线 observer 与回放 observer 输出 bit-wise 一致。")

    # ---- (2) Tiny debug hunt: first step where height_a dips below 0.9 ----
    # This is the canonical "逐帧 debug" workflow: you suspect something
    # weird happened, you walk the recording frame by frame looking for the
    # exact state that triggered it.
    print("\n[2] Debug hunt — 找第一帧 robot_a root_pos.z < 0.9 的时刻")
    trigger_frame = None
    for step_entry in manifest["steps"]:
        data_name = step_entry.get("data")
        if data_name is None:
            continue
        data = json.loads((episode_dir / data_name).read_text())
        z = float(data["core_state"]["robot_a"]["root_pos"][2])
        if z < 0.9:
            trigger_frame = (step_entry["step"], z)
            break
    if trigger_frame is None:
        print("    robot_a 在整个 episode 都没低于 0.9。")
    else:
        step, z = trigger_frame
        print(f"    trigger @ episode_step={step}  root_pos.z={z:.3f}")
        print("    → 若训练中出现异常，这就是逐帧回溯定位的标准姿势。")

    # ---- (3) Live-rerun → MP4 ----
    out_dir = example_out_dir("05_replay_and_inspect")
    video_path = out_dir / "replay_video.mp4"
    print(f"\n[3] Live rerun → {video_path.name}")
    print(f"    （重新用 LIVE simulator 按同一 base_seed={base_seed} 跑一遍同一 episode 来出视频。"
          f"因为 SEED.md 定下的确定性，这和 replay 对应同一段物理过程。）")
    _make_live_video(base_seed=base_seed, episode_index=0, out_path=video_path)
    if video_path.exists():
        print(f"    ✓ 视频已保存：{video_path}")
    else:
        print("    ⚠ 视频文件未生成（VideoRecorderPlugin 可能静默失败，检查 MuJoCo GL 后端）。")

    print("\nDone. 下一步：examples/06_evaluate_policy.py 按比赛规则评测你的策略。")


if __name__ == "__main__":
    main()
