"""Replay hybrid standup+balance policy with video.

Exports the HybridActor (standup_v2_r14 + follow_v2 fallback) to a
deployable policy, then runs episodes in the hybrid environment with
video recording.

Usage:
  PYTHONPATH=/data1/mono/things/combatbench python3 \
    baseline/humanoid21/replay_hybrid.py \
    --episodes 5 --video-dir /tmp/hybrid_videos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.framework import BasePlugin
from envs.framework.blueprint import EnvBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.policy import Policy, PolicyBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

from baseline.humanoid21.curriculum.hybrid_actor import HybridActor

_OBS_R00 = 42
_OBS_R10 = 43
_OBS_R01 = 45
_OBS_R11 = 46


class TracePlugin(BasePlugin):
    """Record per-step height, uprightness, and active mode."""

    BLUEPRINT_EXCLUDE = True
    priority = 0

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._trace: List[Dict[str, Any]] = []
        self._step = 0

    @property
    def name(self) -> str:
        return "trace"

    def on_pre_episode(self, ctx) -> None:
        self._trace.clear()
        self._step = 0

    def on_post_action_step(self, ctx) -> None:
        core_state = ctx.accessor.get_core_state()
        cs = core_state.get(self.agent_id, {})
        derived = ctx.accessor.get_derived_state([self.agent_id])
        ds = derived.get(self.agent_id, {})
        height = float(cs.get("root_pos", [0, 0, 0])[2])
        upright = float(
            np.asarray(ds.get("uprightness", [0.0]), dtype=np.float32).reshape(-1)[0]
        )
        self._trace.append({
            "step": self._step,
            "height": height,
            "uprightness": upright,
        })
        self._step += 1

    def on_post_episode(self, ctx) -> None:
        pass

    def get_trace(self) -> List[Dict[str, Any]]:
        return self._trace


class ZeroPolicy(Policy):
    """Minimal policy that outputs zero actions (for robot_b)."""

    def act(self, observation, want_extra=False):
        action = np.zeros(21, dtype=np.float32)
        if want_extra:
            return action, {"log_prob": 0.0}
        return action, None

    def reset(self, seed=None):
        pass

    def close(self):
        pass


def run_episode(
    blueprint: EnvBlueprint,
    policy_a: Policy,
    policy_b: Policy,
    video_plugin: VideoRecorderPlugin,
    trace_plugin: TracePlugin,
    seed: int = 42,
) -> Dict[str, Any]:
    debug_plugins = [video_plugin, trace_plugin]
    runtime = blueprint.build(debug_plugins=debug_plugins)
    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=policy_a,
        policy_b=policy_b,
    )

    if hasattr(policy_a, "reset"):
        policy_a.reset(seed=seed)

    t0 = time.perf_counter()
    runner.run_episode(seed=seed, want_extras=False)
    elapsed = time.perf_counter() - t0

    ctx = runtime.ctx
    trace = trace_plugin.get_trace()
    result = {
        "steps": int(ctx.episode_step),
        "termination_reasons": {aid: list(ctx.agent_termination_proposals[aid]) for aid in ("robot_a", "robot_b")},
        "seed": seed,
        "elapsed": elapsed,
        "trace": trace,
    }

    if trace:
        result["final_height"] = trace[-1]["height"]
        result["final_uprightness"] = trace[-1]["uprightness"]
        result["max_height"] = max(t["height"] for t in trace)
        result["max_uprightness"] = max(t["uprightness"] for t in trace)
        # Count mode transitions
        upright_arr = np.array([t["uprightness"] for t in trace])
        balance_steps = int((upright_arr >= 0.97).sum())
        result["balance_steps"] = balance_steps
        result["balance_ratio"] = float(balance_steps / max(1, len(trace)))
    else:
        result["final_height"] = 0.0
        result["final_uprightness"] = 0.0
        result["max_height"] = 0.0
        result["max_uprightness"] = 0.0
        result["balance_steps"] = 0
        result["balance_ratio"] = 0.0

    runtime.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Replay hybrid standup+balance policy")
    parser.add_argument(
        "--standup-ckpt",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/runs/standup_v2_r14/checkpoints/checkpoint_u04615.pt",
    )
    parser.add_argument(
        "--balance-model",
        type=str,
        default="/data1/mono/things/combatbench/policy/baseline/follow_v2/u09168/fallback/model.pt",
    )
    parser.add_argument(
        "--env-blueprint",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/blueprints/hybrid_env.yaml",
    )
    parser.add_argument("--switch-uprightness", type=float, default=0.97)
    parser.add_argument("--fall-uprightness", type=float, default=0.30)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video-dir", type=str, default="/tmp/hybrid_videos")
    parser.add_argument("--trace-dir", type=str, default="/tmp/hybrid_traces")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Build env blueprint
    env_pb = ParameterizedEnvBlueprint.load(args.env_blueprint)
    blueprint = env_pb.materialize(
        agent_id="robot_a",
        max_steps=args.max_steps,
    )

    # Build hybrid actor and export to deployable policy
    print("Building HybridActor...")
    actor = HybridActor(
        standup_model_path=args.standup_ckpt,
        balance_model_path=args.balance_model,
        switch_uprightness=args.switch_uprightness,
        device="cpu",
    )
    export_dir = "/tmp/hybrid_replay_export"
    bp = actor.to_blueprint(export_dir)
    bp.config["stochastic"] = False
    policy_a = bp.build()
    policy_b = ZeroPolicy()

    print(f"Switch uprightness: {args.switch_uprightness} (~{np.degrees(np.arccos(args.switch_uprightness)):.1f}°)")
    print(f"Fall uprightness: {args.fall_uprightness} (~{np.degrees(np.arccos(args.fall_uprightness)):.1f}°)")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print()

    all_results = []
    for ep_idx in range(args.episodes):
        seed = 1000 + ep_idx
        video_path = video_dir / f"episode_{ep_idx:03d}.mp4"
        video_plugin = VideoRecorderPlugin(fps=30, output_path=str(video_path))
        trace_plugin = TracePlugin(agent_id="robot_a")

        print(f"--- Episode {ep_idx} (seed={seed}) ---")
        result = run_episode(blueprint, policy_a, policy_b, video_plugin, trace_plugin, seed)

        print(
            f"  steps={result['steps']}  "
            f"max_h={result['max_height']:.3f}  "
            f"final_h={result['final_height']:.3f}  "
            f"max_upright={result['max_uprightness']:.3f}  "
            f"final_upright={result['final_uprightness']:.3f}  "
            f"balance={result['balance_steps']}/{result['steps']} ({result['balance_ratio']:.0%})  "
            f"term={result['termination_reasons']}  "
            f"time={result['elapsed']:.1f}s"
        )

        trace_path = trace_dir / f"episode_{ep_idx:03d}.json"
        with open(trace_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  trace: {trace_path}")
        print(f"  video: {video_path}")
        print()

        all_results.append(result)

    print("=" * 60)
    print("Summary:")
    for i, r in enumerate(all_results):
        print(
            f"  ep{i}: steps={r['steps']} "
            f"max_h={r['max_height']:.3f} final_h={r['final_height']:.3f} "
            f"max_u={r['max_uprightness']:.3f} final_u={r['final_uprightness']:.3f} "
            f"balance={r['balance_ratio']:.0%}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
