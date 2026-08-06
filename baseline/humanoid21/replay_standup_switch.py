"""Replay standup_v2_r14 + fallback balance recovery policy with video.

Runs the standup_v2 environment with a HeightSwitchPolicy that:
  1. Uses standup_v2_r14 model to get the robot standing
  2. Switches to follow_v2 fallback model when height >= switch_height
  3. Records video and per-step trace (height, uprightness, mode)

Usage:
  PYTHONPATH=/data1/mono/things/combatbench python3 \
    baseline/humanoid21/replay_standup_switch.py \
    --episodes 5 --video-dir /tmp/standup_switch_videos
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

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.framework import BasePlugin
from envs.framework.blueprint import EnvBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.policy import Policy, PolicyBlueprint
from baseline.humanoid21.curriculum.height_switch_policy import HeightSwitchPolicy


def build_switch_policy(
    standup_bp_path: str,
    fallback_bp_path: str,
    switch_height: float = 0.55,
    release_height: float = 0.35,
) -> HeightSwitchPolicy:
    """Build HeightSwitchPolicy from blueprint paths."""
    return HeightSwitchPolicy(
        standup_policy_bp=standup_bp_path,
        fallback_policy_bp=fallback_bp_path,
        switch_height=switch_height,
        release_height=release_height,
    )


class TracePlugin(BasePlugin):
    """Minimal plugin to record per-step height/uprightness/mode for debugging."""

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
        obs = np.asarray(observation, dtype=np.float32)
        action = np.zeros(obs.shape[0] - 54, dtype=np.float32)  # crude action dim
        # Actually, action dim for humanoid21 is 21
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
    policy_a: HeightSwitchPolicy,
    policy_b: Any,
    video_plugin: VideoRecorderPlugin,
    trace_plugin: TracePlugin,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single episode and return trace + termination info."""
    debug_plugins = [video_plugin, trace_plugin]
    runtime = blueprint.build(debug_plugins=debug_plugins)
    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=policy_a,
        policy_b=policy_b,
    )

    policy_a.reset(seed=seed)

    t0 = time.perf_counter()
    runner.run_episode(seed=seed, want_extras=False)
    elapsed = time.perf_counter() - t0

    ctx = runtime.ctx
    result = {
        "steps": int(ctx.episode_step),
        "termination_reasons": {aid: list(ctx.agent_termination_proposals[aid]) for aid in ("robot_a", "robot_b")},
        "seed": seed,
        "elapsed": elapsed,
        "trace": trace_plugin.get_trace(),
    }

    # Extract final height/uprightness from trace
    if trace_plugin.get_trace():
        result["final_height"] = trace_plugin.get_trace()[-1]["height"]
        result["final_uprightness"] = trace_plugin.get_trace()[-1]["uprightness"]
        result["max_height"] = max(t["height"] for t in trace_plugin.get_trace())
        result["max_uprightness"] = max(t["uprightness"] for t in trace_plugin.get_trace())
    else:
        result["final_height"] = 0.0
        result["final_uprightness"] = 0.0
        result["max_height"] = 0.0
        result["max_uprightness"] = 0.0

    runtime.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Replay standup + fallback switch")
    parser.add_argument(
        "--standup-bp",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/runs/standup_v2_r14/policy/policy_blueprint.yaml",
        help="Path to standup policy blueprint YAML",
    )
    parser.add_argument(
        "--fallback-bp",
        type=str,
        default="/data1/mono/things/combatbench/policy/baseline/follow_v2/u09168/fallback/policy_blueprint.yaml",
        help="Path to fallback policy blueprint YAML",
    )
    parser.add_argument(
        "--env-blueprint",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/blueprints/standup_v2_env.yaml",
        help="Path to environment blueprint YAML",
    )
    parser.add_argument(
        "--switch-height",
        type=float,
        default=0.55,
        help="Height threshold to switch from standup to fallback",
    )
    parser.add_argument(
        "--release-height",
        type=float,
        default=0.35,
        help="Height below which to switch back to standup",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/tmp/standup_switch_videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="/tmp/standup_switch_traces",
        help="Directory to save per-step traces",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Max steps per episode",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Load env blueprint with custom max_steps
    from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
    env_pb = ParameterizedEnvBlueprint.load(args.env_blueprint)
    blueprint = env_pb.materialize(
        agent_id="robot_a",
        max_steps=args.max_steps,
    )

    # Build switch policy
    policy_a = build_switch_policy(
        args.standup_bp,
        args.fallback_bp,
        switch_height=args.switch_height,
        release_height=args.release_height,
    )
    policy_b = ZeroPolicy()

    print(f"Standup blueprint: {args.standup_bp}")
    print(f"Fallback blueprint: {args.fallback_bp}")
    print(f"Switch height: {args.switch_height}, Release height: {args.release_height}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print()

    all_results = []
    for ep_idx in range(args.episodes):
        seed = 1000 + ep_idx
        video_path = video_dir / f"episode_{ep_idx:03d}.mp4"
        video_plugin = VideoRecorderPlugin(fps=30, output_path=str(video_path))
        trace_plugin = TracePlugin(agent_id="robot_a")

        # Set debug env for HeightSwitchPolicy
        os.environ["HEIGHT_SWITCH_DEBUG"] = "1"

        print(f"--- Episode {ep_idx} (seed={seed}) ---")
        result = run_episode(blueprint, policy_a, policy_b, video_plugin, trace_plugin, seed)

        print(
            f"  steps={result['steps']}  "
            f"max_h={result['max_height']:.3f}  "
            f"final_h={result['final_height']:.3f}  "
            f"max_upright={result['max_uprightness']:.3f}  "
            f"final_upright={result['final_uprightness']:.3f}  "
            f"term={result['termination_reasons']}  "
            f"time={result['elapsed']:.1f}s"
        )

        # Save trace
        trace_path = trace_dir / f"episode_{ep_idx:03d}.json"
        with open(trace_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  trace: {trace_path}")
        print(f"  video: {video_path}")
        print()

        all_results.append(result)

    # Summary
    print("=" * 60)
    print("Summary:")
    for i, r in enumerate(all_results):
        print(
            f"  ep{i}: steps={r['steps']} max_h={r['max_height']:.3f} "
            f"final_h={r['final_height']:.3f} term={r['termination_reasons']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
