"""Replay 4-stage standup checkpoint with BaseFrameRecorder.

Usage:
  PYTHONPATH=/data1/mono/things/combatbench python3 \
    baseline/humanoid21/replay_4stage_recorder.py \
    --policy-dir baseline/humanoid21/runs/4stage_standup_4stage_b_20260709_182916/policy_exports/u04000 \
    --output-dir /data1/dev/replay_4stage_u04000 \
    --episodes 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.framework.blueprint import EnvBlueprint
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import Policy, PolicyBlueprint
from envs.framework.recorder import BaseFrameRecorder


class ZeroPolicy(Policy):
    def act(self, observation, want_extra=False):
        return np.zeros(21, dtype=np.float32), None

    def reset(self, seed=None):
        pass

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Replay 4-stage checkpoint with recorder")
    parser.add_argument(
        "--policy-dir",
        type=str,
        default="baseline/humanoid21/runs/4stage_standup_4stage_b_20260709_182916/policy_exports/u04000",
    )
    parser.add_argument(
        "--env-blueprint",
        type=str,
        default=str(Path(__file__).resolve().parent / "blueprints" / "standup_4stage_env.yaml"),
    )
    parser.add_argument("--output-dir", type=str, default="/data1/dev/replay_4stage_u04000")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    policy_bp_path = Path(args.policy_dir) / "policy_blueprint.yaml"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load env blueprint
    env_pb = ParameterizedEnvBlueprint.load(args.env_blueprint)
    blueprint = env_pb.materialize(agent_id="robot_a", max_steps=args.max_steps)

    # Load policy from blueprint
    policy_a = PolicyBlueprint.load(str(policy_bp_path)).build()
    policy_b = ZeroPolicy()

    print(f"Policy: {policy_bp_path}")
    print(f"Env: {args.env_blueprint}")
    print(f"Output: {output_dir}")
    print(f"Episodes: {args.episodes}, max_steps: {args.max_steps}")
    print()

    for ep_idx in range(args.episodes):
        seed = 1000 + ep_idx
        ep_dir = output_dir / f"episode_{ep_idx:03d}"

        recorder = BaseFrameRecorder(
            output_dir=ep_dir,
            save_image=True,
            save_observer_outputs=True,
            save_core_state=True,
            save_derived_state=True,
            save_sensor_data=False,
            save_static_data=True,
            save_action=True,
            save_action_extras=False,
            save_observation=False,
        )

        runtime = blueprint.build()
        runtime.attach_recorder(recorder)

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
        steps = int(ctx.episode_step)
        terms = {aid: list(ctx.agent_termination_proposals[aid]) for aid in ("robot_a", "robot_b")}

        # Read observer output from last step
        try:
            oo = runtime.get_observer_output("standup")
            last_stage = float(oo.get("stage", -1))
            last_potential = float(oo.get("potential", 0.0))
        except Exception:
            last_stage = -1
            last_potential = 0.0

        print(
            f"  ep{ep_idx}: steps={steps}  "
            f"stage={last_stage:.3f}  potential={last_potential:.3f}  "
            f"term={terms}  time={elapsed:.1f}s"
        )

        runtime.close()

    print(f"\nDone. Recordings in {output_dir}/")


if __name__ == "__main__":
    main()
