"""验证 ConstantForcePlugin 直接施力 vs RelativeImpulsePlugin 内部 sim 施力的一致性。

用 RoundRunner 跑两条路径，生成对比视频。

路径 A: ConstantForcePlugin 直接挂在 EnvRuntime 上，EpisodeRunner 跑完整 episode。
路径 B: RelativeImpulsePlugin 在 on_pre_episode 中用内部 sim 施力后写回，
        EpisodeRunner 跑完整 episode。

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 \
        baseline/humanoid21/balance_recover/verify_consistency.py \
        --policy-blueprint-path baseline/runs/.../policy_blueprint.yaml \
        --seed 42 --force 200 --direction 90 --duration 4 \
        --output-dir /tmp/verify_consistency
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.framework.blueprint import EnvBlueprint, ClassSpec
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint, Policy
from envs.framework.recorder import BaseFrameRecorder
from envs.framework.round_runner import RoundRunner
from baseline.humanoid21.balance_recover.freeze_robot_plugin import FreezeRobotPlugin

PHY_STEPS_PER_ACTION = 25
MAX_STEPS = 600


class _ZeroPolicy(Policy):
    """策略返回 zero action，用于排除策略随机性。"""
    def act(self, obs, want_extra=False):
        return np.zeros(21, dtype=np.float32), {}
    def reset(self, seed=None):
        pass


def _build_blueprint_a(force: float, direction: float, duration: int) -> EnvBlueprint:
    """构建路径 A 的 EnvBlueprint: ConstantForcePlugin + FreezeRobotPlugin。"""
    return EnvBlueprint(
        simulator=ClassSpec(
            cls="envs.humanoid21.simulator:Humanoid21Simulator",
            config={
                "initial_distance": 2.0,
                "initial_pose_a": "standing",
                "initial_pose_b": "standing",
            },
        ),
        plugins=(
            ClassSpec(
                cls="baseline.humanoid21.plugins.imbalance_termination:DualImbalanceTerminationPlugin",
                config={"force_threshold": 1.0, "tolerance": 1},
            ),
            ClassSpec(
                cls="envs.humanoid21.disturbance_plugins:ConstantForcePlugin",
                config={
                    "agent_id": "robot_a",
                    "force": force,
                    "direction": direction,
                    "duration_action_steps": duration,
                    "body_name": "torso",
                },
            ),
            ClassSpec(
                cls="baseline.humanoid21.balance_recover.freeze_robot_plugin:FreezeRobotPlugin",
                config={
                    "robot_id": "robot_b",
                    "freeze_steps": duration,
                },
            ),
        ),
        observer_plugins={},
        phy_steps_per_action=PHY_STEPS_PER_ACTION,
        max_steps=MAX_STEPS,
    )


def _build_blueprint_b(policy_bp_path: Optional[str]) -> EnvBlueprint:
    """构建路径 B 的 EnvBlueprint: RelativeImpulsePlugin (from YAML)。"""
    bp_path = Path(__file__).resolve().parent / "weighted_impulse_env.yaml"
    pb = ParameterizedEnvBlueprint.load(bp_path)
    return pb.materialize(
        max_steps=MAX_STEPS,
        policy_blueprint_path=policy_bp_path,
    )


def _extract_result(runtime) -> Dict[str, Any]:
    state = runtime.simulator.get_core_state()
    result = {
        "episode_step": runtime.ctx.episode_step,
        "terminated": runtime.ctx.all_agents_terminated,
    }
    for rid in ("robot_a", "robot_b"):
        s = state[rid]
        for k in ("root_pos", "root_rot", "joint_pos_norm", "root_vel_local",
                   "root_angular_vel_local", "joint_vel_norm"):
            result[f"{rid}_{k}"] = np.asarray(s[k]).copy()
    return result


def run_path(label: str, blueprint: EnvBlueprint, policy_bp_path: Optional[str],
             seed: int, video_path: str, recorder_dir: str,
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """用 RoundRunner 跑一条路径，生成视频和 per-step 记录。"""
    print(f"\n=== {label} ===")

    if policy_bp_path:
        policy = PolicyBlueprint.load(Path(policy_bp_path)).build()
    else:
        policy = _ZeroPolicy()

    video = VideoRecorderPlugin(fps=30, output_path=video_path)
    recorder = BaseFrameRecorder(output_dir=recorder_dir)

    with RoundRunner(
        blueprint=blueprint,
        policy_a=policy,
        policy_b=policy,
        video_plugin=video,
        recorders=(recorder,),
    ) as runner:
        result = runner.run(seed=seed, options=options)
        state_result = _extract_result(runner.runtime)
        state_result["round_result"] = result

    print(f"  steps: {result['steps']}, termination: {result['termination_reasons']}")
    print(f"  video: {video_path}")
    print(f"  recorder: {recorder_dir}")
    return state_result


def compare_robot_a(a: Dict[str, Any], b: Dict[str, Any], duration: int) -> bool:
    print(f"\n=== Comparison (robot_a only) ===")
    ok = True

    ra, rb = a["round_result"], b["round_result"]
    steps_a, steps_b = ra["steps"], rb["steps"]
    if steps_a == steps_b:
        print(f"  OK   steps: {steps_a}")
    elif steps_a == steps_b + duration:
        print(f"  OK   steps: A={steps_a} = B({steps_b}) + duration({duration}) — expected")
    elif steps_b == steps_a + duration:
        print(f"  OK   steps: B={steps_b} = A({steps_a}) + duration({duration}) — expected")
    else:
        print(f"  FAIL steps: A={steps_a} B={steps_b} (diff={abs(steps_a-steps_b)}, duration={duration})")
        ok = False

    keys = [
        "robot_a_root_pos", "robot_a_root_rot", "robot_a_joint_pos_norm",
        "robot_a_root_vel_local", "robot_a_root_angular_vel_local", "robot_a_joint_vel_norm",
    ]
    for key in keys:
        va, vb = a[key], b[key]
        max_diff = float(np.max(np.abs(va - vb)))
        if max_diff > 1e-5:
            print(f"  FAIL {key}: max_diff={max_diff:.2e}")
            ok = False
        else:
            print(f"  OK   {key}: max_diff={max_diff:.2e}")

    return ok


def main():
    p = argparse.ArgumentParser(description="Verify ConstantForcePlugin vs RelativeImpulsePlugin consistency")
    p.add_argument("--policy-blueprint-path", default=None,
                   help="Policy blueprint path. If omitted, uses zero action.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", type=float, default=200.0)
    p.add_argument("--direction", type=float, default=90.0)
    p.add_argument("--duration", type=int, default=4)
    p.add_argument("--output-dir", default="/tmp/verify_consistency",
                   help="Output directory for videos and recordings")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Path A: ConstantForcePlugin direct
    bp_a = _build_blueprint_a(args.force, args.direction, args.duration)
    a = run_path(
        "Path A: ConstantForcePlugin direct",
        bp_a, args.policy_blueprint_path, args.seed,
        str(out / "path_a.mp4"),
        str(out / "path_a_rec"),
    )

    # Path B: RelativeImpulsePlugin internal sim
    bp_b = _build_blueprint_b(args.policy_blueprint_path)
    impulse_options = {
        "impulse_params": {
            "robot_a": {
                "direction_angle": args.direction,
                "force": args.force,
                "duration_action_steps": args.duration,
                "body": "torso",
            },
        },
    }
    b = run_path(
        "Path B: RelativeImpulsePlugin internal sim",
        bp_b, args.policy_blueprint_path, args.seed,
        str(out / "path_b.mp4"),
        str(out / "path_b_rec"),
        options=impulse_options,
    )

    ok = compare_robot_a(a, b, args.duration)
    if ok:
        print("\n✅ CONSISTENT — robot_a state matches")
        sys.exit(0)
    else:
        print("\n❌ INCONSISTENT — robot_a state diverges")
        sys.exit(1)


if __name__ == "__main__":
    main()
