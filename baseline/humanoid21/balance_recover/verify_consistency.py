"""验证 ConstantForcePlugin 直接施力 vs RelativeImpulsePlugin 内部 sim 施力的一致性。

两条路径：
  A. ConstantForcePlugin 挂在真实 EnvRuntime 上，EpisodeRunner 跑完整 episode
  B. RelativeImpulsePlugin 在 on_pre_episode 中用内部 sim 施力后写回，
     然后 EpisodeRunner 继续跑完整 episode

预期：两条路径的 episode 长度、终止原因、最终 core state 一致。

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 \
        baseline/humanoid21/balance_recover/verify_consistency.py \
        --policy-blueprint-path baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/policy_blueprint.yaml \
        --seed 42 --force 200 --direction 90 --duration 4
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

from envs.framework.episode_runner import EpisodeRunner
from envs.framework.env_runtime import EnvRuntime
from envs.framework.policy import PolicyBlueprint, Policy
from envs.humanoid21.simulator import Humanoid21Simulator
from envs.humanoid21.disturbance_plugins import ConstantForcePlugin
from baseline.humanoid21.balance_recover.relative_impulse_plugin import RelativeImpulsePlugin
from baseline.humanoid21.plugins.imbalance_termination import DualImbalanceTerminationPlugin

PHY_STEPS_PER_ACTION = 25
MAX_STEPS = 600


class _ZeroPolicy(Policy):
    """策略返回 zero action，用于排除策略随机性。"""
    def act(self, obs, want_extra=False):
        return np.zeros(21, dtype=np.float32), {}
    def reset(self, seed=None):
        pass


def _build_runtime(plugins, phy_steps_per_action=25, max_steps=600):
    sim = Humanoid21Simulator(initial_distance=2.0, initial_pose_a="standing", initial_pose_b="standing")
    return EnvRuntime(
        simulator=sim,
        plugins=plugins,
        phy_steps_per_action=phy_steps_per_action,
        max_steps=max_steps,
    )


def _extract_result(runtime: EnvRuntime) -> Dict[str, Any]:
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


def run_path_a(policy_bp_path: str, seed: int, force: float, direction: float,
               duration: int) -> Dict[str, Any]:
    """路径 A: ConstantForcePlugin 直接挂在真实 EnvRuntime 上。

    用 EnvRuntime 直接跑 duration 步（zero action for both），
    返回 impulse 后的 core state。
    """
    print(f"\n=== Path A: ConstantForcePlugin direct ===")
    force_plugin = ConstantForcePlugin(
        agent_id="robot_a",
        force=force,
        direction=direction,
        duration_action_steps=duration,
        body_name="torso",
    )
    runtime = _build_runtime(plugins=[force_plugin])

    runtime.reset(seed=seed)
    zero_action = np.zeros(21, dtype=np.float32)

    for _ in range(duration):
        runtime.step(zero_action, zero_action)

    result = _extract_result(runtime)
    print(f"  episode_step after impulse: {result['episode_step']}")
    runtime.close()
    return result


def run_path_b(policy_bp_path: str, seed: int, force: float, direction: float,
               duration: int) -> Dict[str, Any]:
    """路径 B: RelativeImpulsePlugin 内部 sim 施力后写回。

    用 EnvRuntime.reset() 触发 on_pre_episode（impulse），
    然后立即取 core state（episode_step=0，未跑任何 action step）。
    """
    print(f"\n=== Path B: RelativeImpulsePlugin internal sim ===")
    impulse_plugin = RelativeImpulsePlugin(
        target_robots=["robot_a"],
        policy_blueprint_path=None,  # zero action in internal sim
        impulse_body="torso",
        phy_steps_per_action=PHY_STEPS_PER_ACTION,
    )
    runtime = _build_runtime(plugins=[impulse_plugin])

    options = {
        "impulse_params": {
            "robot_a": {
                "direction_angle": direction,
                "force": force,
                "duration_action_steps": duration,
                "body": "torso",
            },
        },
    }

    runtime.reset(seed=seed, options=options)

    result = _extract_result(runtime)
    print(f"  episode_step after impulse: {result['episode_step']}")
    runtime.close()
    return result


def compare_robot_a(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    print(f"\n=== Comparison (robot_a only) ===")
    ok = True

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


def run_path_a_full(policy_bp_path: str, seed: int, force: float, direction: float,
                    duration: int) -> Dict[str, Any]:
    """路径 A 完整 episode: ConstantForcePlugin 直接挂载，EpisodeRunner 跑完整 episode。"""
    print(f"\n=== Path A (full episode): ConstantForcePlugin direct ===")
    force_plugin = ConstantForcePlugin(
        agent_id="robot_a",
        force=force,
        direction=direction,
        duration_action_steps=duration,
        body_name="torso",
    )
    term_plugin = DualImbalanceTerminationPlugin(force_threshold=1.0, tolerance=1)
    runtime = _build_runtime(plugins=[term_plugin, force_plugin], max_steps=MAX_STEPS)

    if policy_bp_path:
        policy_bp = PolicyBlueprint.load(Path(policy_bp_path))
        policy = policy_bp.build()
    else:
        policy = _ZeroPolicy()

    runner = EpisodeRunner(runtime=runtime, policy_a=policy, policy_b=policy)
    runner.run_episode(seed=seed)

    result = _extract_result(runtime)
    print(f"  episode_step: {result['episode_step']}, terminated: {result['terminated']}")
    runtime.close()
    return result


def run_path_b_full(policy_bp_path: str, seed: int, force: float, direction: float,
                    duration: int) -> Dict[str, Any]:
    """路径 B 完整 episode: RelativeImpulsePlugin + EpisodeRunner 跑完整 episode。"""
    print(f"\n=== Path B (full episode): RelativeImpulsePlugin internal sim ===")
    impulse_plugin = RelativeImpulsePlugin(
        target_robots=["robot_a"],
        policy_blueprint_path=policy_bp_path if policy_bp_path else None,
        impulse_body="torso",
        phy_steps_per_action=PHY_STEPS_PER_ACTION,
    )
    term_plugin = DualImbalanceTerminationPlugin(force_threshold=1.0, tolerance=1)
    runtime = _build_runtime(plugins=[term_plugin, impulse_plugin], max_steps=MAX_STEPS)

    if policy_bp_path:
        policy_bp = PolicyBlueprint.load(Path(policy_bp_path))
        policy = policy_bp.build()
    else:
        policy = _ZeroPolicy()

    options = {
        "impulse_params": {
            "robot_a": {
                "direction_angle": direction,
                "force": force,
                "duration_action_steps": duration,
                "body": "torso",
            },
        },
    }

    runner = EpisodeRunner(runtime=runtime, policy_a=policy, policy_b=policy)
    runner.run_episode(seed=seed, options=options)

    result = _extract_result(runtime)
    print(f"  episode_step: {result['episode_step']}, terminated: {result['terminated']}")
    runtime.close()
    return result


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    print(f"\n=== Comparison ===")
    ok = True

    if a["episode_step"] != b["episode_step"]:
        print(f"  FAIL episode_step: A={a['episode_step']} B={b['episode_step']}")
        ok = False
    else:
        print(f"  OK   episode_step: {a['episode_step']}")

    if a["terminated"] != b["terminated"]:
        print(f"  FAIL terminated: A={a['terminated']} B={b['terminated']}")
        ok = False
    else:
        print(f"  OK   terminated: {a['terminated']}")

    keys = [
        "robot_a_root_pos", "robot_a_root_rot", "robot_a_joint_pos_norm",
        "robot_a_root_vel_local", "robot_a_root_angular_vel_local", "robot_a_joint_vel_norm",
        "robot_b_root_pos", "robot_b_root_rot", "robot_b_joint_pos_norm",
        "robot_b_root_vel_local", "robot_b_root_angular_vel_local", "robot_b_joint_vel_norm",
    ]
    for key in keys:
        va, vb = a[key], b[key]
        max_diff = float(np.max(np.abs(va - vb)))
        if max_diff > 1e-6:
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
    p.add_argument("--full", action="store_true",
                   help="Run full episode comparison (not just post-impulse state)")
    args = p.parse_args()

    if args.full:
        a = run_path_a_full(args.policy_blueprint_path, args.seed, args.force, args.direction, args.duration)
        b = run_path_b_full(args.policy_blueprint_path, args.seed, args.force, args.direction, args.duration)
        ok = compare(a, b)
    else:
        a = run_path_a(args.policy_blueprint_path, args.seed, args.force, args.direction, args.duration)
        b = run_path_b(args.policy_blueprint_path, args.seed, args.force, args.direction, args.duration)

        print(f"\nNote: Comparing post-impulse robot_a state only.")
        print(f"  Path A: episode_step={a['episode_step']} (impulse steps counted)")
        print(f"  Path B: episode_step={b['episode_step']} (impulse in pre-episode, not counted)")
        print(f"  robot_a state should match if impulse physics is correct.")
        print(f"  robot_b expected to differ: Path B resets robot_b each step in internal sim.")

        ok = compare_robot_a(a, b)
    if ok:
        print("\n✅ CONSISTENT — both paths produce identical results")
        sys.exit(0)
    else:
        print("\n❌ INCONSISTENT — paths diverge")
        sys.exit(1)


if __name__ == "__main__":
    main()
