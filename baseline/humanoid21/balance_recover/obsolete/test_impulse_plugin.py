"""检验 ImpulsePerturbationPlugin 生成的扰动状态是否物理合理。

用法::

    python3 baseline/framework/test_impulse_plugin.py \
        --policy-export baseline/runs/train_basic_balance_v2_standup_ppo_20260801_003425/policy \
        --force 200 --duration 4 --direction 1,0,0 --body torso

检验项：
1. 扰动后 root_vel_local 在推力方向上有非零分量
2. 扰动后 root_pos[2] 在合理范围内 (不穿透地面、不飞天)
3. joint_pos_norm 和 joint_vel_norm 无 NaN
4. 不同 force 值产生的状态有可测量差异
5. 不同 seed 产生的状态不同 (随机性生效)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from envs.humanoid21.disturbance_plugins import ImpulsePerturbationPlugin
from envs.humanoid21.simulator import Humanoid21Simulator


def _run_one(
    policy_path: str | None,
    force: float,
    duration: int,
    direction: list[float],
    body: str,
    seed: int,
    phy_steps_per_action: int = 25,
) -> dict:
    """用内部 sim + 插件逻辑跑一次扰动，返回扰动前后状态。"""
    sim = Humanoid21Simulator()
    sim.reset()
    before = sim.get_core_state()["robot_a"]

    plugin = ImpulsePerturbationPlugin(
        target_robot="robot_a",
        policy_blueprint_path=policy_path,
        impulse_body=body,
        force_magnitude=force,
        duration_action_steps=duration,
        direction_mode="fixed",
        fixed_direction=direction,
        phy_steps_per_action=phy_steps_per_action,
        random_seed=seed,
    )

    # 模拟 SimContext 的最小接口
    class FakeCtx:
        class _Accessor:
            def get_core_state(self):
                return sim.get_core_state()
        class _Mutator:
            def set_core_state(self, state):
                sim.set_core_state(state)
        accessor = _Accessor()
        mutator = _Mutator()
        metrics = {}
        episode_options = {
            "impulse_params": {
                "impulse_body": body,
                "impulse_direction": direction,
                "impulse_force": force,
                "impulse_duration_steps": duration,
            }
        }

    ctx = FakeCtx()
    plugin.set_episode_seed(seed)
    plugin.on_pre_episode(ctx)

    after = sim.get_core_state()["robot_a"]
    return {
        "before": before,
        "after": after,
        "metrics": ctx.metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Test ImpulsePerturbationPlugin")
    parser.add_argument("--policy-export", type=str, default=None,
                        help="Path to policy export directory (containing policy_blueprint.yaml)")
    parser.add_argument("--force", type=float, default=200.0)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--direction", type=str, default="1,0,0",
                        help="Force direction as comma-separated x,y,z")
    parser.add_argument("--body", type=str, default="torso")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    policy_path = None
    if args.policy_export:
        policy_path = str(Path(args.policy_export) / "policy_blueprint.yaml")

    direction = [float(x) for x in args.direction.split(",")]

    print(f"=== ImpulsePerturbationPlugin 检验 ===")
    print(f"policy: {policy_path or '(none)'}")
    print(f"force={args.force}N  duration={args.duration} action steps  "
          f"direction={direction}  body={args.body}  seed={args.seed}")
    print()

    # --- 检验 1-3: 基本物理合理性 ---
    result = _run_one(policy_path, args.force, args.duration, direction,
                      args.body, args.seed)
    before = result["before"]
    after = result["after"]
    metrics = result["metrics"]

    print("--- 扰动前后状态对比 ---")
    print(f"root_pos before: {before['root_pos']}")
    print(f"root_pos after:  {after['root_pos']}")
    print(f"root_vel_local before: {before['root_vel_local']}")
    print(f"root_vel_local after:  {after['root_vel_local']}")
    print(f"root_angular_vel_local before: {before['root_angular_vel_local']}")
    print(f"root_angular_vel_local after:  {after['root_angular_vel_local']}")
    print()

    # 检验 1: 推力方向上有非零速度分量
    dir_arr = np.array(direction, dtype=np.float64)
    dir_arr = dir_arr / max(np.linalg.norm(dir_arr), 1e-8)
    vel_after = np.asarray(after["root_vel_local"], dtype=np.float64)
    vel_proj = float(np.dot(vel_after, dir_arr))
    print(f"[检验1] 推力方向速度分量: {vel_proj:.4f} m/s")
    if abs(vel_proj) < 0.01:
        print("  FAIL: 推力方向上速度分量几乎为零，扰动可能未生效")
    else:
        print("  PASS: 推力方向上有非零速度分量")
    print()

    # 检验 2: root_pos[2] 在合理范围
    height = float(after["root_pos"][2])
    print(f"[检验2] root_pos[2] = {height:.4f} m")
    if height < 0.2 or height > 2.0:
        print("  FAIL: 高度异常 (不在 [0.2, 2.0] 范围内)")
    else:
        print("  PASS: 高度在合理范围内")
    print()

    # 检验 3: 无 NaN
    has_nan = False
    for key in ("root_pos", "root_rot", "root_vel_local", "root_angular_vel_local",
                "joint_pos_norm", "joint_vel_norm"):
        arr = np.asarray(after[key])
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"[检验3] FAIL: {key} contains NaN/Inf")
            has_nan = True
    if not has_nan:
        print("[检验3] PASS: 所有状态字段无 NaN/Inf")
    print()

    # 检验 4: 不同 force 产生不同状态
    if args.force > 0:
        result_low = _run_one(policy_path, max(args.force * 0.1, 10), args.duration,
                              direction, args.body, args.seed)
        vel_low = np.asarray(result_low["after"]["root_vel_local"], dtype=np.float64)
        vel_high = vel_after
        vel_diff = float(np.linalg.norm(vel_high - vel_low))
        print(f"[检验4] force={args.force} vs force={max(args.force*0.1,10):.0f} "
              f"速度差: {vel_diff:.4f}")
        if vel_diff < 0.01:
            print("  FAIL: 不同 force 产生的状态几乎相同")
        else:
            print("  PASS: 不同 force 产生可测量差异")
    print()

    # 检验 5: 不同 seed 产生不同状态
    result_seed2 = _run_one(policy_path, args.force, args.duration, direction,
                            args.body, seed=args.seed + 1000)
    vel_s1 = vel_after
    vel_s2 = np.asarray(result_seed2["after"]["root_vel_local"], dtype=np.float64)
    seed_diff = float(np.linalg.norm(vel_s1 - vel_s2))
    print(f"[检验5] seed={args.seed} vs seed={args.seed+1000} 速度差: {seed_diff:.4f}")
    # 随机模式下不同 seed 应该产生不同方向，但固定模式下方向相同
    # 固定模式下策略 stochastic=True 可能仍有差异；若 deterministic 则可能相同
    if seed_diff < 0.001:
        print("  WARN: 不同 seed 状态相同 (策略可能 deterministic，或固定模式下方向相同)")
    else:
        print("  PASS: 不同 seed 产生不同状态")
    print()

    # --- metrics 检查 ---
    print("--- 插件 metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print()
    print("=== 检验完成 ===")


if __name__ == "__main__":
    main()
