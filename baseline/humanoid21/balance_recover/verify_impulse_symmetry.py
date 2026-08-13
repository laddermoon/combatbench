"""Verify RelativeImpulsePlugin applies the impulse symmetrically to both robots.

Root cause being checked: EnvRuntime.step(action_a, action_b) is positional,
so the internal impulse-application loop must route the policy action into the
*target* robot's slot. If it always fills robot_a's slot, then whenever the
target is robot_b the target receives an uncommanded mid-range pose target and
sags, producing unrealistic initial states for robot_b only.

Usage:
    python3 baseline/humanoid21/balance_recover/verify_impulse_symmetry.py \
        --policy baseline/runs/recovery_v3_gen0/policy_exports/u00415/policy_blueprint.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_runtime(policy_bp_path: str):
    from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

    bp_path = Path(__file__).resolve().parent / "weighted_impulse_env.yaml"
    env_pb = ParameterizedEnvBlueprint.load(bp_path)
    env_bp = env_pb.materialize(
        max_steps=200,
        policy_blueprint_path=str(Path(policy_bp_path).resolve()),
    )
    return env_bp.build()


def _heights(runtime) -> dict:
    state = runtime.ctx.accessor.get_core_state()
    return {rid: float(state[rid]["root_pos"][2]) for rid in ("robot_a", "robot_b")}


def run_case(runtime, target: str, params: dict, seed: int) -> float:
    """Reset with an impulse applied to ``target`` only; return its torso height."""
    runtime.reset(seed=seed, options={"impulse_params": {target: params}})
    return _heights(runtime)[target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    runtime = _build_runtime(args.policy)

    cases = [
        {"direction_angle": 0.0, "force": 100.0, "duration_action_steps": 4, "body": "torso"},
        {"direction_angle": 90.0, "force": 100.0, "duration_action_steps": 4, "body": "torso"},
        {"direction_angle": 180.0, "force": 100.0, "duration_action_steps": 4, "body": "torso"},
        {"direction_angle": 270.0, "force": 100.0, "duration_action_steps": 4, "body": "torso"},
    ]

    print(f"{'angle':>6} {'h_a':>8} {'h_b':>8} {'diff':>8}")
    diffs = []
    for p in cases:
        h_a = run_case(runtime, "robot_a", p, args.seed)
        h_b = run_case(runtime, "robot_b", p, args.seed)
        d = abs(h_a - h_b)
        diffs.append(d)
        print(f"{p['direction_angle']:>6.0f} {h_a:>8.4f} {h_b:>8.4f} {d:>8.4f}")

    runtime.close()

    max_diff = max(diffs)
    print(f"\nmax |h_a - h_b| = {max_diff:.4f}")
    if max_diff > 0.05:
        print("FAIL: asymmetric impulse response -> target robot is not policy-controlled")
        raise SystemExit(1)
    print("PASS: impulse response is symmetric across robot_a / robot_b")


if __name__ == "__main__":
    main()
