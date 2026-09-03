"""验证 surv_rate(duration) 对固定 (direction, force) 是否单调递减。

固定相对方向角度和力大小，扫描 duration = 1..N，每个跑 M 个 episode，
统计存活率，输出 CSV + 终端表格 + 单调性分析。

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 balance_recover/verify_monotonicity.py \
        --policy-export baseline/runs/fixaw_survonly_crossphi2_s42/policy \
        --direction-angles 0,45,90,180 \
        --force 100 \
        --episodes-per-cell 20 \
        --workers 8 \
        --output balance_recover/monotonicity_check.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.rollout import ParallelRollouter
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """计算 Spearman 相关系数。"""
    n = len(x)
    if n < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((rx * ry).sum() / denom)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify duration monotonicity for binary search")
    p.add_argument("--policy-export", required=True,
                   help="Path to policy export directory (containing policy_blueprint.yaml)")
    p.add_argument("--blueprint", type=str,
                   default="baseline/humanoid21/balance_recover/relative_impulse_env.yaml",
                   help="Path to relative impulse env blueprint YAML.")
    p.add_argument("--direction-angles", type=str, default="0,45,90,180",
                   help="Comma-separated direction angles (degrees), relative to robot heading.")
    p.add_argument("--force", type=float, default=100.0,
                   help="Fixed force magnitude (N).")
    p.add_argument("--duration-min", type=int, default=1,
                   help="Minimum duration (action steps).")
    p.add_argument("--duration-max", type=int, default=20,
                   help="Maximum duration (action steps).")
    p.add_argument("--episodes-per-cell", type=int, default=20,
                   help="Episodes per (direction, duration) cell.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=600,
                   help="Max action steps per episode.")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. If omitted, only prints to stdout.")
    p.add_argument("--agent-id", type=str, default="robot_a")
    args = p.parse_args()

    angles = _parse_float_list(args.direction_angles)
    durations = list(range(args.duration_min, args.duration_max + 1))

    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    policy_path_abs = str((Path(args.policy_export) / "policy_blueprint.yaml").resolve())

    env_pb = ParameterizedEnvBlueprint.load(args.blueprint)

    max_steps_val = args.max_steps

    print(f"=== Duration Monotonicity Verification ===")
    print(f"policy: {policy_path_abs}")
    print(f"direction_angles: {angles}")
    print(f"force: {args.force}N")
    print(f"durations: {durations[0]}..{durations[-1]} ({len(durations)} values)")
    print(f"episodes/cell: {args.episodes_per_cell}")
    print(f"workers: {args.workers}")
    print()

    # Build all jobs
    all_jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
    cell_map: List[Tuple[float, int, int, int]] = []  # (angle, duration, start_idx, count)

    base_seed = args.seed
    for angle in angles:
        for duration in durations:
            env_bp = env_pb.materialize(
                max_steps=max_steps_val,
                agent_id=args.agent_id,
                tolerance=6,
                policy_blueprint_path=policy_path_abs,
                force_magnitude=args.force,
                duration_action_steps=duration,
                direction_angle=angle,
            )
            start = len(all_jobs)
            for i in range(args.episodes_per_cell):
                seed = base_seed + i
                all_jobs.append((
                    policy_bp, policy_bp,
                    env_bp, seed,
                    {"agent_id": args.agent_id, "initial_distance": 2.0},
                ))
            cell_map.append((angle, duration, start, args.episodes_per_cell))
            base_seed += args.episodes_per_cell

    total = len(all_jobs)
    print(f"Total episodes: {total}")
    t0 = time.perf_counter()

    rollouter = ParallelRollouter(num_workers=args.workers)
    episodes = rollouter.collect(all_jobs)
    rollouter.close()

    elapsed = time.perf_counter() - t0
    print(f"Rollout time: {elapsed:.1f}s ({elapsed/total:.2f}s/episode)")
    print()

    # Analyze results
    rows: List[Dict[str, Any]] = []

    for angle in angles:
        print(f"--- direction_angle={angle:.0f}°, force={args.force:.0f}N ---")
        print(f"{'dur':>4} {'survived':>9} {'fell':>5} {'total':>6} {'surv_rate':>10} {'mean_len':>9}")
        print("-" * 55)

        angle_rows = []
        for duration in durations:
            cell = next(c for c in cell_map if c[0] == angle and c[1] == duration)
            _, _, start, count = cell
            cell_eps = episodes[start:start + count]
            survived = 0
            lengths = []
            for ep in cell_eps:
                fell = all(r.startswith("imbalance") for r in ep.agent_termination_reason.values())
                if not fell:
                    survived += 1
                lengths.append(ep.num_frames)
            surv_rate = survived / count
            mean_len = float(np.mean(lengths)) if lengths else 0.0
            print(f"{duration:>4d} {survived:>9d} {count - survived:>5d} {count:>6d} {surv_rate:>10.3f} {mean_len:>9.1f}")

            row = {
                "direction_angle": angle,
                "force": args.force,
                "duration": duration,
                "survived": survived,
                "fell": count - survived,
                "total": count,
                "surv_rate": surv_rate,
                "mean_len": mean_len,
            }
            rows.append(row)
            angle_rows.append(row)

        # Monotonicity analysis for this angle
        surv_rates = [r["surv_rate"] for r in angle_rows]
        durs = [r["duration"] for r in angle_rows]
        rho = _spearman_rho(durs, surv_rates)

        # Find non-monotonic points
        non_mono = []
        for i in range(len(surv_rates)):
            for j in range(i + 1, len(surv_rates)):
                if surv_rates[i] < surv_rates[j] - 0.05:  # allow 5% tolerance
                    non_mono.append((durs[i], durs[j], surv_rates[i], surv_rates[j]))

        print(f"\n  Spearman rho: {rho:.3f}")
        if non_mono:
            print(f"  Non-monotonic points (tol=5%): {len(non_mono)}")
            for d_i, d_j, r_i, r_j in non_mono[:5]:
                print(f"    dur={d_i} surv={r_i:.3f} < dur={d_j} surv={r_j:.3f}")
            if len(non_mono) > 5:
                print(f"    ... and {len(non_mono) - 5} more")
        else:
            print(f"  No non-monotonic points (tol=5%)")

        if rho < -0.5:
            print(f"  VERDICT: MONOTONIC (rho={rho:.3f})")
        elif rho < 0:
            print(f"  VERDICT: WEAKLY MONOTONIC (rho={rho:.3f})")
        else:
            print(f"  VERDICT: NOT MONOTONIC (rho={rho:.3f})")
        print()

    # Write CSV
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "direction_angle", "force", "duration",
                "survived", "fell", "total", "surv_rate", "mean_len",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {out_path}")

    # Overall summary
    print("\n=== Overall Summary ===")
    all_rhos = []
    all_monotonic = True
    for angle in angles:
        angle_rows = [r for r in rows if r["direction_angle"] == angle]
        surv_rates = [r["surv_rate"] for r in angle_rows]
        durs = [r["duration"] for r in angle_rows]
        rho = _spearman_rho(durs, surv_rates)
        all_rhos.append(rho)
        is_mono = rho < -0.5
        all_monotonic = all_monotonic and is_mono
        print(f"  angle={angle:.0f}°: rho={rho:.3f} {'MONOTONIC' if is_mono else 'NOT MONOTONIC'}")

    print(f"\n  All monotonic: {all_monotonic}")
    print(f"  Mean rho: {np.mean(all_rhos):.3f}")
    if all_monotonic:
        print("\n  CONCLUSION: Binary search on duration is FEASIBLE.")
    else:
        print("\n  CONCLUSION: Binary search may NOT be reliable. Consider full grid scan.")


if __name__ == "__main__":
    main()
