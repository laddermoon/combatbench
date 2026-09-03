"""边界探测脚本（全量并行扫描）。

给定任意策略，对每个 (direction, force, duration) 组合并行跑 episode，
统计存活/摔倒，提取 critical_duration 边界并保存。

全量扫描比二分查找更快：所有 episode 一次性提交并行执行，
无需串行等待。边界精度也更高（每个 duration 都有数据点）。

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/probe_boundary.py \
        --policy-blueprint-path baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/policy_blueprint.yaml \
        --output baseline/humanoid21/balance_recover/boundary.csv

输出：
  - CSV 文件：每行 (direction, force, duration, survived)
  - JSON 文件：每行 (direction, force, critical_duration) + 元数据
  - 终端汇总表
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from baseline.framework.rollout import ParallelRollouter
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description="Boundary probe via parallel full scan")
    p.add_argument("--policy-blueprint-path", required=True,
                   help="Path to policy_blueprint.yaml")
    p.add_argument("--blueprint", type=str,
                   default="baseline/humanoid21/balance_recover/weighted_impulse_env.yaml",
                   help="Path to env blueprint YAML.")
    p.add_argument("--directions", type=str,
                   default="0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5",
                   help="Comma-separated direction angles (degrees).")
    p.add_argument("--forces", type=str, default="40,100,200",
                   help="Comma-separated force magnitudes (N). Default: 40/100/200 (light/medium/heavy).")
    p.add_argument("--duration-min", type=int, default=1,
                   help="Minimum duration to scan (action steps).")
    p.add_argument("--duration-max", type=int, default=40,
                   help="Maximum duration to scan (action steps).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for episodes.")
    p.add_argument("--max-steps", type=int, default=600,
                   help="Max action steps per episode.")
    p.add_argument("--agent-id", type=str, default="robot_a")
    p.add_argument("--workers", type=int, default=96,
                   help="Number of parallel workers.")
    p.add_argument("--output", type=str, default="baseline/humanoid21/balance_recover/boundary.csv",
                   help="Output CSV path (full scan data).")
    p.add_argument("--json-output", type=str, default="baseline/humanoid21/balance_recover/boundary.json",
                   help="Output JSON path (critical_duration summary).")
    args = p.parse_args()

    angles = _parse_float_list(args.directions)
    forces = _parse_float_list(args.forces)
    durations = list(range(args.duration_min, args.duration_max + 1))

    policy_bp_path = Path(args.policy_blueprint_path)
    policy_path_abs = str(policy_bp_path.resolve())
    policy_bp = PolicyBlueprint.load(policy_bp_path)
    env_pb = ParameterizedEnvBlueprint.load(args.blueprint)

    total_episodes = len(angles) * len(forces) * len(durations)
    print(f"=== Boundary Probe (Full Parallel Scan) ===")
    print(f"policy: {policy_path_abs}")
    print(f"directions: {len(angles)} angles")
    print(f"forces: {forces}")
    print(f"durations: {durations[0]}~{durations[-1]} ({len(durations)} values)")
    print(f"total episodes: {total_episodes}")
    print(f"workers: {args.workers}")
    print()

    # 构建所有 jobs
    all_jobs = []
    cell_map: List[tuple] = []  # (angle, force, duration, start_idx)
    base_seed = args.seed

    env_bp = env_pb.materialize(
        max_steps=args.max_steps,
        policy_blueprint_path=policy_path_abs,
    )

    for angle in angles:
        for force in forces:
            for duration in durations:
                options = {
                    "initial_distance": 2.0,
                    "impulse_params": {
                        args.agent_id: {
                            "direction_angle": angle,
                            "force": force,
                            "duration_action_steps": duration,
                            "body": "torso",
                        },
                    },
                }
                start = len(all_jobs)
                all_jobs.append((
                    policy_bp, policy_bp,
                    env_bp, base_seed,
                    options,
                ))
                cell_map.append((angle, force, duration, start))
                base_seed += 1

    t0 = time.perf_counter()

    rollouter = ParallelRollouter(num_workers=args.workers)
    episodes = rollouter.collect(all_jobs)
    rollouter.close()

    elapsed = time.perf_counter() - t0
    print(f"Rollout time: {elapsed:.1f}s ({elapsed/total_episodes:.3f}s/episode)")
    print()

    # 分析结果：提取 critical_duration
    results: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    for angle in angles:
        for force in forces:
            crit_dur = 0
            for duration in durations:
                cell = next(c for c in cell_map
                            if c[0] == angle and c[1] == force and c[2] == duration)
                _, _, _, start = cell
                ep = episodes[start]
                term = ep.agent_termination_reason.get(args.agent_id, "")
                survived = not term.startswith("imbalance")

                raw_rows.append({
                    "direction_angle": angle,
                    "force": force,
                    "duration": duration,
                    "survived": int(survived),
                    "mean_len": ep.num_frames,
                })

                if survived:
                    crit_dur = duration

            results.append({
                "direction_angle": angle,
                "force": force,
                "critical_duration": crit_dur,
            })

    # 保存完整扫描 CSV
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "direction_angle", "force", "duration", "survived", "mean_len",
        ])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Full scan CSV saved to {out_csv}")

    # 保存边界 JSON
    out_json = Path(args.json_output)
    boundary: Dict[str, Any] = {
        "policy_blueprint_path": str(Path(args.policy_blueprint_path).resolve()),
        "agent_id": args.agent_id,
        "duration_range": [args.duration_min, args.duration_max],
        "seed": args.seed,
        "forces": forces,
        "directions": angles,
        "results": results,
    }
    with open(out_json, "w") as f:
        json.dump(boundary, f, indent=2)
    print(f"Boundary JSON saved to {out_json}")

    # 打印汇总表
    print(f"\n=== Boundary Summary (critical_duration = last surviving duration) ===")
    header = f"{'angle':>8}"
    for force in forces:
        header += f" {'F='+str(int(force)):>8}"
    print(header)
    print("-" * len(header))

    for angle in angles:
        row = f"{angle:>8.1f}"
        for force in forces:
            crit = next((r["critical_duration"] for r in results
                         if r["direction_angle"] == angle and r["force"] == force), None)
            row += f" {crit:>8d}" if crit is not None else f" {'?':>8}"
        print(row)

    # 统计
    print(f"\n=== Statistics ===")
    for force in forces:
        crits = [r["critical_duration"] for r in results if r["force"] == force]
        print(f"  F={int(force):>3d}N: mean={sum(crits)/len(crits):.1f}  "
              f"min={min(crits)}  max={max(crits)}")


if __name__ == "__main__":
    main()
