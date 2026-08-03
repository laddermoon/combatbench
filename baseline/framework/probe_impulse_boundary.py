"""Boundary mapping: sweep (force, duration) grid and measure survival rate.

For each grid cell, runs N episodes with the given impulse parameters
(random horizontal direction per episode) and records whether the robot
recovered (no imbalance termination) or fell.

Output: CSV file with columns force, duration, survived, episode_length,
        and a printed summary table.

Usage::

    python3 baseline/framework/probe_impulse_boundary.py \
        --policy-export baseline/runs/<basic_balance_v2_run>/policy \
        --force-grid 50,100,150,200,300,400,500,700 \
        --duration-grid 1,2,4,8,12,20 \
        --episodes-per-cell 20 \
        --output baseline/runs/recovery_iter/gen0_boundary.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.common.rollout import ParallelRollouter
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def _build_jobs(
    env_pb: ParameterizedEnvBlueprint,
    policy_bp: PolicyBlueprint,
    force: float,
    duration: int,
    episodes: int,
    base_seed: int,
    agent_id: str,
) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
    """Build N jobs for one grid cell, each with random direction."""
    env_bp = env_pb.materialize(
        max_steps=env_pb.parameters_dict.get("max_steps", {}).get("default", 600)
        if hasattr(env_pb, "parameters_dict") else 600,
        agent_id=agent_id,
        tolerance=6,
        force_magnitude=force,
        duration_action_steps=duration,
        direction_mode="random_horizontal",
    )
    jobs = []
    for i in range(episodes):
        seed = base_seed + i
        jobs.append((
            policy_bp, policy_bp,
            env_bp, seed,
            {"agent_id": agent_id, "initial_distance": 2.0},
        ))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description="Impulse boundary mapping")
    p.add_argument("--policy-export", required=True,
                   help="Path to policy export directory (containing policy_blueprint.yaml)")
    p.add_argument("--blueprint", type=str,
                   default="baseline/humanoid21/blueprints/impulse_boundary_env.yaml",
                   help="Path to impulse boundary env blueprint YAML.")
    p.add_argument("--force-grid", type=str, default="50,100,150,200,300,400,500,700",
                   help="Comma-separated force values (N).")
    p.add_argument("--duration-grid", type=str, default="1,2,4,8,12,20",
                   help="Comma-separated duration values (action steps).")
    p.add_argument("--episodes-per-cell", type=int, default=20,
                   help="Episodes per grid cell.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. If omitted, only prints to stdout.")
    p.add_argument("--agent-id", type=str, default="robot_a")
    args = p.parse_args()

    forces = _parse_float_list(args.force_grid)
    durations = _parse_int_list(args.duration_grid)

    # Load policy blueprint
    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    policy_path_abs = str((Path(args.policy_export) / "policy_blueprint.yaml").resolve())

    # Load env blueprint
    env_pb = ParameterizedEnvBlueprint.load(args.blueprint)

    # Determine max_steps from blueprint defaults
    # We need to read the default from the blueprint template
    template = env_pb.template
    runtime_section = template.get("runtime", {})
    max_steps_val = runtime_section.get("max_steps", 600)
    if isinstance(max_steps_val, str):
        # It's a ${...} reference; find the parameter default
        for param in env_pb.parameters:
            if param.name == "max_steps":
                max_steps_val = param.default
                break
    max_steps_val = int(max_steps_val)

    print(f"=== Impulse Boundary Mapping ===")
    print(f"policy: {policy_path_abs}")
    print(f"forces: {forces}")
    print(f"durations: {durations}")
    print(f"episodes/cell: {args.episodes_per_cell}")
    print(f"workers: {args.workers}")
    print(f"max_steps: {max_steps_val}")
    print()

    # Build all jobs across all grid cells
    all_jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
    cell_map: List[Tuple[float, int, int, int]] = []  # (force, duration, start_idx, count)

    base_seed = args.seed
    for force in forces:
        for duration in durations:
            env_bp = env_pb.materialize(
                max_steps=max_steps_val,
                agent_id=args.agent_id,
                tolerance=6,
                policy_blueprint_path=policy_path_abs,
                force_magnitude=force,
                duration_action_steps=duration,
                direction_mode="random_horizontal",
            )
            start = len(all_jobs)
            for i in range(args.episodes_per_cell):
                seed = base_seed + i
                all_jobs.append((
                    policy_bp, policy_bp,
                    env_bp, seed,
                    {"agent_id": args.agent_id, "initial_distance": 2.0},
                ))
            cell_map.append((force, duration, start, args.episodes_per_cell))
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
    print(f"{'force':>7} {'dur':>4} {'survived':>9} {'fell':>5} {'total':>6} {'surv_rate':>10} {'mean_len':>9}")
    print("-" * 60)

    for force, duration, start, count in cell_map:
        cell_eps = episodes[start:start + count]
        survived = 0
        lengths = []
        for ep in cell_eps:
            fell = "imbalance" in ep.termination_proposals
            if not fell:
                survived += 1
            lengths.append(ep.num_frames)
        surv_rate = survived / count
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        print(f"{force:>7.0f} {duration:>4d} {survived:>9d} {count - survived:>5d} {count:>6d} {surv_rate:>10.3f} {mean_len:>9.1f}")

        rows.append({
            "force": force,
            "duration": duration,
            "survived": survived,
            "fell": count - survived,
            "total": count,
            "surv_rate": surv_rate,
            "mean_len": mean_len,
        })

    # Write CSV
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["force", "duration", "survived", "fell", "total", "surv_rate", "mean_len"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {out_path}")

    # Summary checks
    print("\n=== Summary Checks ===")
    # Monotonicity: for fixed duration, survival rate should generally decrease with force
    for duration in durations:
        dur_rows = [r for r in rows if r["duration"] == duration]
        rates = [r["surv_rate"] for r in dur_rows]
        if len(rates) >= 2:
            # Check if first (lowest force) > last (highest force)
            if rates[0] > rates[-1]:
                print(f"  duration={duration}: monotonic trend OK ({rates[0]:.2f} -> {rates[-1]:.2f})")
            else:
                print(f"  duration={duration}: WARNING non-monotonic ({rates[0]:.2f} -> {rates[-1]:.2f})")

    # Boundary existence: any cell with surv_rate in [0.2, 0.8]?
    boundary_cells = [r for r in rows if 0.2 <= r["surv_rate"] <= 0.8]
    if boundary_cells:
        print(f"  Boundary cells (surv_rate in [0.2, 0.8]): {len(boundary_cells)}")
        for r in boundary_cells:
            print(f"    force={r['force']:.0f}N  duration={r['duration']}  surv={r['surv_rate']:.3f}")
    else:
        print("  WARNING: no boundary cells found — grid may not cover the transition zone")

    # Extreme values
    low_force_rows = [r for r in rows if r["force"] == min(forces)]
    high_force_rows = [r for r in rows if r["force"] == max(forces)]
    low_max = max(r["surv_rate"] for r in low_force_rows)
    high_min = min(r["surv_rate"] for r in high_force_rows)
    print(f"  Lowest force ({min(forces)}N): max surv_rate={low_max:.3f}")
    print(f"  Highest force ({max(forces)}N): min surv_rate={high_min:.3f}")


if __name__ == "__main__":
    main()
