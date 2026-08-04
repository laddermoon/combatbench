"""Iterative balance recovery training loop.

Automates the generate → sample → train → re-generate cycle, tracking the
recovery boundary across iterations. Each iteration:

1. Generate state bank using current policy (ImpulsePerturbationPlugin)
   — over-generate: large grid × many episodes per cell
2. Analyze per-cell survival rates, compute boundary_force (50% survival point)
3. Sample N training states from full bank → train_state_bank.npz
4. Train PPO with sampled states (warm-start from current policy)
5. Evaluate: measure boundary_force, check if it shifted rightward

The boundary_force metric tracks progress: if the policy improves, it can
survive stronger perturbations, so the 50% survival point shifts rightward.

Usage::

    python3 baseline/framework/recovery_iter_loop.py \\
        --base-policy baseline/runs/train_basic_balance_v2_standup_ppo_20260801_003425/policy \\
        --output-dir baseline/runs/recovery_iter \\
        --max-iters 5 \\
        --train-updates 5000 \\
        --episodes-per-cell 100 \\
        --train-states 5000

    # Smoke test (1 iter, 2 train updates, small grid)
    python3 baseline/framework/recovery_iter_loop.py \\
        --base-policy baseline/runs/.../policy \\
        --output-dir /tmp/recovery_iter_smoke \\
        --max-iters 1 --train-updates 2 \\
        --force-grid 50,100,150 --duration-grid 2,4 \\
        --episodes-per-cell 4 --gen-workers 4 --rollout-workers 2 \\
        --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CB_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PY = CB_ROOT / "baseline" / "framework" / "train.py"
GEN_PY = CB_ROOT / "baseline" / "framework" / "generate_state_bank.py"

CORE_STATE_TOTAL = 55


# ---------------------------------------------------------------------------
# State bank analysis
# ---------------------------------------------------------------------------

def analyze_state_bank(npz_path: str) -> Dict[str, Any]:
    """Load .npz state bank and compute per-cell survival statistics.

    Returns dict with:
      - n: total states
      - overall_survival_rate
      - cells: list of {force, duration, n, survived, rate, mean_ep_len}
      - boundary_force: interpolated force at 50% survival (across durations)
      - boundary_cells: indices of cells with survival in [lo, hi]
    """
    data = np.load(npz_path, allow_pickle=True)
    forces = data["forces"]
    durations = data["durations"]
    labels = data["labels"]
    ep_lengths = data["ep_lengths"]

    n = len(labels)
    overall_rate = float(labels.mean())

    # Group by (force, duration) cell
    cells: List[Dict[str, Any]] = []
    unique_forces = sorted(set(forces.tolist()))
    unique_durations = sorted(set(durations.tolist()))

    for f in unique_forces:
        for d in unique_durations:
            mask = (forces == f) & (durations == d)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            cell_labels = labels[idx]
            cell_ep_len = ep_lengths[idx]
            survived = int(cell_labels.sum())
            cells.append({
                "force": float(f),
                "duration": int(d),
                "n": len(idx),
                "survived": survived,
                "rate": survived / len(idx),
                "mean_ep_len": float(cell_ep_len.mean()),
                "indices": idx.tolist(),
            })

    # Compute boundary_force: for each duration, find force at ~50% survival
    # via linear interpolation, then average across durations
    boundary_forces_per_dur: List[float] = []
    for d in unique_durations:
        dur_cells = [c for c in cells if c["duration"] == d]
        dur_cells.sort(key=lambda c: c["force"])
        if len(dur_cells) < 2:
            continue
        # Find cells bracketing 50% survival
        for i in range(len(dur_cells) - 1):
            r_lo = dur_cells[i]["rate"]
            r_hi = dur_cells[i + 1]["rate"]
            if r_lo >= 0.5 >= r_hi or r_lo <= 0.5 <= r_hi:
                if abs(r_hi - r_lo) < 1e-6:
                    bf = dur_cells[i]["force"]
                else:
                    t = (0.5 - r_lo) / (r_hi - r_lo)
                    bf = dur_cells[i]["force"] + t * (
                        dur_cells[i + 1]["force"] - dur_cells[i]["force"]
                    )
                boundary_forces_per_dur.append(bf)
                break

    boundary_force = (
        float(np.mean(boundary_forces_per_dur))
        if boundary_forces_per_dur
        else float(unique_forces[-1])  # fallback: max force
    )

    return {
        "n": n,
        "overall_survival_rate": overall_rate,
        "cells": cells,
        "boundary_force": boundary_force,
    }


def sample_training_states(
    npz_path: str,
    output_path: str,
    n_target: int,
    seed: int = 42,
) -> int:
    """Sample n_target states from full state bank for training.

    If total states <= n_target, keeps all.
    Otherwise randomly samples n_target states (seeded for reproducibility).
    Saves sampled .npz and returns number of states kept.
    """
    data = np.load(npz_path, allow_pickle=True)
    n_total = len(data["labels"])

    if n_total <= n_target:
        # Keep all states — not enough to sample
        idx = np.arange(n_total)
        print(f"  [sample] Total {n_total} <= target {n_target}, keeping all")
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=n_target, replace=False)
        idx.sort()
        print(f"  [sample] Sampled {n_target}/{n_total} states (seed={seed})")

    np.savez_compressed(
        output_path,
        states=data["states"][idx],
        observations=data["observations"][idx],
        forces=data["forces"][idx],
        durations=data["durations"][idx],
        directions=data["directions"][idx],
        labels=data["labels"][idx],
        ep_lengths=data["ep_lengths"][idx],
        core_state_fields=data["core_state_fields"],
        core_state_dims=data["core_state_dims"],
    )
    return len(idx)


def adapt_force_grid(
    prev_boundary_force: float,
    prev_grid: List[float],
    all_survived: bool,
    all_fell: bool,
) -> List[float]:
    """Adapt force grid for next iteration based on boundary position.

    Strategy:
    - If all survived: push grid higher (1.5x max)
    - If all fell: pull grid lower (0.5x min)
    - Otherwise: focus around boundary_force (0.5x to 2x boundary)
    """
    if all_survived:
        max_f = max(prev_grid)
        new_max = max_f * 1.5
        # Generate ~8 points from prev max to new max
        new_forces = list(np.linspace(max_f, new_max, 6).round().astype(int))
        # Merge with old grid, dedup, sort
        combined = sorted(set(prev_grid + new_forces))
        return combined

    if all_fell:
        min_f = min(prev_grid)
        new_min = max(5, int(min_f * 0.5))
        new_forces = list(np.linspace(new_min, min_f, 6).round().astype(int))
        combined = sorted(set([int(f) for f in new_forces] + prev_grid))
        return combined

    # Focus around boundary
    lo = max(5, int(prev_boundary_force * 0.5))
    hi = int(prev_boundary_force * 2.0)
    new_forces = sorted(set(
        int(x) for x in np.linspace(lo, hi, 8).round().astype(int)
    ))
    # Merge with old grid for continuity
    combined = sorted(set(prev_grid + new_forces))
    return combined


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run_generate_state_bank(
    policy_path: str,
    output_path: str,
    force_grid: List[float],
    duration_grid: List[int],
    episodes_per_cell: int,
    workers: int,
    max_steps: int,
    tolerance: int,
    agent_id: str,
    seed: int,
) -> None:
    """Run generate_state_bank.py via subprocess."""
    cmd = [
        sys.executable, str(GEN_PY),
        "--policy-export", policy_path,
        "--output", output_path,
        "--force-grid", ",".join(str(int(f)) for f in force_grid),
        "--duration-grid", ",".join(str(d) for d in duration_grid),
        "--episodes-per-cell", str(episodes_per_cell),
        "--workers", str(workers),
        "--max-steps", str(max_steps),
        "--tolerance", str(tolerance),
        "--agent-id", agent_id,
        "--seed", str(seed),
    ]
    print(f"  [gen] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CB_ROOT)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"generate_state_bank.py failed (exit {result.returncode})")
    # Print last few lines for progress
    lines = result.stdout.strip().split("\n")
    for line in lines[-8:]:
        print(f"  [gen] {line}")


def run_train(
    base_policy_path: str,
    policy_blueprint_path: str,
    force_min: float,
    force_max: float,
    dur_min: int,
    dur_max: int,
    run_dir: str,
    rollout_workers: int,
    train_updates: int,
    smoke: bool = False,
) -> str:
    """Run train.py with balance_recover_v4 (online impulse perturbation).

    Returns path to exported policy.
    """
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--experiment", "balance_recover_v4",
        "--algo", "ppo",
        "--run-dir", run_dir,
        "--no-snapshot",
    ]
    if smoke:
        cmd.append("--smoke")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CB_ROOT)
    if base_policy_path:
        env["BASE_POLICY_PATH"] = str(Path(base_policy_path).resolve())
    env["POLICY_BLUEPRINT_PATH"] = str(Path(policy_blueprint_path).resolve())
    env["IMPULSE_FORCE_MIN"] = str(force_min)
    env["IMPULSE_FORCE_MAX"] = str(force_max)
    env["IMPULSE_DURATION_MIN"] = str(dur_min)
    env["IMPULSE_DURATION_MAX"] = str(dur_max)
    env["ROLLOUT_WORKERS"] = str(rollout_workers)
    env["TRAIN_UPDATES"] = str(train_updates)

    print(f"  [train] IMPULSE_FORCE=[{force_min}, {force_max}] "
          f"DURATION=[{dur_min}, {dur_max}]")
    print(f"  [train] POLICY_BLUEPRINT_PATH={env['POLICY_BLUEPRINT_PATH']}")
    print(f"  [train] BASE_POLICY_PATH={env.get('BASE_POLICY_PATH', '(none)')}")
    print(f"  [train] ROLLOUT_WORKERS={env['ROLLOUT_WORKERS']} TRAIN_UPDATES={env['TRAIN_UPDATES']}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"train.py failed (exit {result.returncode})")

    # Print last few lines
    lines = result.stdout.strip().split("\n")
    for line in lines[-10:]:
        print(f"  [train] {line}")

    policy_dir = Path(run_dir) / "policy"
    if not policy_dir.exists():
        raise RuntimeError(f"Policy not exported to {policy_dir}")
    return str(policy_dir)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Iterative balance recovery training loop"
    )
    p.add_argument("--base-policy", required=True,
                   help="Initial policy directory (with model.pt + policy_blueprint.yaml)")
    p.add_argument("--output-dir", default="baseline/runs/recovery_iter",
                   help="Root output directory")
    p.add_argument("--max-iters", type=int, default=5,
                   help="Maximum iterations")
    p.add_argument("--train-updates", type=int, default=5000,
                   help="PPO updates per iteration")
    p.add_argument("--force-grid", default="10,20,30,50,70,100,150,200",
                   help="Initial force grid (N)")
    p.add_argument("--duration-grid", default="1,2,3,4,6,8",
                   help="Duration grid (action steps)")
    p.add_argument("--episodes-per-cell", type=int, default=20,
                   help="Episodes per grid cell for state bank generation")
    p.add_argument("--gen-workers", type=int, default=96,
                   help="Parallel workers for state bank generation")
    p.add_argument("--rollout-workers", type=int, default=96,
                   help="Parallel workers for PPO training rollouts")
    p.add_argument("--max-steps", type=int, default=600,
                   help="Max action steps per episode")
    p.add_argument("--tolerance", type=int, default=6,
                   help="Imbalance tolerance steps")
    p.add_argument("--agent-id", default="robot_a")
    p.add_argument("--no-improve-patience", type=int, default=2,
                   help="Stop after N consecutive iterations without boundary improvement")
    p.add_argument("--target-boundary-force", type=float, default=300.0,
                   help="Stop when boundary_force exceeds this value")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test mode (small grid, 2 train updates)")
    p.add_argument("--no-adapt-grid", action="store_true",
                   help="Disable force grid adaptation (keep initial grid)")
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    force_grid = [float(x) for x in args.force_grid.split(",")]
    duration_grid = [int(x) for x in args.duration_grid.split(",")]

    iter_log: List[Dict[str, Any]] = []
    log_path = output_dir / "iter_log.json"

    # Load existing log if resuming
    if log_path.exists():
        with open(log_path) as f:
            iter_log = json.load(f)
        print(f"Resuming from iter_log with {len(iter_log)} previous iterations")

    current_policy = str(Path(args.base_policy).resolve())
    no_improve_count = 0
    best_boundary_force = 0.0

    for it in range(args.max_iters):
        iter_num = len(iter_log)
        print(f"\n{'='*60}")
        print(f"  Iteration {iter_num}")
        print(f"{'='*60}")
        print(f"  Policy: {current_policy}")
        print(f"  Force grid: {force_grid}")

        iter_dir = output_dir / f"gen{iter_num}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Generate state bank ---
        print(f"\n  [Step 1] Generating state bank...")
        full_bank = str(iter_dir / "full_state_bank.npz")
        t0 = time.perf_counter()
        run_generate_state_bank(
            policy_path=current_policy,
            output_path=full_bank,
            force_grid=force_grid,
            duration_grid=duration_grid,
            episodes_per_cell=args.episodes_per_cell,
            workers=args.gen_workers,
            max_steps=args.max_steps,
            tolerance=args.tolerance,
            agent_id=args.agent_id,
            seed=args.seed + iter_num * 1000,
        )
        gen_time = time.perf_counter() - t0
        print(f"  [gen] Time: {gen_time:.1f}s")

        # --- Step 2: Analyze boundary ---
        print(f"\n  [Step 2] Analyzing boundary...")
        analysis = analyze_state_bank(full_bank)

        print(f"  [analysis] States: {analysis['n']}")
        print(f"  [analysis] Overall survival: {analysis['overall_survival_rate']:.3f}")
        print(f"  [analysis] Boundary force (50% survival): {analysis['boundary_force']:.1f}N")

        # Print per-cell summary
        print(f"  {'force':>7} {'dur':>4} {'n':>5} {'surv':>5} {'rate':>6} {'mean_len':>9}")
        print(f"  {'-'*42}")
        for cell in analysis["cells"]:
            print(f"  {cell['force']:>7.0f} {cell['duration']:>4d} "
                  f"{cell['n']:>5d} {cell['survived']:>5d} "
                  f"{cell['rate']:>6.3f} {cell['mean_ep_len']:>9.1f}")

        # Save analysis CSV
        csv_path = iter_dir / "boundary_analysis.csv"
        with open(csv_path, "w") as f:
            f.write("force,duration,n,survived,rate,mean_ep_len\n")
            for cell in analysis["cells"]:
                f.write(f"{cell['force']:.0f},{cell['duration']},{cell['n']},"
                        f"{cell['survived']},{cell['rate']:.4f},{cell['mean_ep_len']:.1f}\n")

        # --- Step 3: Compute impulse range for training ---
        boundary_force = analysis["boundary_force"]
        # Train with forces from 0.5x to 1.5x boundary force
        train_force_min = max(5.0, boundary_force * 0.5)
        train_force_max = boundary_force * 1.5
        # Duration range: use full grid range
        train_dur_min = min(duration_grid)
        train_dur_max = max(duration_grid)

        print(f"\n  [Step 3] Setting impulse range for training:")
        print(f"  [impulse] force: [{train_force_min:.0f}, {train_force_max:.0f}] N "
              f"(boundary_force={boundary_force:.1f}N)")
        print(f"  [impulse] duration: [{train_dur_min}, {train_dur_max}] steps")

        # --- Step 4: Train PPO with online impulse perturbation ---
        print(f"\n  [Step 4] Training PPO (online impulse perturbation)...")
        train_dir = str(iter_dir / "train")
        policy_bp_path = str(Path(current_policy) / "policy_blueprint.yaml")
        t0 = time.perf_counter()
        new_policy = run_train(
            base_policy_path=current_policy,
            policy_blueprint_path=policy_bp_path,
            force_min=train_force_min,
            force_max=train_force_max,
            dur_min=train_dur_min,
            dur_max=train_dur_max,
            run_dir=train_dir,
            rollout_workers=args.rollout_workers,
            train_updates=args.train_updates,
            smoke=args.smoke,
        )
        train_time = time.perf_counter() - t0
        print(f"  [train] Time: {train_time:.1f}s")
        print(f"  [train] New policy: {new_policy}")

        # --- Step 5: Evaluate improvement ---
        improved = boundary_force > best_boundary_force + 1.0  # 1N threshold
        if improved:
            best_boundary_force = boundary_force
            no_improve_count = 0
        else:
            no_improve_count += 1

        # --- Log ---
        entry = {
            "iter": iter_num,
            "policy_path": current_policy,
            "new_policy_path": new_policy,
            "state_bank_path": full_bank,
            "n_states": analysis["n"],
            "overall_survival_rate": analysis["overall_survival_rate"],
            "boundary_force": boundary_force,
            "best_boundary_force": best_boundary_force,
            "improved": improved,
            "force_grid": force_grid,
            "train_force_range": [train_force_min, train_force_max],
            "train_duration_range": [train_dur_min, train_dur_max],
            "gen_time_s": gen_time,
            "train_time_s": train_time,
        }
        iter_log.append(entry)
        with open(log_path, "w") as f:
            json.dump(iter_log, f, indent=2)

        print(f"\n  [summary] iter={iter_num} boundary_force={boundary_force:.1f}N "
              f"best={best_boundary_force:.1f}N improved={improved}")

        # --- Check stopping conditions ---
        if boundary_force >= args.target_boundary_force:
            print(f"\n  [stop] Boundary force {boundary_force:.1f}N >= target "
                  f"{args.target_boundary_force}N")
            break

        if no_improve_count >= args.no_improve_patience:
            print(f"\n  [stop] No improvement for {no_improve_count} iterations")
            break

        # --- Adapt grid for next iteration ---
        if not args.no_adapt_grid:
            all_survived = analysis["overall_survival_rate"] >= 0.95
            all_fell = analysis["overall_survival_rate"] <= 0.05
            force_grid = adapt_force_grid(
                prev_boundary_force=boundary_force,
                prev_grid=force_grid,
                all_survived=all_survived,
                all_fell=all_fell,
            )

        # Update policy for next iteration
        current_policy = new_policy

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"  Iterative Training Complete")
    print(f"{'='*60}")
    print(f"  Total iterations: {len(iter_log)}")
    print(f"  Best boundary force: {best_boundary_force:.1f}N")
    print(f"  Final policy: {current_policy}")
    print(f"  Log: {log_path}")

    for entry in iter_log:
        status = "✓" if entry["improved"] else " "
        fr = entry.get("train_force_range", [0, 0])
        print(f"  {status} iter={entry['iter']} "
              f"bf={entry['boundary_force']:>6.1f}N "
              f"surv={entry['overall_survival_rate']:.3f} "
              f"train_force=[{fr[0]:.0f},{fr[1]:.0f}]")


if __name__ == "__main__":
    main()
