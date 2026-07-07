"""Chain-run all 9 standup reproduction stages sequentially.

Each stage runs for a target number of updates (derived from the original
training history), then the latest checkpoint is passed to the next stage
via --resume-from.

Usage::

    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_repro_chain.py
    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_repro_chain.py --start-stage 5
    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_repro_chain.py --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment


# Stage definitions: (experiment_name, target_total_updates)
# Targets derived from STANDUP_V2_TRAINING_HISTORY.md
# Linear chain: each stage resumes from the previous stage's last checkpoint.
STAGES = [
    ("standup_repro_s1", 190),    # r2: 190 updates from scratch
    ("standup_repro_s2", 330),    # r3: +140 updates
    ("standup_repro_s3", 450),    # r4: +120 updates
    ("standup_repro_s4", 545),    # r5: +95 updates (was abandoned in original, but we keep it)
    ("standup_repro_s5", 645),    # r7: +100 updates (V2 potential switch)
    ("standup_repro_s6", 2780),   # r8: +2135 updates (breakthrough stage)
    ("standup_repro_s7", 3470),   # r9: +690 updates
    ("standup_repro_s8", 4145),   # r10: +675 updates
    ("standup_repro_s9", 4725),   # r14: +580 updates (final)
]


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest checkpoint in a run directory."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_u*.pt"))
    if not ckpts:
        return None
    return ckpts[-1]


def find_best_checkpoint(run_dir: Path) -> Path | None:
    """Find the best checkpoint (if marked) or fall back to latest."""
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        return best
    return find_latest_checkpoint(run_dir)


def run_stage(
    stage_idx: int,
    experiment_name: str,
    target_updates: int,
    resume_from: Path | None,
    smoke: bool = False,
) -> Path:
    """Run a single training stage and return the run directory."""
    from baseline.framework.ppo_loop import train_ppo

    experiment = get_experiment(experiment_name)

    if smoke:
        experiment.max_updates = 2
        experiment.episodes_per_update = 8
        experiment.eval_episodes = 4
        experiment.eval_interval = 1
        experiment.rollout_workers = 2
        experiment.minibatch_size = 64
    else:
        experiment.max_updates = target_updates

    run_name = f"repro_{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    experiment.save_run_config(run_dir, smoke=smoke, algo="ppo")
    print(f"\n{'='*70}", flush=True)
    print(f"[chain] Stage {stage_idx+1}/{len(STAGES)}: {experiment_name}", flush=True)
    print(f"[chain] target_updates={target_updates}, run_dir={run_dir}", flush=True)
    if resume_from:
        print(f"[chain] resume_from={resume_from}", flush=True)
    print(f"{'='*70}\n", flush=True)

    train_ppo(
        experiment,
        run_dir=run_dir,
        resume_from=resume_from,
        use_confidence=True,
    )

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Chain-run standup reproduction stages.")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Start from this stage (1-indexed). Requires --resume-from for stages > 1.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to resume from (required if start-stage > 1).")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (2 updates per stage).")
    parser.add_argument("--stop-stage", type=int, default=None,
                        help="Stop after this stage (1-indexed). Default: run all.")
    args = parser.parse_args()

    start_idx = args.start_stage - 1
    stop_idx = args.stop_stage or len(STAGES)

    if start_idx > 0 and not args.resume_from:
        # Try to find the latest checkpoint from the previous stage's run
        prev_name = STAGES[start_idx - 1][0]
        runs_dir = Path(__file__).resolve().parent.parent / "runs"
        prev_runs = sorted(runs_dir.glob(f"repro_{prev_name}_*"))
        if prev_runs:
            ckpt = find_best_checkpoint(prev_runs[-1])
            if ckpt:
                args.resume_from = str(ckpt)
                print(f"[chain] Auto-found checkpoint from previous stage: {ckpt}", flush=True)

    if start_idx > 0 and not args.resume_from:
        raise SystemExit(
            f"Error: --resume-from required when starting from stage {args.start_stage}."
        )

    resume_from = Path(args.resume_from).resolve() if args.resume_from else None
    total_start = time.perf_counter()

    for i in range(start_idx, min(stop_idx, len(STAGES))):
        exp_name, target = STAGES[i]
        run_dir = run_stage(i, exp_name, target, resume_from, smoke=args.smoke)

        ckpt = find_best_checkpoint(run_dir)
        if ckpt is None:
            print(f"[chain] ERROR: No checkpoint found in {run_dir}, stopping.", flush=True)
            break

        resume_from = ckpt
        print(f"\n[chain] Stage {i+1} complete. Checkpoint: {ckpt}", flush=True)

    elapsed = time.perf_counter() - total_start
    print(f"\n[chain] All stages complete. Total time: {elapsed/3600:.1f}h", flush=True)
    if resume_from:
        print(f"[chain] Final checkpoint: {resume_from}", flush=True)


if __name__ == "__main__":
    main()
