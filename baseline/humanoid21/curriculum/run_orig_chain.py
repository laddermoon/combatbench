"""Chain-run all 9 original-code standup reproduction stages sequentially.

Each stage runs for a target number of updates, then the latest checkpoint
is passed to the next stage via --resume-from.

Usage::

    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_orig_chain.py
    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_orig_chain.py --start-stage 5
    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_orig_chain.py --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment


# Stage definitions: (experiment_name, target_total_updates)
# Targets derived from STANDUP_V2_TRAINING_HISTORY.md
STAGES = [
    ("standup_orig_s1", 500),    # r2: initial training
    ("standup_orig_s2", 1000),   # r3: amplified reward
    ("standup_orig_s3", 1500),   # r4: disabled entropy
    ("standup_orig_s4", 2000),   # r5: increased LR, h_reward=5
    ("standup_orig_s5", 2500),   # r6: removed height, time penalty
    ("standup_orig_s6", 4000),   # r7: V2 gapped potential
    ("standup_orig_s7", 8000),   # r8: pot=10 breakthrough
    ("standup_orig_s8", 12000),  # r9/r10: velocity-gated, sustainable
    ("standup_orig_s9", 16000),  # r14: wall-aware, final
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

    run_name = f"orig_{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
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
    parser = argparse.ArgumentParser(description="Chain-run original-code standup reproduction.")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Start from this stage (1-indexed).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (2 updates per stage).")
    parser.add_argument("--stop-stage", type=int, default=None,
                        help="Stop after this stage (1-indexed).")
    args = parser.parse_args()

    start_idx = args.start_stage - 1
    stop_idx = args.stop_stage or len(STAGES)

    if start_idx > 0 and not args.resume_from:
        prev_name = STAGES[start_idx - 1][0]
        runs_dir = Path(__file__).resolve().parent.parent / "runs"
        prev_runs = sorted(runs_dir.glob(f"orig_{prev_name}_*"))
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
