"""Chain-run the 2-phase 4-stage standup training.

Phase A: exploration (entropy=1e-3, 1000 updates)
Phase B: precise execution (entropy=0.0, 4000 updates total)

Usage::

    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_4stage_chain.py
    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_4stage_chain.py --start-stage 2
    PYTHONPATH=. python3 baseline/humanoid21/curriculum/run_4stage_chain.py --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment


STAGES = [
    ("standup_4stage_a", 1000),    # Phase A: exploration
    ("standup_4stage_b", 4000),    # Phase B: precise (total 4000)
]


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_u*.pt"))
    if not ckpts:
        return None
    return ckpts[-1]


def find_last_checkpoint(run_dir: Path) -> Path | None:
    """Use the latest checkpoint (not best) so Phase B resumes from final state."""
    return find_latest_checkpoint(run_dir)


def run_stage(
    stage_idx: int,
    experiment_name: str,
    target_updates: int,
    resume_from: Path | None,
    smoke: bool = False,
) -> Path:
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

    run_name = f"4stage_{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    experiment.save_run_config(run_dir, smoke=smoke, algo="ppo")
    print(f"\n{'='*70}", flush=True)
    print(f"[chain] Phase {stage_idx+1}/{len(STAGES)}: {experiment_name}", flush=True)
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
    parser = argparse.ArgumentParser(description="Chain-run 4-stage standup training.")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Start from this phase (1-indexed).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (2 updates per phase).")
    parser.add_argument("--stop-stage", type=int, default=None,
                        help="Stop after this phase (1-indexed).")
    args = parser.parse_args()

    start_idx = args.start_stage - 1
    stop_idx = args.stop_stage or len(STAGES)

    if start_idx > 0 and not args.resume_from:
        prev_name = STAGES[start_idx - 1][0]
        runs_dir = Path(__file__).resolve().parent.parent / "runs"
        prev_runs = sorted(runs_dir.glob(f"4stage_{prev_name}_*"))
        if prev_runs:
            ckpt = find_last_checkpoint(prev_runs[-1])
            if ckpt:
                args.resume_from = str(ckpt)
                print(f"[chain] Auto-found checkpoint from previous phase: {ckpt}", flush=True)

    if start_idx > 0 and not args.resume_from:
        raise SystemExit(
            f"Error: --resume-from required when starting from phase {args.start_stage}."
        )

    resume_from = Path(args.resume_from).resolve() if args.resume_from else None
    total_start = time.perf_counter()

    for i in range(start_idx, min(stop_idx, len(STAGES))):
        exp_name, target = STAGES[i]
        run_dir = run_stage(i, exp_name, target, resume_from, smoke=args.smoke)

        ckpt = find_last_checkpoint(run_dir)
        if ckpt is None:
            print(f"[chain] ERROR: No checkpoint found in {run_dir}, stopping.", flush=True)
            break

        resume_from = ckpt
        print(f"\n[chain] Phase {i+1} complete. Checkpoint: {ckpt}", flush=True)

    elapsed = time.perf_counter() - total_start
    print(f"\n[chain] All phases complete. Total time: {elapsed/3600:.1f}h", flush=True)
    if resume_from:
        print(f"[chain] Final checkpoint: {resume_from}", flush=True)


if __name__ == "__main__":
    main()
