"""Subprocess driver for the SAC resume-equivalence test.

Usage:
    python _sac_resume_driver.py --run-dir DIR --max-env-steps N \
        [--resume-from CKPT_DIR]

Shrinks ``sac_balance`` (real MuJoCo env) via instance attributes and
calls ``train_sac``.  MUST run as its own process: resume equivalence
is a process-boundary property — an in-process rerun would inherit
module state a real resume cannot see.

The run's ``run_terminal.json`` (written by every exit path since
R2-W10) tells the caller where the terminal checkpoint lives.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-env-steps", type=int, required=True)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()

    from baseline.experiments_sac import get_sac_experiment
    from baseline.framework.sac.loop import train_sac

    exp = get_sac_experiment("sac_balance")

    # Shrink to a few short rounds: 2 episodes/round, tiny replay and
    # update budget, no evals/videos (round-1 + terminal checkpoints
    # already give the resume boundary).
    exp.episodes_per_update = 2
    exp.max_env_steps = int(args.max_env_steps)
    exp.eval_interval = 10 ** 12
    exp.eval_episodes = 1
    exp.video_eval_interval = 0
    exp.rollout_workers = 1
    exp.random_start_transitions = 0

    sp = exp.sac_params()
    import dataclasses

    sp = dataclasses.replace(
        sp,
        replay_buffer_size=512,
        batch_size=8,
        warmup_steps=4,
        utd_ratio=0.5,
        max_grad_steps_per_round=4,
    )
    exp.sac_params = lambda: sp  # type: ignore

    train_sac(
        exp,
        run_dir=Path(args.run_dir),
        resume_from=args.resume_from,
        checkpoint_keep_last=0,  # keep every checkpoint for the test
    )


if __name__ == "__main__":
    main()
