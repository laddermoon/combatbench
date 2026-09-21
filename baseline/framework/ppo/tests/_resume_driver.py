"""Subprocess driver for the resume-equivalence test.

Usage:
    python _resume_driver.py --run-dir DIR --max-updates N [--resume-from CKPT]

Shrinks the ``minimal`` experiment exactly like ``train.py --smoke``
does (dataclasses.replace on common_params/ppo_params) and calls
``train_ppo``.  This MUST run as its own process: resume equivalence
is a process-boundary property — an in-process rerun would inherit
module-level state a real resume cannot see.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-updates", type=int, required=True)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()

    from baseline.experiments_ppo import get_ppo_experiment
    from baseline.framework.ppo.loop import train_ppo

    exp = get_ppo_experiment("minimal")
    cp = exp.common_params()
    cp = dataclasses.replace(
        cp,
        max_updates=args.max_updates,
        episodes_per_update=8,
        eval_episodes=4,
        eval_interval=2,
        rollout_workers=2,
    )
    exp.common_params = lambda: cp  # type: ignore
    pp = exp.ppo_params()
    pp = dataclasses.replace(pp, update_epochs=2, minibatch_size=64)
    exp.ppo_params = lambda: pp  # type: ignore

    train_ppo(
        experiment=exp,
        run_dir=Path(args.run_dir),
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )


if __name__ == "__main__":
    main()
