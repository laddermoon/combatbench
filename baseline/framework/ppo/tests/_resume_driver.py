"""Subprocess driver for the resume-equivalence test.

Usage:
    python _resume_driver.py --run-dir DIR --max-updates N \
        [--resume-from CKPT] [--dump-updates 1,3]

Shrinks the ``minimal`` experiment exactly like ``train.py --smoke``
does (dataclasses.replace on common_params/ppo_params) and calls
``train_ppo``.  This MUST run as its own process: resume equivalence
is a process-boundary property — an in-process rerun would inherit
module-level state a real resume cannot see.

``--dump-updates`` lists update indices that should carry a dump
request (the only trigger for the gradsig diagnostic).  For update 1
the sentinel is written before ``train_ppo`` starts; for later updates
an ``on_update`` wrapper writes it after the preceding update — the
loop polls the sentinel at the top of each update, so this is
deterministic, not timing-dependent.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-updates", type=int, required=True)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--dump-updates", default="")
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

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_updates = {
        int(x) for x in args.dump_updates.split(",") if x.strip()
    }
    if dump_updates:
        import re
        def _write_sentinel() -> None:
            (run_dir / "dump_request.json").write_text(json.dumps({
                "hypothesis": "resume-equivalence probe",
            }))

        # First update this process will run: 1 for a fresh run, or
        # ckpt_update+1 when resuming (parsed from the filename).
        first_update = 1
        if args.resume_from:
            m = re.search(r"checkpoint_u(\d+)\.pt$", args.resume_from)
            if m:
                first_update = int(m.group(1)) + 1
        if first_update in dump_updates:
            _write_sentinel()
        orig_on_update = exp.on_update

        def _on_update(stats, update):
            out = orig_on_update(stats, update)
            if update + 1 in dump_updates:
                _write_sentinel()
            return out

        exp.on_update = _on_update  # type: ignore

    train_ppo(
        experiment=exp,
        run_dir=run_dir,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )


if __name__ == "__main__":
    main()
