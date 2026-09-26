#!/usr/bin/env python3
"""sweep_ef05 — 8 truncnorm-family policies × 3 seeds, floor=0, ef=0.5,
600 updates.  Companion sweep to run_sweep_ufloor0.py (ef=0 baseline).

Throughput-informed scheduling: bench_throughput showed that spreading
192 workers across many small jobs beats 2 x 96 (89.3 vs 74.4 eps/s).
So ALL 24 runs launch at once with rollout_workers=8 (24x8 = 192
workers, one full wave; GPUs round-robined across 8 devices).

Semantics
---------
- A run is DONE iff the child exits 0 AND its train.log's last
  ``__RAW_STATS__`` update >= 600.  Anything else is FAILED — no
  retry/resume (failures are signal; triage after the sweep).
- Progress: ``baseline/runs/_sweep_ef05/status.json`` (rewritten
  every poll) + ``orchestrator.log``.

Launch detached:  setsid python3 baseline/run_sweep_ef05.py
Dry run:          python3 baseline/run_sweep_ef05.py --dry-run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # things/combatbench
RUNS_DIR = REPO / "baseline" / "runs"
SWEEP_DIR = RUNS_DIR / "_sweep_ef05"
STATUS_FILE = SWEEP_DIR / "status.json"
ORCH_LOG = SWEEP_DIR / "orchestrator.log"

MAX_UPDATES = 600
EXPLORE_FACTOR = 0.5
ROLLOUT_WORKERS = 8
MAX_CONCURRENT = 24          # all 24 runs at once: 24 x 8 = 192 workers
POLL_SEC = 120
GPUS = [str(i) for i in range(8)]  # slot index → CUDA_VISIBLE_DEVICES

EXPERIMENTS = [
    "standup_floor04",                          # (no,no,no)  TruncatedNormal
    "standup_floor04_boundedstd",               # (no,no,yes) BoundedStd
    "standup_floor04_statesig",                 # (no,yes,no) StateTruncatedNormal
    "standup_floor04_boundedstd_statesig",      # (no,yes,yes) StateBoundedStd
    "standup_floor04_mixture",                  # (yes,yes,no) Mixture
    "standup_floor04_mixture_shared",           # (yes,no,no) SharedMixture
    "standup_floor04_mixture_shared_boundedstd",  # (yes,no,yes) SharedMixtureBoundedStd
    "standup_floor04_mixture_boundedstd_statesig",  # (yes,yes,yes) StateMixtureBoundedStd
]
SEEDS = [42, 43, 44]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a") as f:
        f.write(line + "\n")


def build_queue():
    """seed-major: for each seed, all 8 policy cells."""
    return deque(
        (exp, seed) for seed in SEEDS for exp in EXPERIMENTS
    )


def run_name_for(exp: str, seed: int) -> str:
    return f"sweep_ef05_{exp}_s{seed}"


def command_for(exp: str, seed: int):
    return [
        sys.executable, "baseline/framework/train.py",
        "--experiment", exp, "--algo", "ppo",
        "--seed", str(seed),
        "--set", "uncertainty_floor=0",
        "--set", f"explore_factor={EXPLORE_FACTOR}",
        "--set", f"max_updates={MAX_UPDATES}",
        "--set", f"rollout_workers={ROLLOUT_WORKERS}",
        "--run-name", run_name_for(exp, seed),
    ]


def last_update(run_dir: Path) -> int:
    """Last __RAW_STATS__ update index in train.log (0 if none)."""
    log_path = run_dir / "train.log"
    last = 0
    try:
        with open(log_path) as f:
            for line in f:
                if line.startswith("__RAW_STATS__"):
                    last = json.loads(line[len("__RAW_STATS__"):])["update"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return last


def write_status(running, done, failed, queue) -> None:
    STATUS_FILE.write_text(json.dumps({
        "sweep": "ef05, 8 policies x 3 seeds, floor=0, ef=0.5, "
                 "600 updates, 24-concurrent x 8 workers",
        "remaining_in_queue": len(queue),
        "running": [
            {k: v for k, v in r.items() if k != "proc" and k != "console"}
            for r in running.values()
        ],
        "done": done,
        "failed": failed,
    }, indent=1))


def launch(exp: str, seed: int, gpu: str):
    run_name = run_name_for(exp, seed)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    console = open(SWEEP_DIR / f"console_{run_name}.log", "wb")
    proc = subprocess.Popen(
        command_for(exp, seed),
        cwd=str(REPO),
        env=env,
        stdout=console,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from our session/ctl-c
    )
    return {
        "exp": exp, "seed": seed, "run_name": run_name,
        "run_dir": str(RUNS_DIR / run_name),
        "pid": proc.pid, "started": time.strftime("%H:%M:%S"),
        "proc": proc, "console": console,
    }


def main() -> int:
    queue = build_queue()
    if "--dry-run" in sys.argv:
        for exp, seed in queue:
            print(" ".join(command_for(exp, seed)))
        print(f"\n{len(queue)} runs total")
        return 0

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    running = {}   # slot -> record
    done, failed = [], []
    log(f"sweep start: {len(queue)} runs, {MAX_CONCURRENT} concurrent "
        f"x {ROLLOUT_WORKERS} workers, ef={EXPLORE_FACTOR}")

    try:
        while queue or running:
            # Fill free slots.
            for slot in range(MAX_CONCURRENT):
                if slot not in running and queue:
                    exp, seed = queue.popleft()
                    rec = launch(exp, seed, GPUS[slot % len(GPUS)])
                    running[slot] = rec
                    log(f"launch slot{slot} gpu{GPUS[slot % len(GPUS)]} "
                        f"{rec['run_name']} pid={rec['pid']}")
                    write_status(running, done, failed, queue)
                    time.sleep(10)  # stagger startup

            time.sleep(POLL_SEC)

            # Reap finished runs.
            for slot in list(running):
                rec = running[slot]
                rc = rec["proc"].poll()
                if rc is None:
                    continue
                rec["console"].close()
                u = last_update(Path(rec["run_dir"]))
                rec_out = {k: v for k, v in rec.items()
                           if k not in ("proc", "console")}
                rec_out["ended"] = time.strftime("%H:%M:%S")
                rec_out["returncode"] = rc
                rec_out["last_update"] = u
                ok = (rc == 0 and u >= MAX_UPDATES)
                (done if ok else failed).append(rec_out)
                del running[slot]
                log(f"{'DONE' if ok else 'FAILED'} {rec['run_name']} "
                    f"rc={rc} last_update={u}")
                write_status(running, done, failed, queue)
    except KeyboardInterrupt:
        log("orchestrator interrupted — children keep running "
            "(start_new_session)")
        write_status(running, done, failed, queue)
        return 2

    log(f"sweep finished: {len(done)} done, {len(failed)} failed")
    for r in failed:
        log(f"  FAILED: {r['run_name']} rc={r['returncode']} "
            f"u={r['last_update']}")
    write_status(running, done, failed, queue)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
