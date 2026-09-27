#!/usr/bin/env python3
"""sweep_floor — 8 truncnorm-family policies × 3 seeds, ef=0, 600 updates,
with the uncertainty floor ENABLED at per-metric-family values.

Floor calibration (see RESULTS_truncnorm_sweeps.md §U curves): the U
metric differs by family — single-component U starts ~0.46 and decays
past 0.40 around u60-75; MoG U starts ~0.65 and decays past 0.60 around
u60-70.  floor = 0.4 (single) / 0.6 (MoG) makes the floor start pulling
at ~u60-90 for every cell — isolating "floor on vs off" from the
metric-scale asymmetry.

Scheduling: all 24 runs at once, rollout_workers=8 (24x8 = 192 workers;
measured optimum from bench_throughput: 89.3 eps/s vs 74.4 for 2x96).

Semantics
---------
- A run is DONE iff the child exits 0 AND its train.log's last
  ``__RAW_STATS__`` update >= 600.  Anything else is FAILED — no
  retry/resume (failures are signal; triage after the sweep).
- Progress: ``baseline/runs/_sweep_floor/status.json`` + ``orchestrator.log``.

Launch detached:  setsid python3 baseline/run_sweep_floor.py
Dry run:          python3 baseline/run_sweep_floor.py --dry-run
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
SWEEP_DIR = RUNS_DIR / "_sweep_floor"
STATUS_FILE = SWEEP_DIR / "status.json"
ORCH_LOG = SWEEP_DIR / "orchestrator.log"

MAX_UPDATES = 600
ROLLOUT_WORKERS = 8
MAX_CONCURRENT = 24
POLL_SEC = 120
GPUS = [str(i) for i in range(8)]

FLOOR_SINGLE = 0.4   # single-component U: onset ~u60-75
FLOOR_MOG = 0.6      # MoG marginal U: onset ~u60-70

# (experiment, is_mog) — MoG cells use the marginal-Rényi U metric.
EXPERIMENTS = [
    ("standup_floor04", False),
    ("standup_floor04_boundedstd", False),
    ("standup_floor04_statesig", False),
    ("standup_floor04_boundedstd_statesig", False),
    ("standup_floor04_mixture", True),
    ("standup_floor04_mixture_shared", True),
    ("standup_floor04_mixture_shared_boundedstd", True),
    ("standup_floor04_mixture_boundedstd_statesig", True),
]
SEEDS = [42, 43, 44]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a") as f:
        f.write(line + "\n")


def build_queue():
    return deque((exp, mog, seed) for seed in SEEDS for exp, mog in EXPERIMENTS)


def run_name_for(exp: str, seed: int) -> str:
    return f"sweep_floor_{exp}_s{seed}"


def command_for(exp: str, mog: bool, seed: int):
    floor = FLOOR_MOG if mog else FLOOR_SINGLE
    return [
        sys.executable, "baseline/framework/train.py",
        "--experiment", exp, "--algo", "ppo",
        "--seed", str(seed),
        "--set", f"uncertainty_floor={floor}",
        "--set", f"max_updates={MAX_UPDATES}",
        "--set", f"rollout_workers={ROLLOUT_WORKERS}",
        "--run-name", run_name_for(exp, seed),
    ]


def last_update(run_dir: Path) -> int:
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
        "sweep": "floor, 8 policies x 3 seeds, ef=0, 600 updates, "
                 f"floor={FLOOR_SINGLE}(single)/{FLOOR_MOG}(MoG), "
                 "24-concurrent x 8 workers",
        "remaining_in_queue": len(queue),
        "running": [
            {k: v for k, v in r.items() if k != "proc" and k != "console"}
            for r in running.values()
        ],
        "done": done,
        "failed": failed,
    }, indent=1))


def launch(exp: str, mog: bool, seed: int, gpu: str):
    run_name = run_name_for(exp, seed)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    console = open(SWEEP_DIR / f"console_{run_name}.log", "wb")
    proc = subprocess.Popen(
        command_for(exp, mog, seed),
        cwd=str(REPO),
        env=env,
        stdout=console,
        stderr=subprocess.STDOUT,
        start_new_session=True,
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
        for exp, mog, seed in queue:
            print(" ".join(command_for(exp, mog, seed)))
        print(f"\n{len(queue)} runs total")
        return 0

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    running = {}
    done, failed = [], []
    log(f"sweep start: {len(queue)} runs, floor single={FLOOR_SINGLE} "
        f"mog={FLOOR_MOG}, {MAX_CONCURRENT} concurrent x {ROLLOUT_WORKERS}w")

    try:
        while queue or running:
            for slot in range(MAX_CONCURRENT):
                if slot not in running and queue:
                    exp, mog, seed = queue.popleft()
                    rec = launch(exp, mog, seed, GPUS[slot % len(GPUS)])
                    running[slot] = rec
                    log(f"launch slot{slot} gpu{GPUS[slot % len(GPUS)]} "
                        f"{rec['run_name']} pid={rec['pid']}")
                    write_status(running, done, failed, queue)
                    time.sleep(10)

            time.sleep(POLL_SEC)

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
        log("orchestrator interrupted — children keep running")
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
