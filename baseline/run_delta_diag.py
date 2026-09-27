#!/usr/bin/env python3
"""run_delta_diag — staged policy-drift (Δ) diagnostics for the 8
truncnorm-family policies, on top of the sweep_uf0 (ef=0, floor=0) s42
runs.

For each (experiment, stage) pair: resume from the uf0 run's checkpoint
at ``stage-5``, train 5 updates with ``--dump-at stage``, then merge the
original run's ``policy_exports`` (symlinks, earlier updates only) so
``debug.py delta`` sees the full generation chain, then compute delta on
DELTA_EPISODES episodes with gens=10.

Stages chosen to cover the whole training arc (early drift → annealed).

Launch detached:  setsid python3 baseline/run_delta_diag.py
Dry run:          python3 baseline/run_delta_diag.py --dry-run
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
SWEEP_DIR = RUNS_DIR / "_delta_diag"
STATUS_FILE = SWEEP_DIR / "status.json"
ORCH_LOG = SWEEP_DIR / "orchestrator.log"

SRC_TAG = "sweep_uf0"          # source sweep: ef=0, floor=0
SEED = 42
STAGES = [10, 50, 100, 200, 300, 500]   # dump update (resume from stage-5)
RESUME_SPAN = 5                          # updates trained per stage
ROLLOUT_WORKERS = 6
MAX_CONCURRENT = 4                       # diag jobs on top of live sweep
DELTA_GENS = 10
DELTA_EPISODES = [0, 1, 2]
POLL_SEC = 60
GPUS = [str(i) for i in range(8)]

EXPERIMENTS = [
    "standup_floor04",
    "standup_floor04_boundedstd",
    "standup_floor04_statesig",
    "standup_floor04_boundedstd_statesig",
    "standup_floor04_mixture",
    "standup_floor04_mixture_shared",
    "standup_floor04_mixture_shared_boundedstd",
    "standup_floor04_mixture_boundedstd_statesig",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a") as f:
        f.write(line + "\n")


def src_run_dir(exp: str) -> Path:
    return RUNS_DIR / f"{SRC_TAG}_{exp}_s{SEED}"


def run_name_for(exp: str, stage: int) -> str:
    return f"delta_diag_{exp}_u{stage}"


def command_for(exp: str, stage: int):
    ckpt = (src_run_dir(exp) / "checkpoints"
            / f"checkpoint_u{stage - RESUME_SPAN:05d}.pt")
    return [
        sys.executable, "baseline/framework/train.py",
        "--experiment", exp, "--algo", "ppo",
        "--seed", str(SEED),
        "--resume-from", str(ckpt),
        "--dump-at", str(stage),
        "--set", "uncertainty_floor=0",
        "--set", f"max_updates={stage}",
        "--set", f"rollout_workers={ROLLOUT_WORKERS}",
        "--run-name", run_name_for(exp, stage),
    ]


def last_update(run_dir: Path) -> int:
    last = 0
    try:
        with open(run_dir / "train.log") as f:
            for line in f:
                if line.startswith("__RAW_STATS__"):
                    last = json.loads(line[len("__RAW_STATS__"):])["update"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return last


def link_source_exports(exp: str, stage: int) -> int:
    """Symlink the source run's policy_exports u<=stage-span into the
    diag run's exports dir so delta has the full generation chain.
    Only links names that don't already exist (never clobber new
    exports)."""
    src = src_run_dir(exp) / "policy_exports"
    dst = RUNS_DIR / run_name_for(exp, stage) / "policy_exports"
    n = 0
    for d in sorted(src.iterdir()):
        if not d.is_dir() or d.name.endswith("_eval"):
            continue
        u = int(d.name[1:])
        if u > stage - RESUME_SPAN:
            continue
        target = dst / d.name
        if not target.exists():
            target.symlink_to(d.resolve())
            n += 1
    return n


def compute_deltas(exp: str, stage: int) -> int:
    dump_dir = (RUNS_DIR / run_name_for(exp, stage)
                / "dumps" / f"u{stage:05d}")
    if not dump_dir.is_dir():
        return -1
    ok = 0
    for ep in DELTA_EPISODES:
        rc = subprocess.run(
            [sys.executable, "-B", "-m", "baseline.framework.ppo.debug",
             "delta", str(dump_dir), "--episode", str(ep),
             "--gens", str(DELTA_GENS)],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        ok += (rc == 0)
    return ok


def write_status(running, done, failed, queue) -> None:
    STATUS_FILE.write_text(json.dumps({
        "task": f"delta diag: {len(EXPERIMENTS)} policies x {len(STAGES)} "
                f"stages {STAGES}, seed={SEED}, src={SRC_TAG}",
        "remaining_in_queue": len(queue),
        "running": [
            {k: v for k, v in r.items() if k != "proc" and k != "console"}
            for r in running.values()
        ],
        "done": done,
        "failed": failed,
    }, indent=1))


def launch(exp: str, stage: int, gpu: str):
    run_name = run_name_for(exp, stage)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    console = open(SWEEP_DIR / f"console_{run_name}.log", "wb")
    proc = subprocess.Popen(
        command_for(exp, stage),
        cwd=str(REPO),
        env=env,
        stdout=console,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "exp": exp, "stage": stage, "run_name": run_name,
        "run_dir": str(RUNS_DIR / run_name),
        "pid": proc.pid, "started": time.strftime("%H:%M:%S"),
        "proc": proc, "console": console,
    }


def finish_run(rec) -> None:
    """Post-run: link exports + compute deltas (fast, pure local)."""
    n_link = link_source_exports(rec["exp"], rec["stage"])
    n_delta = compute_deltas(rec["exp"], rec["stage"])
    rec["n_linked_exports"] = n_link
    rec["n_delta_ok"] = n_delta
    log(f"post {rec['run_name']}: linked={n_link} "
        f"delta_ok={n_delta}/{len(DELTA_EPISODES)}")


def main() -> int:
    queue = deque((e, s) for e in EXPERIMENTS for s in STAGES)
    if "--dry-run" in sys.argv:
        for exp, stage in queue:
            print(" ".join(command_for(exp, stage)))
        print(f"\n{len(queue)} diag runs total")
        return 0

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    running = {}
    done, failed = [], []
    log(f"delta-diag start: {len(queue)} runs, {MAX_CONCURRENT} concurrent")

    try:
        while queue or running:
            for slot in range(MAX_CONCURRENT):
                if slot not in running and queue:
                    exp, stage = queue.popleft()
                    rec = launch(exp, stage, GPUS[slot % len(GPUS)])
                    running[slot] = rec
                    log(f"launch slot{slot} {rec['run_name']} "
                        f"pid={rec['pid']}")
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
                ok = (rc == 0 and u >= rec["stage"])
                if ok:
                    finish_run(rec)
                    rec_out["n_delta_ok"] = rec.get("n_delta_ok")
                    done.append(rec_out)
                else:
                    failed.append(rec_out)
                del running[slot]
                log(f"{'DONE' if ok else 'FAILED'} {rec['run_name']} "
                    f"rc={rc} last_update={u}")
                write_status(running, done, failed, queue)
    except KeyboardInterrupt:
        log("orchestrator interrupted — children keep running")
        write_status(running, done, failed, queue)
        return 2

    log(f"delta-diag finished: {len(done)} done, {len(failed)} failed")
    for r in failed:
        log(f"  FAILED: {r['run_name']} rc={r['returncode']} "
            f"u={r['last_update']}")
    write_status(running, done, failed, queue)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
