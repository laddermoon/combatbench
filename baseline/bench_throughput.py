#!/usr/bin/env python3
"""bench_throughput — find the max-throughput (concurrency, rollout_workers)
configuration for PPO training on this machine.

For each grid cell (C concurrent runs x W rollout_workers each), launches
C foreground ``train.py`` runs of ``standup_floor04`` for a short window
(BENCH_UPDATES updates) and measures aggregate episodes/sec over wall
clock.  ``timing.total`` per update from RAW_STATS is logged alongside
as a sanity check.

Runs cells sequentially — never overlaps benchmark cells.  GPU is
round-robined across the C runs of a cell.

Usage:  python3 baseline/bench_throughput.py [--dry-run]
Output: baseline/runs/_bench_tp/results.json + printed table.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "baseline" / "runs"
OUT_DIR = RUNS_DIR / "_bench_tp"
RESULTS = OUT_DIR / "results.json"

EXPERIMENT = "standup_floor04"
EPISODES_PER_UPDATE = 512
BENCH_UPDATES = 8
POLL_SEC = 20

# (concurrency C, rollout_workers W) — covers total-worker sweep and
# different splits at ~192 total workers, plus oversubscription probes.
GRID = [
    (1, 96),
    (2, 96),   # current config
    (3, 64),
    (4, 48),
    (6, 32),
    (8, 24),
    (4, 96),   # oversubscribed: 384 > 192 cores
    (2, 48),   # underutilized probe
]


def run_names(cell_idx: int, c: int, w: int):
    return [f"bench_tp_c{c}_w{w}_g{cell_idx}_r{i}" for i in range(c)]


def cmd_for(run_name: str, workers: int, seed: int):
    return [
        sys.executable, "baseline/framework/train.py",
        "--experiment", EXPERIMENT, "--algo", "ppo",
        "--seed", str(seed),
        "--set", "uncertainty_floor=0",
        "--set", f"max_updates={BENCH_UPDATES}",
        "--set", f"rollout_workers={workers}",
        "--run-name", run_name,
    ]


def mean_update_time(run_dir: Path):
    """Mean timing.total over all __RAW_STATS__ (excluding u1 warmup)."""
    ts = []
    try:
        for line in open(run_dir / "train.log"):
            if line.startswith("__RAW_STATS__"):
                j = json.loads(line[len("__RAW_STATS__"):])
                if j["update"] > 1 and "timing" in j:
                    ts.append(j["timing"]["total"])
    except (OSError, json.JSONDecodeError):
        pass
    return sum(ts) / len(ts) if ts else None


def run_cell(cell_idx: int, c: int, w: int):
    names = run_names(cell_idx, c, w)
    procs, consoles = [], []
    t0 = time.time()
    for i, rn in enumerate(names):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        env["CUDA_VISIBLE_DEVICES"] = str(i % 8)
        f = open(OUT_DIR / f"console_{rn}.log", "wb")
        consoles.append(f)
        procs.append(subprocess.Popen(
            cmd_for(rn, w, seed=100 + cell_idx * 10 + i),
            cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
        ))
        time.sleep(5)  # gentle stagger to avoid env-build stampede
    while any(p.poll() is None for p in procs):
        time.sleep(POLL_SEC)
    wall = time.time() - t0
    for f in consoles:
        f.close()

    rcs = [p.returncode for p in procs]
    updates = [sum(1 for l in open(RUNS_DIR / rn / "train.log")
                   if l.startswith("__RAW_STATS__"))
               if (RUNS_DIR / rn / "train.log").exists() else 0
               for rn in names]
    per_run_t = [mean_update_time(RUNS_DIR / rn) for rn in names]
    total_eps = sum(u * EPISODES_PER_UPDATE for u in updates)
    return {
        "cell": f"c{c}_w{w}", "concurrency": c, "workers": w,
        "total_workers": c * w, "wall_s": round(wall, 1),
        "updates_per_run": updates, "returncodes": rcs,
        "mean_update_s": [round(t, 2) if t else None for t in per_run_t],
        "eps_per_s": round(total_eps / wall, 1),
        "episodes": total_eps,
    }


def main() -> int:
    if "--dry-run" in sys.argv:
        for c, w in GRID:
            print(f"c={c} w={w} total_workers={c*w} "
                  f"est_wall={(EPISODES_PER_UPDATE*2.75/max(w,1)+3)*BENCH_UPDATES:.0f}s")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, (c, w) in enumerate(GRID):
        print(f"[cell {idx+1}/{len(GRID)}] c={c} w={w} "
              f"(total_workers={c*w})", flush=True)
        r = run_cell(idx, c, w)
        results.append(r)
        RESULTS.write_text(json.dumps(results, indent=1))
        print(f"  wall={r['wall_s']}s updates={r['updates_per_run']} "
              f"mean_update={r['mean_update_s']} eps/s={r['eps_per_s']}",
              flush=True)

    print("\n=== THROUGHPUT RANKING ===")
    for r in sorted(results, key=lambda x: -x["eps_per_s"]):
        print(f"{r['cell']:10s} workers={r['total_workers']:>3} "
              f"eps/s={r['eps_per_s']:>7}  mean_update={r['mean_update_s']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
