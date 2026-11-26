"""Compare DynaBPTT core runs — aligned eval/J/gate tables.

Usage:
    python3 -m baseline.framework.dynabptt.core.compare run1 run2 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(run: str) -> list[dict]:
    p = RUNS / run / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p)
            if '"update"' in l and json.loads(l).get("type") == "update"]


def main() -> None:
    runs = sys.argv[1:] or sorted(
        d.name for d in RUNS.iterdir()
        if d.is_dir() and not d.name.startswith("smoke"))
    data = {r: load(r) for r in runs}
    print(f"{'run':<34}{'upd':>5} {'eval%':>6} {'J':>7} {'v':>6} "
          f"{'gn':>7} {'gate':>6} {'phi10':>6} {'len':>5}")
    for r in runs:
        ups = data[r]
        if not ups:
            print(f"{r:<34}  no data")
            continue
        last = ups[-1]
        ev = [x for x in ups if "eval_surv" in x]
        evs = ev[-1]["eval_surv"] if ev else float("nan")
        print(f"{r:<34}{len(ups):>5} {evs:>6.3f} {last.get('J', 0):>7.2f} "
              f"{last.get('v_mean', 0):>6.1f} {last.get('gnorm', 0):>7.4f} "
              f"{last.get('gate_phi_mae', 9):>6.3f} "
              f"{last.get('ss_phi_p10', 0):>6.2f} "
              f"{last.get('branch_len', 0):>5.1f}")
    print()
    # eval trajectories aligned
    for r in runs:
        ev = [(x["u"], x["eval_surv"]) for x in data[r] if "eval_surv" in x]
        if ev:
            traj = " ".join(f"{u}:{v:.2f}" for u, v in ev)
            print(f"{r:<34} eval: {traj}")


if __name__ == "__main__":
    main()
