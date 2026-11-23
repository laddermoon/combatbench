"""最速前进扫描（STEPACE 扩展）：sag × 切换门控网格。

变量：
    sag      固定步幅 {0.15,0.25,0.35,0.45,0.55}
    timing   land(落地换) / air(腾空即换) / ageK(令龄K帧换)
指标：task_speed（任务阶段速度，不含尾部保持）、dist、
    fall_frac、min_phi、d_mv、lat_drift。
用法：
    PYTHONPATH=. python3 baseline/humanoid21/v1/sprint_sweep.py \
        --policy stepcmd --episodes 2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step_accept import (_MAX_FRAMES, _load_policy, _resolve_policy,
                         _rollout, _V1, _ROBOTS)
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

_SAGS = [0.15, 0.25, 0.35, 0.45, 0.55]
_TIMINGS: List[Dict[str, Any]] = [
    {"gate": "land"},
    {"gate": "air"},
    {"gate": "age", "min_age": 3},
    {"gate": "age", "min_age": 5},
    {"gate": "age", "min_age": 7},
    {"gate": "age", "min_age": 9},
]


def _timing_name(t: Dict[str, Any]) -> str:
    if t["gate"] == "age":
        return f"age{t['min_age']}"
    return t["gate"]


def _sprint_metrics(d: Dict[str, Any], n_cmd: int) -> Dict[str, float]:
    """任务阶段指标：速度只算到最后一条迈步令结算。"""
    out = []
    for rid in _ROBOTS:
        evs = [e for e in d["events"][rid] if e["hold"] < 0.5]
        tr = d["track"][rid]
        if len(evs) < 2:
            continue
        end_t = evs[min(n_cmd, len(evs)) - 1]["t"]
        tr_end = [p for p in tr if p["t"] <= end_t]
        if len(tr_end) < 2:
            continue
        x0, y0, yaw0 = (tr_end[0]["x"], tr_end[0]["y"],
                        tr_end[0]["yaw"])
        dx = np.array([tr_end[-1]["x"] - x0,
                       tr_end[-1]["y"] - y0])
        fwd = np.array([math.cos(yaw0), math.sin(yaw0)])
        dt = max(end_t, 1) * 0.05
        m = {
            "task_speed": float(fwd @ dx) / dt,
            "dist": float(fwd @ dx),
            "lat_drift": float(np.array([-math.sin(yaw0),
                                         math.cos(yaw0)]) @ dx),
            "d_mv": float(np.mean([e["d_mv"] for e in evs])),
            "arr": float(np.mean([e["t_arrive"] >= 0 for e in evs])),
            "min_phi": float(min(p["phi"] for p in tr_end)),
            "fall_frac": float(np.mean(
                [p["phi"] < 0.3 for p in tr_end])),
            "cmd_dur": float(np.mean([e["age"] for e in evs])),
        }
        out.append(m)
    if not out:
        return {"task_speed": 0.0}
    return {k: round(float(np.mean([m[k] for m in out])), 3)
            for k in out[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--sags", default="")
    ap.add_argument("--timings", default="")
    ap.add_argument("--replan", action="store_true")
    args = ap.parse_args()

    pdir = _resolve_policy(args.policy)
    pol = _load_policy(pdir)
    env = ParameterizedEnvBlueprint.load(
        _V1 / "v2_stepace_env.yaml").materialize().build()

    sags = ([float(x) for x in args.sags.split(",")]
            if args.sags else _SAGS)
    timings = _TIMINGS
    if args.timings:
        timings = []
        for tok in args.timings.split(","):
            tok = tok.strip()
            if tok.startswith("age"):
                timings.append({"gate": "age",
                                "min_age": int(tok[3:])})
            else:
                timings.append({"gate": tok})

    print(f"policy={pdir}")
    header = (f"{'sag':>5} {'timing':>6} {'speed':>7} {'dist':>6} "
              f"{'lat':>6} {'d_mv':>5} {'arr':>5} {'dur':>4} "
              f"{'minphi':>6} {'fall':>5}")
    print(header)
    results = []
    for sag in sags:
        for tm in timings:
            tname = _timing_name(tm)
            rows = []
            for ep in range(args.episodes):
                d = _rollout(env, pol, "sprint",
                             "replan" if args.replan else "open",
                             args.seed + ep, _MAX_FRAMES,
                             gate=tm["gate"], sag=sag,
                             min_age=tm.get("min_age", 2))
                rows.append(_sprint_metrics(d, 16))
            agg = {k: round(float(np.mean([r.get(k, 0.0)
                                           for r in rows])), 3)
                   for k in rows[0]}
            agg["_sag"], agg["_timing"] = sag, tname
            results.append(agg)
            print(f"{sag:>5.2f} {tname:>6} {agg['task_speed']:>7.3f} "
                  f"{agg['dist']:>6.2f} {agg['lat_drift']:>6.2f} "
                  f"{agg['d_mv']:>5.2f} {agg['arr']:>5.2f} "
                  f"{agg['cmd_dur']:>4.1f} {agg['min_phi']:>6.2f} "
                  f"{agg['fall_frac']:>5.3f}")
    best = max(results, key=lambda r: r["task_speed"])
    print(f"\nBEST: sag={best['_sag']} timing={best['_timing']} "
          f"speed={best['task_speed']}m/s dist={best['dist']} "
          f"minphi={best['min_phi']}")
    out = Path(str(_V1 / "accept_reports")) / pdir.parent.parent.name
    out.mkdir(parents=True, exist_ok=True)
    import json
    (out / "sprint_sweep.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {out/'sprint_sweep.json'}")


if __name__ == "__main__":
    main()
