"""STEPACE 步令验收 harness（STEPACE_V1_PLAN.md）。

用法：
    PYTHONPATH=. python3 baseline/humanoid21/v1/step_accept.py \
        --policy stepcmd --tasks walk_fwd,torso_twist \
        --episodes 4 --out baseline/humanoid21/v1/accept_reports

--policy：别名（stepcmd/steparm/steptvar→各自 policy_exports 最新）
    或导出目录路径；动作取 act() 确定性输出。
--modes：replan（默认纠偏）/ open（开环对照），逗号分隔。
输出：每 (policy,task,mode) 一份 json + 汇总 md 表。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)
from baseline.humanoid21.v1.step_task_scripts import list_scripts

_V1 = Path(__file__).resolve().parent
_RUNS = _REPO / "baseline" / "runs"
_OUT_DIR = _V1 / "accept_reports"

_ALIASES = {
    "stepcmd": _RUNS / "v2_stepcmd_r1" / "policy_exports",
    "steparm": _RUNS / "v2_steparm_r1" / "policy_exports",
    "steptvar": _RUNS / "v2_steptvar_r1" / "policy_exports",
}

_MAX_FRAMES = 300
_ROBOTS = ("robot_a", "robot_b")


def _resolve_policy(spec: str) -> Path:
    p = _ALIASES.get(spec)
    if p is None:
        p = Path(spec)
    if p.is_dir() and (p / "policy.py").exists():
        return p
    subs = sorted(d for d in p.iterdir()
                  if d.is_dir() and (d / "policy.py").exists())
    if not subs:
        raise FileNotFoundError(f"no policy exports under {p}")
    return subs[-1]


def _load_policy(export_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "exp_pol", export_dir / "policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("Exported"):
            return getattr(mod, name)()
    raise RuntimeError(f"no Exported* class in {export_dir}/policy.py")


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------- rollout --
def _rollout(env, pol, task: str, mode: str, seed: int,
             max_frames: int) -> Dict[str, Any]:
    sim = env.simulator
    sim.configure_script(task, gate=None,
                         replan=(mode == "replan"))
    env.reset(seed=seed)
    pol.reset(seed + 11)
    track: Dict[str, List[Dict[str, Any]]] = {r: [] for r in _ROBOTS}
    events: Dict[str, List[Dict[str, float]]] = {r: [] for r in _ROBOTS}
    t = 0
    while t < max_frames and not env.is_episode_over():
        oa, ob = env.get_observation()
        aa, _ = pol.act(oa)
        ab, _ = pol.act(ob)
        env.step(aa, ab)
        t += 1
        for rid, sfx in (("robot_a", "a"), ("robot_b", "b")):
            fl, fr, yaw = sim._feet_yaw(rid)
            mid = 0.5 * (fl + fr)
            phi = env.get_observer_output(
                f"standing_balance_{sfx}")["potential"]
            rep = env.get_observer_output(f"stepace_report_{sfx}")
            track[rid].append({
                "x": float(mid[0]), "y": float(mid[1]),
                "yaw": yaw, "phi": float(phi)})
            if rep.get("event", 0.0) > 0.5:
                events[rid].append(
                    {k: float(v) for k, v in rep.items()})
    out = {"track": track, "events": events,
           "frames": t}
    for rid in _ROBOTS:
        sc = sim._script.get(rid)
        if sc is not None:
            out.setdefault("script_log", {})[rid] = sc.log
            out.setdefault("corrections", {})[rid] = sc.corrections
    return out


# --------------------------------------------------------------- metrics --
def _common(evts: List[Dict[str, float]]) -> Dict[str, float]:
    mv = [e for e in evts if e["hold"] < 0.5]
    m = {
        "n_ev": len(evts), "n_step": len(mv),
        "hold_frac": (len(evts) - len(mv)) / max(len(evts), 1),
        "arr_rate": float(np.mean([e["t_arrive"] >= 0
                                   for e in mv])) if mv else 0.0,
        "d_mv": float(np.mean([e["d_mv"] for e in mv])) if mv else 0,
        "d_st": float(np.mean([e["d_st"] for e in evts])) if evts else 0,
        "d_yaw": float(np.mean([e["d_yaw"] for e in evts])) if evts else 0,
        "trunc_frac": float(np.mean([e["trunc"] > 0.5
                                     for e in evts])) if evts else 0,
        "res0": float(np.mean([e["res0"] for e in mv])) if mv else 0,
        "dur_mean": float(np.mean([e["dur"] for e in evts])) if evts else 0,
    }
    return m


def _task_metrics(task: str, tr: List[Dict[str, Any]],
                  evts: List[Dict[str, float]],
                  logs: List[Dict[str, Any]],
                  corr: List[Dict[str, float]],
                  frames: int) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if not tr:
        return m
    phis = [p["phi"] for p in tr]
    m["min_phi"] = float(min(phis))
    m["fall_frac"] = float(np.mean([p < 0.3 for p in phis]))
    x0, y0, yaw0 = tr[0]["x"], tr[0]["y"], tr[0]["yaw"]
    xe, ye = tr[-1]["x"], tr[-1]["y"]
    dt = frames * 0.05
    fwd = np.array([math.cos(yaw0), math.sin(yaw0)])
    lat = np.array([-math.sin(yaw0), math.cos(yaw0)])
    dx = np.array([xe - x0, ye - y0])
    corr_abs = float(np.mean([abs(c.get("dyaw", 0.0))
                              for c in corr])) if corr else 0.0
    m["corr_abs"] = corr_abs

    if task in ("walk_fwd", "walk_bwd", "turn_arc"):
        m["speed"] = float(fwd @ dx) / max(dt, 1e-6)
        m["dist"] = float(fwd @ dx)
        m["lat_drift"] = float(lat @ dx)
    if task in ("turn_in_place", "torso_twist", "turn_arc"):
        m["yaw_done"] = float(_wrap_pi(tr[-1]["yaw"] - yaw0))
    if task in ("turn_in_place", "torso_twist"):
        cmd_yaw = sum(l["dyaw"] for l in logs
                      if abs(l.get("foot", 0)) > 0)
        m["yaw_cmd"] = float(cmd_yaw) if cmd_yaw else 0.0
        m["yaw_ratio"] = (m["yaw_done"] / cmd_yaw
                          if abs(cmd_yaw) > 1e-6 else 0.0)
        mid0 = np.array([x0, y0])
        m["trans_drift"] = float(np.linalg.norm(
            np.array([xe, ye]) - mid0))
    if task == "torso_twist":
        mv = [e for e in evts if e["hold"] < 0.5]
        m["foot_viol"] = float(np.mean(
            [e["d_mv"] + e["d_st"] for e in mv])) if mv else 0.0
    return m


def _run_one(env, pol, task: str, mode: str,
             episodes: int, seed0: int) -> Dict[str, Any]:
    rows = []
    for ep in range(episodes):
        d = _rollout(env, pol, task, mode, seed0 + ep, _MAX_FRAMES)
        for rid in _ROBOTS:
            ev = d["events"][rid]
            tr = d["track"][rid]
            corr = d.get("corrections", {}).get(rid, [])
            logs = d.get("script_log", {}).get(rid, [])
            row = _common(ev)
            row.update(_task_metrics(
                task, tr, ev, logs, corr, d["frames"]))
            row["_robot"] = rid
            row["_ep"] = ep
            rows.append(row)
    keys = sorted({k for r in rows for k in r if not k.startswith("_")})
    agg = {k: round(float(np.mean([r[k] for r in rows
                                   if k in r])), 3)
           for k in keys}
    return {"agg": agg, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--modes", default="replan,open")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default=str(_OUT_DIR))
    args = ap.parse_args()

    pdir = _resolve_policy(args.policy)
    pol = _load_policy(pdir)
    tasks = (list_scripts() if args.tasks == "all"
             else args.tasks.split(","))
    modes = args.modes.split(",")
    env = ParameterizedEnvBlueprint.load(
        _V1 / "v2_stepace_env.yaml").materialize().build()

    outdir = Path(args.out) / pdir.parent.parent.name
    outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        for mode in modes:
            res = _run_one(env, pol, task, mode,
                           args.episodes, args.seed)
            summary[f"{task}/{mode}"] = res["agg"]
            (outdir / f"{task}_{mode}.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False))
            print(f"[{task}/{mode}] "
                  + " ".join(f"{k}={v}" for k, v in res["agg"].items()))
    md = outdir / "SUMMARY.md"
    keys = sorted({k for a in summary.values() for k in a})
    lines = [f"# {args.policy} ({pdir.name})\n",
             "| task/mode | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for k, a in summary.items():
        lines.append("| " + k + " | "
                     + " | ".join(str(a.get(kk, "-")) for kk in keys)
                     + " |")
    md.write_text("\n".join(lines) + "\n")
    print(f"-> {md}")


if __name__ == "__main__":
    main()
