"""STEPBUDGET_ACCEPT：方向+预算步令基座的战斗走位验收（三阶梯）。

    --policy <stepbudget 导出目录>   攻击方基座（v2_stepbudget）
    --opp-policy <stepcmd/steparm 导出目录>  stand/walk 档对手
    --mode  frozen|stand|walk

指标（per episode）：
    t_reach     首次进入 strike_range 的帧数（−1=未到达）
    appr_v      逼近阶段速度 = 覆盖距离/时间（m/s，20fps）
    band_frac   d ∈ [close, strike_range+0.15] 的帧占比
    d_min / d_end / d_tail100（末 100 帧均值）
    orbit_deg   ORBIT 状态下对手方位角累计变化（°，绕行覆盖）
    min_phi_self / self_fall / term
    mode 驻留帧数；晋升次数 n_promo

视频：--video 每档录第 1 个 episode。

用法：
    PYTHONPATH=.:baseline/humanoid21/v1 python3 -B \
        baseline/humanoid21/v1/stepbudget_accept.py \
        --policy baseline/runs/v2_stepbudget_r1/policy_exports/u02580 \
        --opp-policy steparm --mode frozen,stand,walk \
        --episodes 2 --video
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_V1 = Path(__file__).resolve().parent
_REPO = _V1.parents[2]
for _p in (str(_REPO), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)
from step_accept import _load_policy, _resolve_policy
from attack_planner import (AttackPolicy, CmdPolicy,
                            make_opp_walk_cmd, opp_hold_cmd)
from stepbudget_planner import (BudgetAttackPlanner,
                                BudgetAttackPolicy)

_ENV_YAML = _V1 / "v2_attack_env.yaml"
_OUT = _V1 / "accept_reports"
_DT = 0.05  # 20 action fps


def _opp_controller(mode: str, opp_pol) -> Any:
    if mode == "frozen":
        return None
    if mode == "stand":
        return CmdPolicy(opp_pol, opp_hold_cmd)
    if mode == "walk":
        return CmdPolicy(opp_pol, make_opp_walk_cmd())
    raise ValueError(mode)


def _core(env):
    return env.simulator.get_core_state()


def _yaw_fn(env):
    """planner 世界真值供给：ego yaw + 双方根 xy。"""
    def fn():
        core = _core(env)
        q = np.asarray(core["robot_a"]["root_rot"], np.float64)
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                         1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        return (yaw,
                np.asarray(core["robot_a"]["root_pos"][:2]),
                np.asarray(core["robot_b"]["root_pos"][:2]))
    return fn


def _run_episode(env, atk: BudgetAttackPolicy, opp, mode: str,
                 seed: int, max_frames: int,
                 video_path: Optional[str] = None) -> Dict[str, Any]:
    env.reset(seed=seed,
              options=({"video_output_path": video_path}
                       if video_path else {}))
    atk.reset(seed + 1)
    if opp is not None:
        opp.reset(seed + 2)

    min_pa = float("inf")
    states: Dict[str, int] = {}
    t_reach = -1
    d_traj: List[float] = []
    bearing_prev: Optional[float] = None
    orbit_deg = 0.0
    n_promo = 0
    promo_age_last = 0
    term_reason = ""

    for t in range(max_frames):
        oa, ob = env.get_observation()
        aa, _ = atk.act(oa)
        ab, _ = opp.act(ob) if opp is not None else (
            np.zeros(21), None)
        env.step(aa, ab)

        core = _core(env)
        d = float(np.linalg.norm(
            np.asarray(core["robot_a"]["root_pos"][:2])
            - np.asarray(core["robot_b"]["root_pos"][:2])))
        d_traj.append(d)
        rel = (np.asarray(core["robot_b"]["root_pos"][:2])
               - np.asarray(core["robot_a"]["root_pos"][:2]))
        bearing = math.atan2(rel[1], rel[0])
        if bearing_prev is not None and \
                atk.planner.state == "ORBIT":
            orbit_deg += abs((bearing - bearing_prev + math.pi)
                             % (2 * math.pi) - math.pi)
        bearing_prev = bearing
        if t_reach < 0 and d <= atk.planner.p["strike_range"]:
            t_reach = t

        pa = float(env.get_observer_output(
            "standing_balance_a")["potential"])
        min_pa = min(min_pa, pa)
        states[atk.planner.state] = states.get(
            atk.planner.state, 0) + 1

        term = env.get_agent_termination()
        if term.get("robot_a"):
            term_reason = f"robot_a:{term['robot_a']}"
            break
        if term.get("robot_b"):
            term_reason = f"robot_b:{term['robot_b']}"
            break
    else:
        term_reason = "timeout"

    da = np.asarray(d_traj)
    lo, hi = (atk.planner.p["close_range"],
              atk.planner.p["strike_range"] + 0.15)
    n = max(len(da), 1)
    appr_v = 0.0
    if t_reach > 0:
        appr_v = (da[0] - atk.planner.p["strike_range"]) \
            / (t_reach * _DT)
    return {
        "mode": mode, "seed": seed, "term": term_reason,
        "t_reach": t_reach,
        "appr_v": round(appr_v, 2),
        "band_frac": round(float(((da >= lo) & (da <= hi)).mean()),
                           3),
        "d_min": round(float(da.min()), 3),
        "d_end": round(float(da[-1]), 3),
        "d_tail100": round(float(da[-100:].mean()), 3),
        "orbit_deg": round(math.degrees(orbit_deg), 1),
        "min_phi_self": round(min_pa if min_pa != float("inf")
                              else 0.0, 3),
        "self_fall": min_pa < 0.3,
        "states": states,
    }


def _agg(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(eps)
    reach = [e["t_reach"] for e in eps if e["t_reach"] >= 0]
    return {
        "episodes": n,
        "reach_rate": round(len(reach) / n, 2),
        "t_reach_mean": round(float(np.mean(reach)), 1)
        if reach else -1,
        "appr_v_mean": round(float(np.mean(
            [e["appr_v"] for e in eps])), 2),
        "band_frac_mean": round(float(np.mean(
            [e["band_frac"] for e in eps])), 3),
        "d_tail_mean": round(float(np.mean(
            [e["d_tail100"] for e in eps])), 3),
        "orbit_deg_mean": round(float(np.mean(
            [e["orbit_deg"] for e in eps])), 1),
        "self_fall_rate": round(sum(e["self_fall"]
                                    for e in eps) / n, 2),
        "min_phi_self": round(min(e["min_phi_self"]
                                  for e in eps), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--opp-policy", default="")
    ap.add_argument("--mode", default="frozen,stand,walk")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--strike-range", type=float, default=0.62)
    ap.add_argument("--close-range", type=float, default=0.42)
    ap.add_argument("--pose-b", default="standing")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    pdir = _resolve_policy(args.policy)
    base = _load_policy(pdir)
    opp_base = _load_policy(
        _resolve_policy(args.opp_policy or args.policy))

    env = (ParameterizedEnvBlueprint.load(_ENV_YAML)
           .materialize(initial_distance=args.distance,
                        pose_b=args.pose_b)
           .build())
    modes = args.mode.split(",")
    if "frozen" in modes:
        from envs.humanoid21.plugins import FrozenRobotPlugin
        env.attach_plugin(FrozenRobotPlugin("robot_b"))
    if args.video:
        from envs.framework.common_plugins import (
            VideoRecorderPlugin,
        )
        env.attach_plugin(VideoRecorderPlugin(fps=30))

    planner = BudgetAttackPlanner({
        "strike_range": args.strike_range,
        "close_range": args.close_range,
    })
    atk = BudgetAttackPolicy(base, planner)
    atk.attach_world(_yaw_fn(env))

    outdir = Path(args.out) / "attack_budget" / pdir.name
    outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    for mode in modes:
        opp = _opp_controller(mode, opp_base)
        eps: List[Dict[str, Any]] = []
        for ep in range(args.episodes):
            vpath = None
            if args.video and ep == 0:
                vdir = outdir / "videos"
                vdir.mkdir(exist_ok=True)
                vpath = str(vdir / f"{mode}.mp4")
            t0 = time.time()
            eps.append(_run_episode(env, atk, opp, mode,
                                    args.seed + ep * 17,
                                    args.frames, vpath))
            print(f"  [{mode} ep{ep}] "
                  f"{time.time() - t0:.0f}s "
                  f"{eps[-1]}")
        summary[mode] = {"agg": _agg(eps), "episodes": eps}
        (outdir / f"{mode}.json").write_text(
            json.dumps(summary[mode], indent=1,
                       ensure_ascii=False))
        print(f"[{mode}] "
              + " ".join(f"{k}={v}" for k, v in
                         summary[mode]["agg"].items()))
    (outdir / "SUMMARY.json").write_text(
        json.dumps({m: s["agg"] for m, s in summary.items()},
                   indent=1, ensure_ascii=False))
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()
