"""ATTACK_ACCEPT：可部署攻击策略验收（三阶梯）。

    --policy <别名|导出目录>   攻击方基座（V2 79 维步+臂策略）
    --mode  frozen|stand|walk  对手档位
    --opp-policy <别名|目录>   stand/walk 档对手的基座（默认同 --policy）

指标（per episode）：
    hits / dmg（对 robot_b 的命中数与累计伤害，scoring observer）
    t_first_hit（首命中帧）/ ko（对手血量归零或终止）
    opp_fall / self_fall（平衡 observer 的 phi<0.3 或 episode 终止原因）
    min_phi_self / min_phi_opp
    d_end（终局根间距）/ 状态机驻留统计
    OOD 事件：自身摔倒、近身被逼停（APPROACH 下 d 不降持续 N 帧）

视频：--video 每档录第 1 个 episode。

用法：
    PYTHONPATH=.:baseline/humanoid21/v1 python3 -B \
        baseline/humanoid21/v1/attack_accept.py \
        --policy steparm --mode frozen,stand,walk --episodes 4 --video
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
from attack_planner import (AttackPlanner, AttackPolicy, CmdPolicy,
                            make_opp_walk_cmd, opp_hold_cmd)

_ENV_YAML = _V1 / "v2_attack_env.yaml"
_OUT = _V1 / "accept_reports"


def _opp_controller(mode: str, opp_pol) -> Any:
    if mode == "frozen":
        return None                         # 插件钉死，动作随便给
    if mode == "stand":
        return CmdPolicy(opp_pol, opp_hold_cmd)
    if mode == "walk":
        return CmdPolicy(opp_pol, make_opp_walk_cmd())
    raise ValueError(mode)


def _root_gap(env) -> float:
    cs = env.simulator.get_core_state()
    pa = np.asarray(cs["robot_a"]["root_pos"], float)
    pb = np.asarray(cs["robot_b"]["root_pos"], float)
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def _run_episode(env, atk: AttackPolicy, opp, mode: str,
                 seed: int, max_frames: int,
                 video_path: Optional[str] = None) -> Dict[str, Any]:
    env.reset(seed=seed,
              options=({"video_output_path": video_path}
                       if video_path else {}))
    atk.reset(seed + 1)
    if opp is not None:
        opp.reset(seed + 2)

    hits = 0
    dmg = 0.0
    t_first = -1
    min_pa = float("inf")          # 只在 step 后累计，避免 reset 初值污染
    min_pb = float("inf")
    stall_run = 0
    d_prev = _root_gap(env)
    ood: List[str] = []
    states: Dict[str, int] = {}
    term_reason = ""

    for t in range(max_frames):
        oa, ob = env.get_observation()
        aa, _ = atk.act(oa)
        ab, _ = opp.act(ob) if opp is not None else (
            np.zeros(21), None)
        env.step(aa, ab)

        sc = env.get_observer_output("combat")
        evs = sc["robot_b"]["step_hit_events"]
        hits += len(evs)
        dmg += float(sc["robot_b"]["step_damage_taken"])
        if t_first < 0 and evs:
            t_first = t
        pa = float(env.get_observer_output(
            "standing_balance_a")["potential"])
        pb = float(env.get_observer_output(
            "standing_balance_b")["potential"])
        min_pa, min_pb = min(min_pa, pa), min(min_pb, pb)
        states[atk.planner.state] = states.get(atk.planner.state, 0) + 1

        if atk.planner.state == "APPROACH":
            d_now = _root_gap(env)
            if d_now < d_prev - 1e-3:
                stall_run = 0
            else:
                stall_run += 1
            if stall_run == 60:
                ood.append(f"t={t}: 近身停滞 60 帧（d={d_now:.2f}）")
            d_prev = min(d_prev, d_now)

        term = env.get_agent_termination()
        if term.get("robot_a"):
            term_reason = f"robot_a:{term['robot_a']}"
            if "fall" in str(term["robot_a"]).lower() or True:
                ood.append(f"t={t}: 自身终止({term['robot_a']})")
            break
        if term.get("robot_b"):
            term_reason = f"robot_b:{term['robot_b']}"
            break
    else:
        term_reason = "timeout"

    d_end = _root_gap(env)
    if min_pa == float("inf"):
        min_pa = 0.0
    if min_pb == float("inf"):
        min_pb = 0.0
    return {
        "mode": mode, "seed": seed,
        "hits": hits, "dmg_dealt": round(dmg, 3),
        "t_first_hit": t_first,
        "ko": bool(env.get_observer_output("combat")
                   ["robot_b"]["is_ko"]),
        "term": term_reason,
        "min_phi_self": round(min_pa, 3),
        "min_phi_opp": round(min_pb, 3),
        "self_fall": min_pa < 0.3,
        "opp_fall": min_pb < 0.3,
        "d_end": round(d_end, 3),
        "states": states,
        "ood": ood,
    }


def _agg(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(eps)
    return {
        "episodes": n,
        "hits_mean": round(sum(e["hits"] for e in eps) / n, 2),
        "dmg_mean": round(sum(e["dmg_dealt"] for e in eps) / n, 2),
        "first_hit_rate": round(
            sum(e["t_first_hit"] >= 0 for e in eps) / n, 2),
        "t_first_mean": round(
            np.mean([e["t_first_hit"] for e in eps
                     if e["t_first_hit"] >= 0]), 1)
        if any(e["t_first_hit"] >= 0 for e in eps) else -1.0,
        "ko_rate": round(sum(e["ko"] for e in eps) / n, 2),
        "opp_fall_rate": round(
            sum(e["opp_fall"] for e in eps) / n, 2),
        "self_fall_rate": round(
            sum(e["self_fall"] for e in eps) / n, 2),
        "min_phi_self": round(min(e["min_phi_self"] for e in eps), 3),
        "d_end_mean": round(
            sum(e["d_end"] for e in eps) / n, 2),
        "ood_n": sum(len(e["ood"]) for e in eps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--opp-policy", default="")
    ap.add_argument("--mode", default="frozen,stand,walk")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--strike-range", type=float, default=0.62)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--pose-b", default="standing",
                    help="robot_b 初始姿态（standing/squat/...）——"
                         "frozen 档换姿态验证命中管线")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    pdir = _resolve_policy(args.policy)
    base = _load_policy(pdir)
    opp_pdir = _resolve_policy(args.opp_policy or args.policy)
    opp_base = _load_policy(opp_pdir)

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

    planner = AttackPlanner({"strike_range": args.strike_range})
    atk = AttackPolicy(base, planner)

    outdir = Path(args.out) / "attack" / pdir.parent.parent.name
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
            eps.append(_run_episode(env, atk, opp, mode,
                                    args.seed + ep * 17,
                                    args.frames, vpath))
        summary[mode] = {"agg": _agg(eps), "episodes": eps}
        (outdir / f"{mode}.json").write_text(
            json.dumps(summary[mode], indent=1, ensure_ascii=False))
        print(f"[{mode}] "
              + " ".join(f"{k}={v}"
                         for k, v in summary[mode]["agg"].items()))
    (outdir / "SUMMARY.json").write_text(
        json.dumps({m: s["agg"] for m, s in summary.items()},
                   indent=1, ensure_ascii=False))
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()
