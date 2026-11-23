"""PUNCH_SWEEP：CmdProbe 拳原语扫描（离线）。

臂 tgt 是"归一化关节目标"（q = tgt*sc + ref，tgt∈[-1,1]³），
语义不直观——实测 tgt=0 是大幅前伸位而非收拳。本脚本扫
tgt 网格，用 CmdProbe 实测每次挥击的 ego 前伸/手部峰速/
平衡代价，选出最强拳原语并（可选）拼"伸展→收回警戒"两段
序列复评。产出写 attack_params.json。

用法：
    PYTHONPATH=.:baseline/humanoid21/v1 python3 -B \
        baseline/humanoid21/v1/punch_sweep.py \
        --policy steparm [--grid 5] [--seq-dur 6,8,10] [--top 8]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_V1 = Path(__file__).resolve().parent
_REPO = _V1.parents[2]
import sys
for _p in (str(_REPO), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cmd_probe import CmdProbe
from envs.humanoid21.simulator import Humanoid21Simulator
from point5.humanoid21.robot_state import joint_ranges

_OUT = _V1 / "accept_reports" / "punch_sweep"


def _obs_standing(seed: int = 7) -> np.ndarray:
    """站立基态 obs96（冒烟同款）。"""
    s = Humanoid21Simulator()
    s.reset(seed=seed)
    return np.asarray(s.get_observation()["robot_a"], np.float64)


def _guard_tgt(obs: np.ndarray, side: str, probe: CmdProbe) -> np.ndarray:
    """当前臂关节角 → 归一化 tgt（"回到原位"的收回段目标）。"""
    i = 15 if side == "r" else 18
    q = np.asarray(obs[i:i + 3], float)      # obs[15:18]=归一化qpos？否——
    # obs[0:21] 是归一化关节位置（RobotState: qpos_norm），
    # 与臂 tgt 同一归一化域——直接可用。
    return np.clip(q, -1.0, 1.0)


def _mk_cmd(tgt_r: Optional[np.ndarray] = None,
            tgt_l: Optional[np.ndarray] = None) -> np.ndarray:
    c = np.zeros(22)
    if tgt_r is not None:
        c[0] = 1.0
        c[1:4] = tgt_r
    if tgt_l is not None:
        c[4] = 1.0
        c[5:8] = tgt_l
    return c


def _score(r: Dict[str, Any], side: str = "r") -> Dict[str, float]:
    h = r["hands"].get(side, {})
    bal = r["balance"]
    return {
        "v_peak": float(h.get("v_peak", 0.0)),
        "v_at_reach": float(h.get("v_at_reach", 0.0)),
        "reach_at_vpeak": float(h.get("reach_at_vpeak", 0.0)),
        "reach_fwd": float(h.get("reach_fwd", 0.0)),
        "reach_z": float(h.get("reach_pos", [0, 0, 0])[2]),
        "t_vpeak": int(h.get("t_vpeak", -1)),
        "t_reach": int(h.get("t_reach", -1)),
        "err_end": float(r["arm"].get("err_r_end", 0.0)),
        "min_phi": float(bal.get("min_phi", 0.0)),
        "fall": bool(bal.get("fall", False)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="steparm")
    ap.add_argument("--grid", type=int, default=5,
                    help="每维采样数（默认 5：-1,-.5,0,.5,1）")
    ap.add_argument("--values", default="",
                    help="自定义网格值逗号串，覆盖 --grid")
    ap.add_argument("--side", default="r", choices=["r", "l"])
    ap.add_argument("--dur", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--obs-seed", type=int, default=7)
    ap.add_argument("--seq-dur", default="6,8,10",
                    help="对 top 候选再拼 [伸展 k f → 收回警戒] 复评")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--phi-min", type=float, default=0.6,
                    help="平衡及格线：低于此的候选丢弃")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    vals = ([float(x) for x in args.values.split(",")]
            if args.values else
            np.linspace(-1.0, 1.0, args.grid).tolist())

    obs = _obs_standing(args.obs_seed)
    probe = CmdProbe(args.policy)
    side = args.side
    guard = _guard_tgt(obs, side, probe)
    print(f"[punch] policy={args.policy} side={side} "
          f"grid={len(vals)}³={len(vals)**3} "
          f"guard_tgt={np.round(guard, 3).tolist()}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, tgt in enumerate(itertools.product(vals, repeat=3)):
        tgt = np.asarray(tgt, float)
        cmd = _mk_cmd(tgt_r=tgt) if side == "r" else _mk_cmd(tgt_l=tgt)
        r = probe.evaluate(obs, cmd, dur=args.dur,
                           horizon=args.horizon)
        m = _score(r, side)
        rows.append({"tgt": tgt.tolist(), "phase": "single", **m})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(vals)**3} "
                  f"({el:.0f}s, eta {el/(i+1)*(len(vals)**3-i-1):.0f}s)")

    ok = [r for r in rows
          if not r["fall"] and r["min_phi"] >= args.phi_min]
    # 排序主键 = 命中速度×前伸（v_at_reach 才是真·出拳末速，
    # v_peak 常出现在段切换的收拳甩鞭帧）
    ok.sort(key=lambda r: (r["v_at_reach"] * max(r["reach_fwd"], 0.0)),
            reverse=True)
    print(f"[single] {len(ok)}/{len(rows)} 过平衡线 "
          f"(phi≥{args.phi_min} & no fall)")
    for r in ok[:args.top]:
        print(f"  tgt={np.round(r['tgt'],2).tolist()} "
              f"v@reach={r['v_at_reach']:.2f} vpk={r['v_peak']:.2f} "
              f"reach={r['reach_fwd']:.3f} "
              f"z={r['reach_z']:+.3f} t_r={r['t_reach']} "
              f"err={r['err_end']:.3f} phi={r['min_phi']:.3f}")

    # 两段序列复评 top 候选：[伸展 k f → 收回警戒 ∞]
    seq_durs = [int(x) for x in args.seq_dur.split(",") if x]
    flag = {"r": np.array([True, False]),
            "l": np.array([False, True])}[side]
    best_seqs: List[Dict[str, Any]] = []
    for r in ok[:args.top]:
        tgt = np.asarray(r["tgt"], float)
        tgt6 = np.zeros(6)
        tgt6[0:3] = tgt if side == "r" else 0.0
        tgt6[3:6] = tgt if side == "l" else 0.0
        g6 = np.zeros(6)
        g6[0:3] = guard if side == "r" else 0.0
        g6[3:6] = guard if side == "l" else 0.0
        for k in seq_durs:
            cmd = (_mk_cmd(tgt_r=tgt) if side == "r"
                   else _mk_cmd(tgt_l=tgt))
            seq = [(flag, tgt6, k), (flag, g6, -1)]
            rr = probe.evaluate(obs, cmd, dur=args.dur,
                                horizon=args.horizon + 5,
                                arm_seq=seq)
            m = _score(rr, side)
            best_seqs.append({"tgt": tgt.tolist(),
                              "ext_dur": k, "phase": "seq",
                              **m})
    best_seqs.sort(
        key=lambda r: (r["v_at_reach"] * max(r["reach_fwd"], 0.0)),
        reverse=True)
    print(f"[seq] top:")
    for r in best_seqs[:args.top]:
        print(f"  tgt={np.round(r['tgt'],2).tolist()} k={r['ext_dur']} "
              f"v@reach={r['v_at_reach']:.2f} vpk={r['v_peak']:.2f} "
              f"reach={r['reach_fwd']:.3f} "
              f"z={r['reach_z']:+.3f} t_r={r['t_reach']} "
              f"phi={r['min_phi']:.3f} fall={r['fall']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    (out / f"sweep_{ts}.json").write_text(json.dumps({
        "policy": args.policy, "side": side,
        "guard_tgt": guard.tolist(),
        "grid": vals, "dur": args.dur, "horizon": args.horizon,
        "single": rows, "seq": best_seqs,
    }, ensure_ascii=False, indent=1))
    (out / "attack_params.json").write_text(json.dumps({
        "side": side, "guard_tgt": guard.tolist(),
        "punch": best_seqs[0] if best_seqs else None,
        "singles_top": ok[:args.top],
    }, ensure_ascii=False, indent=1))
    print(f"[out] {out}/sweep_{ts}.json + attack_params.json")


if __name__ == "__main__":
    main()
