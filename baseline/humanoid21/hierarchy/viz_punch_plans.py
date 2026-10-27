"""Offline validation for the punch planner (per-robot P5 set).

Strike targets are SYNTHESISED relative to the robot's own stance —
the recorded opponent is irrelevant.  Two passes:

1. STATS: every state plans against a canonical target (stance-forward
   at --stat-dist, head height) -> reach/step/moving-foot distribution.
2. MP4: a few sampled states each plan against a small grid of targets
   (dist x lateral-offset) -> three-view videos for eyeballing.

Usage:
    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/viz_punch_plans.py \
        --p5 .../strike_start_p5.npz --out-dir /data1/dev/punch_plans/v2 \
        --hand left --n-states 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _mk_target(head: np.ndarray, fwd: np.ndarray,
               dist: float, lateral: float, z: float) -> np.ndarray:
    """Target = head + fwd·dist + right·lateral + (0,0,z−head_z offset)."""
    right = np.array([fwd[1], -fwd[0], 0.0])     # fwd x up -> right?  verify sign:
    # fwd=(1,0): right=(0,-1) → -Y is right when facing +X.  Correct.
    return np.array([head[0] + fwd[0] * dist + right[0] * lateral,
                     head[1] + fwd[1] * dist + right[1] * lateral,
                     z])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hand", default="left")
    ap.add_argument("--n-states", type=int, default=6)
    ap.add_argument("--stat-dist", type=float, default=0.60)
    ap.add_argument("--max-step", type=float, default=0.35)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    from scipy.spatial.transform import Rotation as R

    from mocap.point5.contracts import Point5State, StaticStartResolver
    from mocap.point5.punch_planner import (
        PunchPlannerConfig, _stance_forward, plan_punch)
    from mocap.point5.trajectory import build_reference
    from mocap.point5.viz import export_mp4

    d = np.load(args.p5, allow_pickle=False)
    cfg = PunchPlannerConfig(max_step=args.max_step)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(d["p5"])

    def stance_fwd(p5):
        f = _stance_forward(p5[4], p5[3])
        return f if np.any(f) else np.array([1.0, 0.0, 0.0])

    # ---------- pass 1: stats over the whole set -------------------
    stats = {"reachable": 0, "out_of_reach": 0, "out_of_sector": 0}
    steps, movers = [], {"foot_left": 0, "foot_right": 0}
    for i in range(n):
        p5 = np.asarray(d["p5"][i], dtype=np.float64)
        st = Point5State(pos=p5)
        head = p5[0]
        tgt = _mk_target(head, stance_fwd(p5), args.stat_dist, 0.0,
                         head[2] - 0.10)
        plan = plan_punch(st, tgt, args.hand, cfg)
        if plan.reachable:
            stats["reachable"] += 1
            steps.append(plan.diag["step_len"])
            movers[plan.diag["moving_foot"]] += 1
        else:
            stats[plan.reason] = stats.get(plan.reason, 0) + 1
    print(f"== canonical target (fwd {args.stat_dist} m, head-0.10 m) "
          f"over {n} states, hand={args.hand} ==", flush=True)
    for k, v in stats.items():
        print(f"  {k:14s} {v:5d} ({v / n:.1%})", flush=True)
    if steps:
        s = np.asarray(steps)
        print(f"  step_len: mean {s.mean():.3f} p50 {np.median(s):.3f} "
              f"max {s.max():.3f}  mover L/R = "
              f"{movers['foot_left']}/{movers['foot_right']}", flush=True)

    # ---------- pass 2: MP4s on sampled states ---------------------
    rng = np.random.default_rng(0)
    picks = rng.choice(n, size=min(args.n_states, n), replace=False)
    variants = [(d_, l_) for d_ in (0.40, 0.85)
                for l_ in (-0.20, 0.0, 0.20)]
    for i in picks:
        p5 = np.asarray(d["p5"][i], dtype=np.float64)
        st = Point5State(pos=p5)
        fwd = stance_fwd(p5)
        head = p5[0]
        for dist, lat in variants:
            tgt = _mk_target(head, fwd, dist, lat, head[2] - 0.10)
            plan = plan_punch(st, tgt, args.hand, cfg)
            tag = (f"s{i:04d}_d{dist:.2f}_l{lat:+.2f}")
            if not plan.reachable:
                print(f"[{tag}] {plan.reason}", flush=True)
                continue
            q = np.asarray(d["root_rot"][i], dtype=np.float64)
            rot_m = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            ref = build_reference(plan.task, StaticStartResolver(
                st, np.asarray(d["root_pos"][i]), rot_m))
            mp4 = out / f"{tag}.mp4"
            export_mp4(ref, mp4)
            dg = plan.diag
            print(f"[{tag}] step={dg['step_len']:.3f} "
                  f"move={dg['moving_foot']} -> {mp4}", flush=True)


if __name__ == "__main__":
    main()
