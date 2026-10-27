"""Offline validation for the punch planner.

Stratifies the extracted P5 set by reach gap (in-range / needs-step /
out-of-reach / out-of-sector), plans a punch per sampled state, and
exports a three-view MP4 per case via mocap.point5.viz — for eyeballing
plan quality before the reward adapter is built.

Usage:
    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/viz_punch_plans.py \
        --p5 baseline/humanoid21/hierarchy/assets/initial_states/strike_start_v1/strike_start_p5.npz \
        --out-dir /tmp/punch_plans --per-bucket 4 --hand left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hand", default="left")
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--max-step", type=float, default=0.35)
    ap.add_argument("--virtual", action="store_true",
                    help="Synthetic target ahead of the stance at "
                         "--virtual-dist (validates punch shape on any "
                         "state, ignoring the recorded opponent)")
    ap.add_argument("--virtual-dist", type=float, default=0.85)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    from scipy.spatial.transform import Rotation as R

    from mocap.point5.contracts import Point5State, StaticStartResolver
    from mocap.point5.punch_planner import (
        PunchPlannerConfig, _norm, _stance_forward, _xy, plan_punch)
    from mocap.point5.trajectory import build_reference
    from mocap.point5.viz import export_mp4

    d = np.load(args.p5, allow_pickle=False)
    cfg = PunchPlannerConfig(max_step=args.max_step)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(d["p5"])
    buckets: dict[str, list] = {
        "in_range": [], "step": [], "unreachable": []}
    stats = {"reachable": 0, "out_of_reach": 0, "out_of_sector": 0}
    def _target(i: int, st: Point5State) -> np.ndarray:
        if not args.virtual:
            return np.asarray(d["target"][i], dtype=np.float64)
        head = st.pos[0]
        fwd = _stance_forward(st.pos[4], st.pos[3])
        if not np.any(fwd):
            fwd = _norm(_xy(d["target"][i] - head))
        return np.array([head[0] + fwd[0] * args.virtual_dist,
                         head[1] + fwd[1] * args.virtual_dist,
                         head[2] - 0.10])

    for i in range(n):
        st = Point5State(pos=np.asarray(d["p5"][i], dtype=np.float64))
        plan = plan_punch(st, _target(i, st), args.hand, cfg)
        if plan.reachable:
            stats["reachable"] += 1
            key = "in_range" if plan.diag["step_len"] <= 1e-6 else "step"
        else:
            stats[plan.reason] = stats.get(plan.reason, 0) + 1
            key = "unreachable"
        buckets[key].append((i, plan))

    print(f"== plan stats over {n} states (hand={args.hand}) ==", flush=True)
    for k, v in stats.items():
        print(f"  {k:14s} {v:5d}  ({v / n:.1%})", flush=True)

    rng = np.random.default_rng(0)
    for bname, items in buckets.items():
        if not items:
            continue
        pick = rng.choice(len(items), size=min(args.per_bucket,
                                               len(items)), replace=False)
        for j in pick:
            i, plan = items[int(j)]
            tag = f"{bname}_{i:04d}"
            if not plan.reachable:
                print(f"[{tag}] unreachable: {plan.reason}", flush=True)
                continue
            st = Point5State(pos=np.asarray(d["p5"][i],
                                            dtype=np.float64))
            q = np.asarray(d["root_rot"][i], dtype=np.float64)
            rot_m = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            resolver = StaticStartResolver(
                st, np.asarray(d["root_pos"][i], dtype=np.float64),
                rot_m)
            ref = build_reference(plan.task, resolver)
            mp4 = out / f"{tag}.mp4"
            export_mp4(ref, mp4)
            d_ = plan.diag
            print(f"[{tag}] step={d_['step_len']:.3f} "
                  f"move={d_['moving_foot']} "
                  f"cos={d_['cos_attack_vs_stance']:+.2f} "
                  f"gap={d_['reach_gap']:+.3f} "
                  f"pair={d['pair_dist'][i]:.2f} -> {mp4}", flush=True)


if __name__ == "__main__":
    main()
