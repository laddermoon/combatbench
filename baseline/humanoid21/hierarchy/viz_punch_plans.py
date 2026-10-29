"""Scenario-space validation for the punch planner.

The task space is a product: (initial pose) x (target) x (hand) —
2 304 sampled stances, continuous target in the frontal envelope,
two hands.  This script samples that space randomly, reports the
outcome distribution, and exports three-view MP4s per outcome
bucket so real plans can be eyeballed (successes AND the honest
failure modes).

Usage:
    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/viz_punch_plans.py \
        --p5 .../strike_start_p5.npz --out-dir /data1/dev/punch_plans/scen \
        --scenarios 400 --per-bucket 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenarios", type=int, default=400)
    ap.add_argument("--per-bucket", type=int, default=8)
    ap.add_argument("--max-step", type=float, default=0.35)
    # target sampling envelope (stance frame, metres)
    ap.add_argument("--dist", type=float, nargs=2, default=(0.3, 1.2))
    ap.add_argument("--lat", type=float, nargs=2, default=(-0.35, 0.35))
    ap.add_argument("--zdrop", type=float, nargs=2,
                    default=(0.0, 0.55), help="target = head_z - U(a,b)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    from scipy.spatial.transform import Rotation as R

    from point5.contracts import Point5State, StaticStartResolver
    from point5.humanoid21.punch_planner import (
        PunchPlannerConfig, _stance_forward, plan_punch)
    from point5.trajectory import build_reference
    from point5.viz import export_mp4

    d = np.load(args.p5, allow_pickle=False)
    cfg = PunchPlannerConfig(max_step=args.max_step)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(d["p5"])
    rng = np.random.default_rng(1)

    def stance_fwd(p5):
        f = _stance_forward(p5[4], p5[3])
        return f if np.any(f) else np.array([1.0, 0.0, 0.0])

    buckets: dict[str, list] = {}
    for k in range(args.scenarios):
        i = int(rng.integers(n))
        p5 = np.asarray(d["p5"][i], dtype=np.float64)
        st = Point5State(pos=p5)
        fwd = stance_fwd(p5)
        right = np.array([fwd[1], -fwd[0], 0.0])
        head = p5[0]
        dist = float(rng.uniform(*args.dist))
        lat = float(rng.uniform(*args.lat))
        zdrop = float(rng.uniform(*args.zdrop))
        tgt = np.array([head[0] + fwd[0] * dist + right[0] * lat,
                        head[1] + fwd[1] * dist + right[1] * lat,
                        head[2] - zdrop])
        hand = "left" if rng.random() < 0.5 else "right"
        plan = plan_punch(st, tgt, hand, cfg)
        if plan.reachable:
            key = "in_range" if plan.diag["step_len"] <= 1e-6 else "step"
        else:
            key = plan.reason
        buckets.setdefault(key, []).append(dict(
            state=i, hand=hand, dist=dist, lat=lat, zdrop=zdrop,
            target=tgt, plan=plan))

    print(f"== {args.scenarios} sampled scenarios ==", flush=True)
    for key, items in sorted(buckets.items(),
                             key=lambda kv: -len(kv[1])):
        print(f"  {key:14s} {len(items):4d} ({len(items)/args.scenarios:.1%})",
              flush=True)

    # one MP4 per sampled case, up to per-bucket
    for key, items in buckets.items():
        take = rng.choice(len(items),
                          size=min(args.per_bucket, len(items)),
                          replace=False)
        for j in take:
            rec = items[int(j)]
            i, plan = rec["state"], rec["plan"]
            if not plan.reachable:
                continue
            st = Point5State(pos=np.asarray(d["p5"][i], np.float64))
            q = np.asarray(d["root_rot"][i], np.float64)
            rot_m = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            ref = build_reference(plan.task, StaticStartResolver(
                st, np.asarray(d["root_pos"][i]), rot_m))
            tag = (f"{key}_s{i:04d}_{rec['hand'][0]}"
                   f"_d{rec['dist']:.2f}l{rec['lat']:+.2f}"
                   f"z{rec['zdrop']:.2f}")
            mp4 = out / f"{tag}.mp4"
            export_mp4(ref, mp4, measured=st.pos)
            dg = plan.diag
            print(f"[{tag}] step={dg['step_len']:.3f} "
                  f"move={dg['moving_foot']}", flush=True)

    (out / "summary.json").write_text(json.dumps({
        k: len(v) for k, v in buckets.items()}, indent=2))


if __name__ == "__main__":
    main()
