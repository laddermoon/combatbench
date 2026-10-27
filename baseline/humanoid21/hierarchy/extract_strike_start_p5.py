"""Extract Point5 initial states + strike targets from the saved
core-state set (strike_start_states.npz).

For every stored pair state: set_core_state -> get_derived_state ->
point5_from_view for both robots.  The STRIKER is ``--striker``
(default robot_a); the TARGET is the opponent's head point (from the
opponent's own P5 extraction — body_xpos is ground truth).

Output npz per state i:
    p5            (5,3) striker's measured Point5State positions
    target        (3,)  world point to hit (opponent head)
    root_pos/rot  (3,)/(4,) striker root — for StaticStartResolver ctx
    pair_dist     scalar  horizontal striker-opponent distance
    src_ep/src_step       provenance

Usage:
    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/extract_strike_start_p5.py \
        --states baseline/humanoid21/hierarchy/assets/initial_states/strike_start_v1/strike_start_states.npz \
        --out baseline/humanoid21/hierarchy/assets/initial_states/strike_start_v1/strike_start_p5.npz \
        --striker robot_a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_FIELDS = (
    "root_pos", "root_rot", "root_vel_local",
    "root_angular_vel_local", "joint_pos_norm", "joint_vel_norm",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--striker", default="robot_a")
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    from envs.humanoid21.simulator import Humanoid21Simulator
    from mocap.point5.humanoid21.adapter import point5_from_view

    other = "robot_b" if args.striker == "robot_a" else "robot_a"
    d = np.load(args.states, allow_pickle=False)
    n = len(d["steps"]) if args.max <= 0 else min(
        args.max, len(d["steps"]))

    sim = Humanoid21Simulator()
    sim.reset(seed=0)

    p5s, tgts, rpos, rrot, pdist = [], [], [], [], []
    for i in range(n):
        cs = {}
        for rid in ("robot_a", "robot_b"):
            cs[rid] = {f: d[f"{rid}/{f}"][i] for f in _FIELDS}
        sim.set_core_state(cs)
        ds = sim.get_derived_state(["robot_a", "robot_b"])
        p5_s = point5_from_view(ds[args.striker], args.striker,
                                with_vel=False)
        p5_o = point5_from_view(ds[other], other, with_vel=False)
        p5s.append(p5_s.pos)
        tgts.append(p5_o.point_pos("head"))
        cs_s = cs[args.striker]
        cs_o = cs[other]
        rpos.append(cs_s["root_pos"])
        rrot.append(cs_s["root_rot"])
        pdist.append(float(np.linalg.norm(
            cs_s["root_pos"][:2] - cs_o["root_pos"][:2])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        p5=np.stack(p5s),
        target=np.stack(tgts),
        root_pos=np.stack(rpos),
        root_rot=np.stack(rrot),
        pair_dist=np.asarray(pdist, dtype=np.float32),
        src_step=d["steps"][:n],
        meta_json=np.asarray(json.dumps({
            "states_file": args.states,
            "striker": args.striker,
            "n": n,
            "target_semantics": "opponent head (Point5 'head')",
        })),
    )
    print(f"extracted {n} p5 states -> {out}", flush=True)


if __name__ == "__main__":
    main()
