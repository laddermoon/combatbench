"""Extract per-robot Point5 start poses from the saved core-state set.

The state set is a pool of SINGLE-robot start poses — the recorded
opponent is irrelevant for strike planning (the punch target is
synthesised relative to the robot's own stance at plan time).  Each
stored pair state therefore yields TWO entries, one per robot:

    p5            (5,3) measured Point5State positions
    root_pos/rot  (3,)/(4,) robot root — for StaticStartResolver ctx
    src_ep/src_step/robot  provenance
    pair_dist     scalar  (informational only)

Usage:
    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/extract_strike_start_p5.py \
        --states .../strike_start_states.npz \
        --out .../strike_start_p5.npz
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
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    from envs.humanoid21.simulator import Humanoid21Simulator
    from mocap.point5.humanoid21.adapter import point5_from_view

    d = np.load(args.states, allow_pickle=False)
    n = len(d["steps"]) if args.max <= 0 else min(
        args.max, len(d["steps"]))

    sim = Humanoid21Simulator()
    sim.reset(seed=0)

    p5s, rpos, rrot, pdist, robot_idx, src_step = [], [], [], [], [], []
    for i in range(n):
        cs = {}
        for rid in ("robot_a", "robot_b"):
            cs[rid] = {f: d[f"{rid}/{f}"][i] for f in _FIELDS}
        sim.set_core_state(cs)
        ds = sim.get_derived_state(["robot_a", "robot_b"])
        for ri, rid in enumerate(("robot_a", "robot_b")):
            p5 = point5_from_view(ds[rid], rid, with_vel=False)
            p5s.append(p5.pos)
            rpos.append(cs[rid]["root_pos"])
            rrot.append(cs[rid]["root_rot"])
            robot_idx.append(ri)
            src_step.append(int(d["steps"][i]))
        pdist.append(float(np.linalg.norm(
            cs["robot_a"]["root_pos"][:2] - cs["robot_b"]["root_pos"][:2])))
        pdist.append(pdist[-1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        p5=np.stack(p5s),
        root_pos=np.stack(rpos),
        root_rot=np.stack(rrot),
        pair_dist=np.asarray(pdist, dtype=np.float32),
        robot=np.asarray(robot_idx, dtype=np.int8),
        src_step=np.asarray(src_step, dtype=np.int32),
        meta_json=np.asarray(json.dumps({
            "states_file": args.states,
            "n": len(p5s),
            "semantics": "per-robot start poses; strike target is "
                        "synthesised from own stance at plan time",
        })),
    )
    print(f"extracted {len(p5s)} per-robot p5 states -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
