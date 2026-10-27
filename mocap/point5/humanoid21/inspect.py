"""Inspect point5 quantities on a live Humanoid21Simulator.

Usage:
    PYTHONPATH=. python3 -m mocap.point5.humanoid21.inspect [--robot robot_a]
    PYTHONPATH=. python3 -m mocap.point5.humanoid21.inspect --task task.json --dt 0.05
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from mocap.point5 import (
    POINT_NAMES,
    Point5Task,
    build_reference,
    is_planted,
)
from mocap.point5.humanoid21 import Humanoid21Point5


def _fmt_state(st) -> str:
    lines = []
    for i, n in enumerate(POINT_NAMES):
        p = st.pos[i]
        flag = " planted" if n.startswith("foot") and is_planted(p) else ""
        lines.append(f"  {n:11s} [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]{flag}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="point5-humanoid21-inspect")
    ap.add_argument("--robot", default="robot_a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", help="Point5Task JSON file to sample")
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args(argv)

    from envs.humanoid21.simulator import Humanoid21Simulator
    sim = Humanoid21Simulator()
    sim.reset(seed=args.seed)
    adapter = Humanoid21Point5(sim)

    if not args.task:
        for rid in ("robot_a", "robot_b"):
            st = adapter.measure(rid)
            print(f"{rid} t={st.time:.3f}\n{_fmt_state(st)}")
        return 0

    with open(args.task) as f:
        task = Point5Task.from_dict(json.load(f))
    ref = build_reference(task, adapter.resolver(args.robot))
    print(f"task {task.name!r} duration={ref.duration:.3f}s "
          f"robot={args.robot}")
    print(f"measured @t=0\n{_fmt_state(adapter.measure(args.robot))}")
    fr = ref.frames(args.dt)
    for k, st in enumerate(fr.states):
        m = st.metadata
        print(f"t={m['t']:.3f} seg={m['seg']} s={m['s']:.2f} "
              f"[{m['keyframe_label']}]\n{_fmt_state(st)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
