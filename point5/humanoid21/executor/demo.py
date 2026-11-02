"""CLI: run Executor on stand_punch from standing pose.

    PYTHONPATH=. python3 -m point5.humanoid21.executor.demo \
        --hand left --target 1.05 0 1.15 --seed 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from point5.contracts import ankle_mid_xy, to_local
from point5.humanoid21.stand_punch import plan_stand_punch, stand_p5
from point5.trajectory import keyframe_world
from point5.humanoid21.executor import Executor, ExecutorConfig
from point5.humanoid21.executor.reference import build_stand_clip
from point5.humanoid21.executor.state import (
    capture, current_state, com_state)


def run(hand="left", target=(1.05, 0.0, 1.15), seed=0, out=None,
        max_steps=160, cfg=None):
    from envs.humanoid21.simulator import Humanoid21Simulator
    from baseline.humanoid21.hierarchy.p5_world import f1_from_state

    cfg = cfg or ExecutorConfig(seed=seed)
    env = Humanoid21Simulator(initial_distance=6.0,
                              initial_pose_a="standing",
                              initial_pose_b="standing")
    env.reset(seed=seed)
    wsim = Humanoid21Simulator(initial_distance=6.0,
                               initial_pose_a="standing",
                               initial_pose_b="standing")
    wsim.reset(seed=seed)
    try:
        cur = current_state(env, "robot_a")
        p5 = cur["p5"]
        f1 = f1_from_state(p5)
        # build submitted clip: plan in local frame, prepend standing start
        plan = plan_stand_punch(np.asarray(target, float), hand,
                                initial=p5)
        assert plan.reachable, plan.reason
        clip = build_stand_clip(plan.task, p5.pos)

        ex = Executor(wsim, cfg)
        sub = ex.submit(cur, clip, f1, hand=hand,
                        target=np.asarray(target, float))
        wpts = keyframe_world(clip, f1)

        log = {"hand": hand, "target": list(map(float, target)),
               "seed": seed, "accepted": sub, "events": ex.events,
               "steps": []}
        hist = {"t": [], "p5w": [], "com": [], "a": []}
        t0 = time.time()
        status = "RUNNING"
        for k in range(max_steps):
            cur = current_state(env, "robot_a")
            cur["sim"] = env
            # sync working sim to the real env state each cycle
            snap = capture(env)
            wsim.set_core_state({rid: dict(st) for rid, st in
                                 snap.core.items()})
            snap_w = capture(wsim)
            r = ex.step(cur, snapshot=snap_w)
            status = r.status
            env.set_action({"robot_a": r.action.astype(np.float32),
                            "robot_b": np.zeros(21, np.float32)})
            for _ in range(25):
                env.physical_step()
            post = current_state(env, "robot_a")
            cst = com_state(env, "robot_a")
            hist["t"].append(post["time"])
            hist["a"].append(r.action.tolist())
            hist["com"].append(np.asarray(cst.com_pos).tolist())
            hist["p5w"].append(
                __import__("baseline.humanoid21.hierarchy.p5_world",
                           fromlist=["p5_to_world", "f1_from_state"])
                .p5_to_world(
                    post["p5"].pos,
                    __import__(
                        "baseline.humanoid21.hierarchy.p5_world",
                        fromlist=["f1_from_state"])
                    .f1_from_state(post["p5"])).tolist())
            log["steps"].append({"k": k, "status": r.status,
                                 "diag": r.diag})
            if r.status in ("FAILED",):
                break
        log["final_status"] = status
        log["wall_s"] = time.time() - t0
        log["events"] = ex.events
        if out:
            od = Path(out)
            od.mkdir(parents=True, exist_ok=True)
            (od / "report.json").write_text(json.dumps(
                log, indent=2, default=float))
            np.savez(od / "traj.npz",
                     t=np.array(hist["t"]),
                     p5w=np.array(hist["p5w"]),
                     com=np.array(hist["com"]),
                     a=np.array(hist["a"]),
                     ref_wpts=wpts)
        return log
    finally:
        env.close()
        wsim.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="left", choices=["left", "right"])
    ap.add_argument("--target", nargs=3, type=float,
                    default=[1.05, 0.0, 1.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=160)
    a = ap.parse_args()
    log = run(a.hand, tuple(a.target), a.seed, a.out, a.max_steps)
    print(json.dumps({k: log[k] for k in
                      ("final_status", "wall_s", "events")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
