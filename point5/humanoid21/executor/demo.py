"""E7 demo: execute a stand-punch clip through the real simulator.

    PYTHONPATH=. python3 -m point5.humanoid21.executor.demo \
        --hand left --target 1.05 0 1.15 --seed 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from point5.humanoid21.executor.clip_cache import (
    build_stand_punch_clip)
from point5.humanoid21.executor.contracts import (
    ExecStatus, ExecutorConfig)
from point5.humanoid21.executor.executor import Executor
from point5.humanoid21.executor.state import capture_current


def run_case(hand: str, target, seed: int = 0, out=None,
             max_steps: int = 140, cfg: ExecutorConfig = None):
    from envs.humanoid21.simulator import Humanoid21Simulator
    cfg = cfg or ExecutorConfig(seed=seed)
    env = Humanoid21Simulator(initial_distance=6.0,
                              initial_pose_a="standing")
    env.reset(seed=seed)
    wsim = Humanoid21Simulator(initial_distance=6.0,
                               initial_pose_a="standing")
    wsim.reset(seed=seed)
    ex = Executor(wsim, cfg)
    t0 = time.time()
    log = {"hand": hand, "target": list(map(float, target)),
           "seed": seed, "steps": []}
    hist = {"p5w": [], "com": [], "a": [], "t": []}
    try:
        cur = capture_current(env)
        clip = build_stand_punch_clip(cur.p5, np.asarray(target, float),
                                      hand)
        log["submit"] = ex.submit(clip, cur)
        status = ExecStatus.RUNNING
        for k in range(max_steps):
            cur = capture_current(env)
            r = ex.step(cur)
            status = r.status
            env.set_action({"robot_a": r.action.astype(np.float32),
                            "robot_b": np.zeros(21, np.float32)})
            for _ in range(cfg.phy_steps):
                env.physical_step()
            post = capture_current(env, with_snapshot=False)
            hist["t"].append(post.time)
            hist["p5w"].append(post.p5_world.tolist())
            hist["com"].append(post.com_pos.tolist())
            hist["a"].append(r.action.tolist())
            log["steps"].append({"k": k, "status": r.status.value,
                                 "diag": r.diag})
            if r.status == ExecStatus.FAILED:
                break
        log["final_status"] = status.value
        log["events"] = ex.events
        log["wall_s"] = time.time() - t0
        if out:
            od = Path(out)
            od.mkdir(parents=True, exist_ok=True)
            (od / "report.json").write_text(
                json.dumps(log, indent=2, default=float))
            np.savez(od / "traj.npz",
                     t=np.array(hist["t"]), p5w=np.array(hist["p5w"]),
                     com=np.array(hist["com"]),
                     a=np.array(hist["a"]))
        return log
    finally:
        env.close()
        wsim.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="left",
                    choices=["left", "right"])
    ap.add_argument("--target", nargs=3, type=float,
                    default=[1.05, 0.0, 1.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=140)
    a = ap.parse_args()
    log = run_case(a.hand, tuple(a.target), a.seed, a.out,
                   a.max_steps)
    print(json.dumps({"final": log["final_status"],
                      "wall_s": log["wall_s"],
                      "events": log["events"]},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
