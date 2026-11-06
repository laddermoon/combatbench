"""Diagnostic: run ExecutorPolicy in a manual sim loop and log per-step
balance metrics.  Isolates whether Algorithm 2 fails on the trivial
"stand" intent (injection/cost problem) vs. only on real intent
(tracking problem).

Usage:
    PYTHONPATH=. python3 -B .../diag_e2e.py [stand|punch] [steps]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator
from point5.contracts import Keyframe, P5Clip, Transfer
from point5.humanoid21.stand_punch import stand_p5, plan_stand_punch
from point5.humanoid21.executor.tools.com_solver import com_estimate
from point5.humanoid21.executor.tools.obs_decoder import ObsDecoder
from point5.humanoid21.executor.e2e_policy import ExecutorPolicy


def make_clip(kind: str) -> P5Clip:
    if kind == "stand":
        p = stand_p5()
        return P5Clip(name="stand",
                      keyframes=(Keyframe(pos=p),
                                 Keyframe(pos=p,
                                          transfer=Transfer(0, 0, 0, 5.0))))
    plan = plan_stand_punch(np.array([1.05, 0.0, 1.15]), "left")
    assert plan.task is not None
    return plan.task


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "stand"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    sim = Humanoid21Simulator(initial_distance=6.0,
                              initial_pose_a="standing")
    sim.reset(seed=0)
    pol = ExecutorPolicy(robot_id="robot_a", clip=make_clip(kind))
    dec = ObsDecoder("robot_a")

    print(f"kind={kind} steps={n_steps}")
    print("step | com_z  tilt_deg | a_norm a_max | lm1_dist | ms")
    for t in range(n_steps):
        obs = sim.get_observation()["robot_a"]
        t0 = time.time()
        act, _ = pol.act(obs)
        ms = (time.time() - t0) * 1e3

        # real sim step (25 phy substeps inside)
        sim.set_action({"robot_a": np.asarray(act, np.float32),
                        "robot_b": np.zeros(21, np.float32)})
        for _ in range(25):
            sim.physical_step()

        obs2 = sim.get_observation()["robot_a"]
        p5, _ = dec.decode_p5_local(obs2)
        com = com_estimate(p5)
        # tilt: gravity in body frame, obs[42:45]; upright => -z
        g = obs2[42:45]
        tilt = np.degrees(np.arccos(
            np.clip(-g[2] / (np.linalg.norm(g) + 1e-9), -1, 1)))
        # left-foot forward travel + left-hand reach (punch intent)
        fl_x = p5[4, 0]                     # left ankle x (frame0 local)
        hl_reach = p5[2, 0] - p5[0, 0]      # left hand vs head, fwd dist
        print(f"{t:4d} | {com[2]:.3f} {tilt:6.1f} | "
              f"{np.linalg.norm(act):.2f} {np.abs(act).max():.2f} | "
              f"fl_x={fl_x:+.3f} reach={hl_reach:+.3f} | {ms:5.0f}",
              flush=True)
        if com[2] < 0.6 or tilt > 60:
            print("FALLEN"); break


if __name__ == "__main__":
    main()
