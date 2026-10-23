"""Run the observation-only standing MPC baseline.

Example:
    PYTHONPATH=. python3 baseline/humanoid21/mpc/run_standing.py --steps 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.simulator import Humanoid21Simulator

from baseline.humanoid21.mpc.standing_mpc import StandingMPCPolicy


def _uprightness(sim: Humanoid21Simulator, robot_id: str) -> float:
    cache = sim._robot(robot_id)
    quat = sim.data.qpos[cache["root_qpos_adr"] + 3:cache["root_qpos_adr"] + 7]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return float(rot[2, 2])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    sim = Humanoid21Simulator(
        initial_distance=args.initial_distance,
        initial_pose_a="standing",
        initial_pose_b="standing",
    )
    sim.reset(seed=args.seed)
    policy = StandingMPCPolicy(
        robot_id=args.robot_id,
        horizon_control_steps=args.horizon,
        replan_interval=args.replan_interval,
        seed=args.seed + 1,
    )
    opponent_id = "robot_b" if args.robot_id == "robot_a" else "robot_a"
    opponent_action = sim.get_action()[opponent_id]

    heights = []
    uprightness = []
    plan_times = []
    replans = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        action, extra = policy.act(
            sim.get_observation()[args.robot_id], want_extra=True
        )
        sim.set_action({args.robot_id: action, opponent_id: opponent_action})
        for _ in range(25):
            sim.physical_step()
        core = sim.get_core_state()[args.robot_id]
        heights.append(float(core["root_pos"][2]))
        uprightness.append(_uprightness(sim, args.robot_id))
        if extra and extra.get("event") == "replan":
            plan_times.append(float(extra["plan_wall_time_s"]))
            replans.append({
                "step": step,
                "gains": extra["active_gains"],
                "score": extra["best_score"],
                "wall_time_s": extra["plan_wall_time_s"],
            })

    elapsed = time.perf_counter() - t0
    result = {
        "steps": int(args.steps),
        "seconds": float(args.steps * 0.05),
        "robot_id": args.robot_id,
        "horizon_control_steps": int(args.horizon),
        "replan_interval": int(args.replan_interval),
        "min_height": float(np.min(heights)),
        "final_height": float(heights[-1]),
        "min_uprightness": float(np.min(uprightness)),
        "final_uprightness": float(uprightness[-1]),
        "wall_time_s": float(elapsed),
        "plan_calls": len(plan_times),
        "plan_time_mean_s": (float(np.mean(plan_times)) if plan_times else 0.0),
        "plan_time_p95_s": (float(np.percentile(plan_times, 95))
                            if plan_times else 0.0),
        "replans": replans if args.verbose else replans[:10],
    }
    policy.close()
    sim.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200,
                        help="control steps; 200 = 10 seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-id", choices=("robot_a", "robot_b"),
                        default="robot_a")
    parser.add_argument("--horizon", type=int, default=40,
                        help="candidate rollout control steps; 40 = 2 seconds")
    parser.add_argument("--replan-interval", type=int, default=10,
                        help="reuse selected gains for this many control steps")
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
