"""Probe the physical state right after initial-state perturbation.

Checks whether perturbing joint positions leaves the robot's feet floating in
the air or penetrating the floor (the plugin does not re-ground the robot).
"""
from __future__ import annotations

import argparse

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin


def foot_heights(sim, robot_id: str) -> tuple[float, float]:
    """Return (min_foot_z, root_z) using MuJoCo body positions."""
    import mujoco
    suffix = "_a" if robot_id == "robot_a" else "_b"
    zs = []
    for name in ("foot_left", "foot_right"):
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}{suffix}")
        if bid >= 0:
            zs.append(float(sim.data.xpos[bid][2]))
    root_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"torso{suffix}")
    root_z = float(sim.data.xpos[root_bid][2]) if root_bid >= 0 else float("nan")
    return (min(zs) if zs else float("nan"), root_z)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=0.90)
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()

    full = {
        "joint_pos_delta_max": 0.5,
        "joint_vel_delta_max": 2.0,
        "root_tilt_deg_max": 20.0,
        "root_linear_velocity_delta_max": 2.0,
        "root_angular_velocity_delta_max": 1.0,
    }

    sim = Humanoid21Simulator()
    sim.reset()
    base_min_foot, base_root = foot_heights(sim, "robot_a")
    print(f"unperturbed: min_foot_z={base_min_foot:.4f}  torso_z={base_root:.4f}")
    print()

    configs = {
        "ONLY joint_pos": {"joint_pos_delta_max": full["joint_pos_delta_max"]},
        "ALL dims": full,
    }

    for label, cfg in configs.items():
        scaled = {k: v * args.scale for k, v in cfg.items()}
        plugin = InitialStatePerturbationPlugin(target_robot="robot_a", **scaled)

        min_feet, roots = [], []
        for i in range(args.n):
            sim.reset()
            plugin.set_episode_seed(1000 + i)

            # Replicate plugin logic directly on the simulator core state.
            core = sim.get_core_state()
            new_state = {"robot_a": {k: np.asarray(v).copy() for k, v in core["robot_a"].items()}}
            rng = np.random.RandomState(1000 + i)
            jp = new_state["robot_a"]["joint_pos_norm"]
            new_state["robot_a"]["joint_pos_norm"] = np.clip(
                jp + rng.uniform(-scaled.get("joint_pos_delta_max", 0.0),
                                 scaled.get("joint_pos_delta_max", 0.0) or 1e-12, size=jp.shape),
                -1.0, 1.0,
            ).astype(np.float32)
            sim.set_core_state(new_state)

            mf, rz = foot_heights(sim, "robot_a")
            min_feet.append(mf)
            roots.append(rz)

        min_feet = np.array(min_feet)
        clearance = min_feet - base_min_foot
        print(f"--- {label} (scale={args.scale}) ---")
        print(f"  min_foot_z:  mean={min_feet.mean():.4f}  std={min_feet.std():.4f}  "
              f"min={min_feet.min():.4f}  max={min_feet.max():.4f}")
        print(f"  clearance vs nominal: mean={clearance.mean():+.4f}m  "
              f"p05={np.percentile(clearance,5):+.4f}  p95={np.percentile(clearance,95):+.4f}")
        print(f"  episodes with feet ABOVE nominal by >5cm (airborne): "
              f"{100.0*np.mean(clearance > 0.05):.1f}%")
        print(f"  episodes with feet BELOW nominal by >5cm (penetrating): "
              f"{100.0*np.mean(clearance < -0.05):.1f}%")
        print()


if __name__ == "__main__":
    main()
