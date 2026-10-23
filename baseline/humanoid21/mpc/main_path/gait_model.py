"""Model-derived gait parameters for reduced-order support planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict

import mujoco
import numpy as np


@dataclass(frozen=True)
class GaitModel:
    """Numerical walking envelope measured from a nominal Humanoid21 pose."""

    mass: float
    gravity: float
    control_dt: float
    stance_width: float
    foot_length: float
    foot_width: float
    foot_height: float
    leg_length: float
    nominal_com_height: float
    nominal_step_length: float
    max_step_length: float
    swing_duration: float
    double_support_duration: float
    settle_duration: float
    target_tolerance: float
    source_pose: str = "standing"
    metadata: Dict[str, float] = field(default_factory=dict)


def _ceil_to_dt(value: float, dt: float) -> float:
    return float(np.ceil(value / dt) * dt)


def build_gait_model(
    sim,
    *,
    pose: str = "standing",
    control_dt: float = 0.05,
) -> GaitModel:
    """Extract gait-relevant dimensions from a simulator instance."""
    model, data = sim.model, sim.data
    cache = sim._robot("robot_a")
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)

    foot_points = {}
    foot_lengths = []
    foot_widths = []
    for foot, key in (("left", "foot_left"), ("right", "foot_right")):
        body_id = int(cache["keypoint_body_ids"][key])
        geom_ids = np.where(model.geom_bodyid == body_id)[0]
        if len(geom_ids) == 0:
            raise ValueError(f"no geoms found for {key}")
        centers = np.asarray(data.geom_xpos[geom_ids], dtype=np.float64)
        foot_points[foot] = centers.mean(axis=0)
        radii = np.asarray(model.geom_size[geom_ids, 0], dtype=np.float64)
        half_lengths = np.asarray(model.geom_size[geom_ids, 1], dtype=np.float64)
        foot_lengths.extend((half_lengths + radii) * 2.0)
        if len(geom_ids) > 1:
            span = np.max(centers[:, 1]) - np.min(centers[:, 1])
            foot_widths.append(float(span + 2.0 * np.max(radii)))
        else:
            foot_widths.append(float(2.0 * np.max(radii)))

    left = foot_points["left"]
    right = foot_points["right"]
    pelvis_id = int(cache["keypoint_body_ids"]["pelvis"])
    support_z = float((left[2] + right[2]) * 0.5)
    leg_length = float(data.xipos[pelvis_id, 2] - support_z)
    body_ids = np.asarray(list(cache["body_ids"]), dtype=np.int32)
    mass = float(np.sum(model.body_mass[body_ids]))
    gravity = float(abs(model.opt.gravity[2]))
    stance_width = float(np.linalg.norm(left[:2] - right[:2]))
    foot_length = float(np.max(foot_lengths))
    foot_width = float(np.max(foot_widths))
    nominal_com_height = float(data.subtree_com[int(cache["root_body_id"]), 2])

    nominal_step_length = min(0.45 * leg_length, 1.30 * foot_length)
    max_step_length = min(0.65 * leg_length, 1.80 * foot_length)
    time_scale = np.sqrt(max(leg_length, 1e-6) / gravity)
    swing_duration = _ceil_to_dt(1.20 * time_scale, control_dt)
    double_support_duration = _ceil_to_dt(0.50 * time_scale, control_dt)
    settle_duration = _ceil_to_dt(2.00 * time_scale, control_dt)

    return GaitModel(
        mass=mass,
        gravity=gravity,
        control_dt=float(control_dt),
        stance_width=stance_width,
        foot_length=foot_length,
        foot_width=foot_width,
        foot_height=support_z,
        leg_length=leg_length,
        nominal_com_height=nominal_com_height,
        nominal_step_length=float(nominal_step_length),
        max_step_length=float(max_step_length),
        swing_duration=swing_duration,
        double_support_duration=double_support_duration,
        settle_duration=settle_duration,
        target_tolerance=max(0.05, control_dt),
        source_pose=pose,
        metadata={
            "time_scale_s": float(time_scale),
            "foot_geom_count": float(len(foot_lengths)),
        },
    )


@lru_cache(maxsize=8)
def default_gait_model(
    pose: str = "standing",
    control_dt: float = 0.05,
) -> GaitModel:
    """Build a nominal model once using a fresh, uncontrolled simulator."""
    from envs.humanoid21.simulator import Humanoid21Simulator

    sim = Humanoid21Simulator(initial_pose_a=pose, initial_pose_b=pose)
    try:
        sim.reset(seed=0)
        return build_gait_model(sim, pose=pose, control_dt=control_dt)
    finally:
        sim.close()
