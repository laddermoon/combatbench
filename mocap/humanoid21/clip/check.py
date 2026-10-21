"""CoreClip ground-penetration checker.

For every frame, kinematically set the clip state into the simulator
and measure the lowest point of the foot capsule geoms (and,
optionally, solver-level contact penetration vs the ground plane).

This is a GEOMETRIC check on recorded states — it says nothing about
whether the motion is dynamically executable.

Used by the unified CLI: `python3 -m mocap.humanoid21.clip check`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

from .core_clip import CoreClip

GROUND_GEOM = "ground"


def _foot_geom_names(robot: str) -> List[str]:
    s = "a" if robot.endswith("a") else "b"
    return [f"foot1_right_{s}", f"foot2_right_{s}",
            f"foot1_left_{s}", f"foot2_left_{s}"]


def geom_bottom_z(model, data, gid: int) -> float:
    """World-frame lowest z of a geom (capsule/sphere/box)."""
    t = model.geom_type[gid]
    p = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    s = model.geom_size[gid]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        hl, r = float(s[1]), float(s[0])
        axis = R[:, 2]  # capsule axis = local z
        return float(p[2] - abs(axis[2]) * hl - r)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(p[2] - s[0])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        ext = np.abs(R) @ np.abs(s)
        return float(p[2] - ext[2])
    # mesh / other: fall back to origin z (conservative overestimate)
    return float(p[2])


@dataclass
class GroundReport:
    clip_id: str
    robot: str
    min_foot_z: np.ndarray           # (T,) lowest foot point per frame
    worst_geom: List[str]            # (T,) name of the lowest foot geom
    penetration: np.ndarray          # (T,) = max(0, -min_foot_z)
    contact_pen: np.ndarray          # (T,) solver penetration (dist<0), m
    per_geom: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def max_penetration(self) -> float:
        return float(self.penetration.max()) if len(self.penetration) else 0.0

    @property
    def bad_frames(self) -> List[int]:
        return [int(k) for k in np.nonzero(self.penetration > 0.002)[0]]

    def summary(self) -> str:
        lines = [
            f"clip={self.clip_id} robot={self.robot} "
            f"T={len(self.penetration)}",
            f"  max penetration: {self.max_penetration * 100:.2f} cm"
            f"   solver max: {self.contact_pen.max() * 100:.2f} cm",
            f"  frames penetrating >2mm: "
            f"{self.bad_frames if self.bad_frames else 'none'}",
        ]
        if self.bad_frames:
            for k in self.bad_frames:
                lines.append(
                    f"    k={k:3d}  t={k * 0.05:.2f}s  "
                    f"pen={self.penetration[k] * 100:5.2f}cm  "
                    f"geom={self.worst_geom[k]}")
        return "\n".join(lines)


def ground_report(clip: CoreClip, robot: str = "robot_a",
                  sim=None) -> GroundReport:
    """Measure per-frame foot penetration depth below z=0."""
    own = sim is None
    if own:
        from envs.humanoid21.simulator import Humanoid21Simulator
        sim = Humanoid21Simulator()
    model, data = sim.model, sim.data
    names = _foot_geom_names(robot)
    gids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in names}
    ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   GROUND_GEOM)
    robot_gids = set(gids.values())

    min_z = np.zeros(clip.T)
    worst = [""] * clip.T
    cpen = np.zeros(clip.T)
    per_geom = {n: np.zeros(clip.T) for n in names}
    for k in range(clip.T):
        sim.set_core_state({robot: clip.state(k)})
        lows = {n: geom_bottom_z(model, data, g) for n, g in gids.items()}
        for n, z in lows.items():
            per_geom[n][k] = z
        wn = min(lows, key=lows.get)
        min_z[k], worst[k] = lows[wn], wn
        # solver-level penetration for foot<->ground contacts
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == ground_gid and c.geom2 in robot_gids) or \
               (c.geom2 == ground_gid and c.geom1 in robot_gids):
                cpen[k] = max(cpen[k], -float(c.dist))
    if own:
        sim.close()
    return GroundReport(clip.clip_id, robot, min_z, worst,
                        np.clip(-min_z, 0, None), cpen, per_geom)


