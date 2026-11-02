"""E1: current-state extraction, sim snapshots, working-sim sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from point5.contracts import POINT_INDEX, to_world
from point5.humanoid21.adapter import point5_from_view
from .contracts import StateError

# mjData fields required for a bitwise-identical continuation.  xfrc is
# cleared every substep by the simulator, but restore anyway.
DATA_FIELDS = ("qpos", "qvel", "ctrl", "act", "xfrc_applied",
               "qfrc_applied", "qacc", "qacc_warmstart")

FOOT_POINT = {"right": POINT_INDEX["foot_right"],
              "left": POINT_INDEX["foot_left"]}


@dataclass
class Current:
    """Measured robot state at one control instant."""

    time: float
    robot_id: str
    core: Dict[str, np.ndarray]          # get_core_state()[robot_id]
    p5: Any                              # Point5State (local + frame meta)
    frame_pose: tuple                    # (origin_xy(2,), yaw_deg)
    p5_world: np.ndarray                 # (5,3)
    p5_vel_world: np.ndarray             # (5,3) — from adapter local vel
    com_pos: np.ndarray                  # (3,) real, mass-weighted
    com_vel: np.ndarray                  # (3,)
    ang_mom: np.ndarray                  # (3,)
    mass: float
    torso_tilt: float                    # rad from world up
    contacts: Dict[str, Any]
    snapshot: "Snapshot"                 # full sim snapshot this instant


@dataclass
class Snapshot:
    """Everything needed to resume deterministically."""

    arrays: Dict[str, np.ndarray]
    time: float
    target_pos_norm: Dict[str, np.ndarray]
    step_count: int


def snapshot(sim) -> Snapshot:
    d = sim.data
    return Snapshot(
        arrays={f: getattr(d, f).copy() for f in DATA_FIELDS},
        time=float(d.time),
        target_pos_norm={r: np.asarray(v).copy()
                         for r, v in sim._target_pos_norm.items()},
        step_count=int(getattr(sim, "_step_count", 0)))


def restore(sim, snap: Snapshot) -> None:
    d = sim.data
    for f, arr in snap.arrays.items():
        getattr(d, f)[:] = arr
    d.time = snap.time
    for r, v in snap.target_pos_norm.items():
        sim._target_pos_norm[r][:] = v
    if hasattr(sim, "_step_count"):
        sim._step_count = snap.step_count
    sim._data_cache.clear()
    mujoco.mj_forward(sim.model, d)


def sync(dst, src) -> Snapshot:
    """Copy the full physics state from ``src`` sim into ``dst``."""
    snap = snapshot(src)
    restore(dst, snap)
    return snap


def _com(sim, robot_id: str):
    from mocap.humanoid21.centroidal_dynamics.state import (
        CentroidalStateExtractor)
    st = CentroidalStateExtractor(sim).state(robot_id)
    return st


def _contacts(sim, robot_id: str) -> Dict[str, Any]:
    """Foot normal forces + count of non-foot, non-ground contacts."""
    m, d = sim.model, sim.data
    sfx = {"robot_a": "_a", "robot_b": "_b"}[robot_id]
    feet = {}
    for side in ("right", "left"):
        gids = set()
        for j in (1, 2):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"foot{j}_{side}{sfx}")
            if gid < 0:
                raise StateError(f"missing geom foot{j}_{side}{sfx}")
            gids.add(gid)
        feet[side] = gids
    out = {"foot_right_force": 0.0, "foot_left_force": 0.0,
           "extra": 0, "extra_force": 0.0}
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        frc = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, frc)
        n = float(frc[0])
        for side, gids in feet.items():
            if g1 in gids or g2 in gids:
                out[f"foot_{side}_force"] += n
                break
        else:
            if n > 1.0:
                out["extra"] += 1
                out["extra_force"] += n
    return out


def _tilt(core: Dict[str, np.ndarray]) -> float:
    w, x, y, z = np.asarray(core["root_rot"], dtype=np.float64)
    uz = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return float(np.arccos(uz))


def capture_current(sim, robot_id: str = "robot_a") -> Current:
    """Full measured state at the current instant."""
    core = {k: np.asarray(v).copy()
            for k, v in sim.get_core_state()[robot_id].items()}
    view = sim.get_derived_state([robot_id])[robot_id]
    p5 = point5_from_view(view, robot_id)
    origin = np.asarray(p5.metadata["frame_origin_xy"], dtype=np.float64)
    yaw = float(p5.metadata["frame_yaw_deg"])
    p5w = to_world(p5.pos, origin, yaw)
    p5vw = np.zeros((5, 3)) if p5.vel is None else \
        to_world(p5.vel, np.zeros(2), yaw)
    st = _com(sim, robot_id)
    return Current(
        time=float(sim.data.time), robot_id=robot_id, core=core, p5=p5,
        frame_pose=(origin, yaw), p5_world=p5w, p5_vel_world=p5vw,
        com_pos=np.asarray(st.com_pos), 
        com_vel=np.asarray(st.lin_mom) / st.mass,
        ang_mom=np.asarray(st.ang_mom), mass=float(st.mass),
        torso_tilt=_tilt(core), contacts=_contacts(sim, robot_id),
        snapshot=snapshot(sim))


def fall_check(cur: Current, cfg) -> Optional[str]:
    """Return failure reason if the robot is down, else None."""
    if cur.torso_tilt > cfg.tilt_fail:
        return "tilt"
    if float(cur.p5_world[POINT_INDEX["head"], 2]) < cfg.head_min_z:
        return "head_low"
    return None
