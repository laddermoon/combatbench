"""Executor state extraction and working-sim snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from point5.humanoid21.adapter import point5_from_view


@dataclass
class Snapshot:
    """Full MuJoCo data + core states for exact restore."""

    data_arrays: Dict[str, np.ndarray]
    core: Dict[str, Dict[str, np.ndarray]]
    time: float


_FIELDS = ("qpos", "qvel", "ctrl", "act", "xfrc_applied",
           "qacc_warmstart", "qacc", "qfrc_applied")


def capture(sim) -> Snapshot:
    """Snapshot everything the physics engine needs for a deterministic
    restore (positions, velocities, actuator state, warmstart, time)."""
    return Snapshot(
        data_arrays={f: np.array(getattr(sim.data, f)).copy()
                     for f in _FIELDS},
        core={rid: {k: np.asarray(v).copy()
                    for k, v in st.items()}
              for rid, st in sim.get_core_state().items()},
        time=float(sim.data.time))


def restore(sim, snap: Snapshot) -> None:
    """Restore a working sim to an exact captured state."""
    d = sim.data
    for f, arr in snap.data_arrays.items():
        getattr(d, f)[:] = arr
    d.time = snap.time
    mujoco.mj_forward(sim.model, d)


def current_state(sim, robot_id: str) -> Dict[str, Any]:
    """Extract real P5 (local+world), COM kinematics and foot contacts."""
    ds = sim.get_derived_state([robot_id])[robot_id]
    p5 = point5_from_view(ds, robot_id, with_vel=True)
    body_xpos = ds["body_xpos"]
    suffix = {"robot_a": "_a", "robot_b": "_b"}[robot_id]
    core = sim.get_core_state()[robot_id]
    return {
        "robot_id": robot_id,
        "core": {k: np.asarray(v).copy() for k, v in core.items()},
        "p5": p5,
        "body_xpos": dict(body_xpos),
        "ankle_z": (float(body_xpos[f"foot_right{suffix}"][2]),
                    float(body_xpos[f"foot_left{suffix}"][2])),
        "time": float(sim.data.time),
    }


def com_state(sim, robot_id: str) -> np.ndarray:
    """True centroidal state (com, lin_mom, ang_mom) via mass sum."""
    from mocap.humanoid21.centroidal_dynamics.model_spec import (
        build_model_spec)
    from mocap.humanoid21.centroidal_dynamics.state import (
        CentroidalStateExtractor)
    ex = CentroidalStateExtractor(sim, build_model_spec(sim))
    st = ex.state(robot_id)
    return st


def foot_contacts(sim, robot_id: str) -> Dict[str, float]:
    """Foot normal force + extra-body contact count via MuJoCo data."""
    m, d = sim.model, sim.data
    suffix = {"robot_a": "_a", "robot_b": "_b"}[robot_id]
    fr = fl = 0.0
    extra = 0
    nfoot = 0
    foot_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                f"foot_{s}{suffix}") for s in
                ("right", "left")}
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        is_f = g1 in foot_ids or g2 in foot_ids
        frc = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, frc)
        if is_f:
            n = frc[0]
            if g1 in foot_ids:
                fr += n
            else:
                fl += n
            nfoot += 1
        elif frc[0] > 1.0:
            extra += 1
    return {"foot_right_force": fr, "foot_left_force": fl,
            "n_foot_contacts": nfoot, "extra_contacts": extra}


def torso_tilt(core: Dict[str, np.ndarray]) -> float:
    """Angle of torso z-axis from world up, radians."""
    w, x, y, z = np.asarray(core["root_rot"], dtype=np.float64)
    # third column of rotation matrix = local z in world
    uz = 2.0 * (x * z + w * y)
    uz = float(np.clip(1.0 - 2.0 * (x * x + y * y) - 0.0, -1, 1)) \
        if False else float(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    return float(np.arccos(uz))
