"""Humanoid21 adapter: convert simulator data into Point5State.

Single function, designed for use inside observer plugins: call it with the
simulator (or anything exposing ``._robots``/``.data``, e.g. an
``IDataAccessor``) and get a measured ``Point5State`` back.
"""
from __future__ import annotations

import numpy as np

from mocap.point5.contracts import (
    POINT_NAMES,
    Point5Error,
    Point5State,
)


def measure_point5(sim, robot_id: str, *,
                   with_vel: bool = True) -> Point5State:
    """World-frame positions (and optional velocities) of the five points.

    Reads the simulator's current kinematics buffers; call after
    ``physical_step()``/``reset()`` when they are fresh. No refresh is
    performed here — ``mj_forward`` would consume pending applied loads.
    """
    tables = sim._robots.get(robot_id)
    if tables is None:
        raise Point5Error(f"unknown robot_id {robot_id!r}")
    kp = tables["keypoint_body_ids"]
    pos = np.stack([np.asarray(sim.data.xpos[kp[n]])
                    for n in POINT_NAMES]).astype(np.float64)
    if not np.all(np.isfinite(pos)):
        raise Point5Error(f"{robot_id} has non-finite point pos")
    vel = None
    if with_vel:
        vel = np.stack([np.asarray(sim.data.cvel[kp[n], 3:6])
                        for n in POINT_NAMES]).astype(np.float64)
    return Point5State(
        pos=pos.copy(), vel=None if vel is None else vel.copy(),
        time=float(sim.data.time), frame="world",
        metadata={"robot_id": robot_id, "source": "humanoid21_sim"})
