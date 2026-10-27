"""Humanoid21 adapter: convert simulator data into Point5State.

Single responsibility: MjData -> Point5State. Nothing else lives here —
task resolution, scoring, observation building are all downstream concerns.
"""
from __future__ import annotations

import numpy as np

from mocap.point5.contracts import (
    POINT_NAMES,
    Point5Error,
    Point5State,
)

ROBOT_IDS = ("robot_a", "robot_b")


class Point5AdapterError(Point5Error):
    """Simulator state cannot produce a valid point5 quantity."""


class Humanoid21Point5:
    """Extract measured ``Point5State`` from a live ``Humanoid21Simulator``."""

    def __init__(self, sim) -> None:
        self.sim = sim
        self._kp = {}
        for rid in ROBOT_IDS:
            tables = sim._robots.get(rid)
            if tables is None:
                raise Point5AdapterError(f"simulator lacks tables for {rid}")
            kp = tables["keypoint_body_ids"]
            missing = [n for n in POINT_NAMES if n not in kp]
            if missing:
                raise Point5AdapterError(
                    f"{rid} keypoint_body_ids missing {missing}")
            self._kp[rid] = {n: int(kp[n]) for n in POINT_NAMES}

    def measure(self, robot_id: str, *,
                with_vel: bool = True) -> Point5State:
        """World-frame positions (and optional velocities) of the five points.

        Reads the simulator's current kinematics buffers; call after
        ``physical_step()``/``reset()`` when they are fresh. No refresh is
        performed here — ``mj_forward`` would consume pending applied loads.
        """
        kp = self._kp.get(robot_id)
        if kp is None:
            raise Point5AdapterError(f"unknown robot_id {robot_id!r}")
        pos = np.stack([np.asarray(self.sim.data.xpos[kp[n]])
                        for n in POINT_NAMES]).astype(np.float64)
        if not np.all(np.isfinite(pos)):
            raise Point5AdapterError(f"{robot_id} has non-finite point pos")
        vel = None
        if with_vel:
            vel = np.stack([np.asarray(self.sim.data.cvel[kp[n], 3:6])
                            for n in POINT_NAMES]).astype(np.float64)
        return Point5State(
            pos=pos.copy(), vel=None if vel is None else vel.copy(),
            time=float(self.sim.data.time), frame="world",
            metadata={"robot_id": robot_id, "source": "humanoid21_sim"})
