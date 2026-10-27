"""Extract measured Point5State and build resolvers from Humanoid21Simulator.

Boundary: everything point5-side stays simulator-unaware; this module is the
single place that maps MjData fields into point5 types.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mocap.point5.contracts import (
    POINT_NAMES,
    Point5Error,
    Point5State,
    StaticStartResolver,
)

ROBOT_IDS = ("robot_a", "robot_b")


class Point5AdapterError(Point5Error):
    """Simulator state cannot produce a valid point5 quantity."""


class Humanoid21Point5:
    """Adapter bound to a live ``Humanoid21Simulator``."""

    def __init__(self, sim) -> None:
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self._kp = {}
        self._root = {}
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
            self._root[rid] = int(tables["root_body_id"])

    # -- measurements --------------------------------------------------------

    def measure(self, robot_id: str, *,
                with_vel: bool = True) -> Point5State:
        """World-frame positions (and optional velocities) of the five points.

        Reads the simulator's current kinematics buffers; call after
        ``physical_step()``/``reset()`` when they are fresh. No refresh is
        performed here — ``mj_forward`` would consume pending applied loads.
        """
        kp = self._kp[robot_id]
        pos = np.stack([np.asarray(self.data.xpos[kp[n]])
                        for n in POINT_NAMES]).astype(np.float64)
        vel = None
        if with_vel:
            vel = np.stack([np.asarray(self.data.cvel[kp[n], 3:6])
                            for n in POINT_NAMES]).astype(np.float64)
        if not np.all(np.isfinite(pos)):
            raise Point5AdapterError(f"{robot_id} has non-finite point pos")
        return Point5State(
            pos=pos.copy(), vel=None if vel is None else vel.copy(),
            time=float(self.data.time), frame="world",
            metadata={"robot_id": robot_id, "source": "humanoid21_sim"})

    def root_pose(self, robot_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Root body world position and rotation (world <- root)."""
        bid = self._root[robot_id]
        return (np.asarray(self.data.xpos[bid], dtype=np.float64).copy(),
                np.asarray(self.data.xmat[bid], dtype=np.float64)
                .reshape(3, 3).copy())

    # -- resolver ------------------------------------------------------------

    def resolver(self, robot_id: str, *,
                 initial: Optional[Point5State] = None,
                 root_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 ) -> StaticStartResolver:
        """Resolver against the sim's *current* state (or supplied snapshot).

        Pass explicit ``initial``/``root_pose`` to resolve a task against a
        recorded start state instead of the live one.
        """
        if initial is None:
            initial = self.measure(robot_id)
        if root_pose is None:
            root_pose = self.root_pose(robot_id)
        return StaticStartResolver(initial=initial,
                                   root_pos=root_pose[0],
                                   root_rot=root_pose[1])
