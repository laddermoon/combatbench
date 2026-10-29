"""Humanoid21 adapter: convert derived-state views into local Point5State.

Single function, designed for use inside observer plugins. The input is the
per-robot view produced by ``Humanoid21Simulator.get_derived_state()``
(e.g. ``accessor.get_derived_state(['robot_a'])['robot_a']``) — plain dicts
keyed by full body names. No simulator object required.

v2 semantics (DESIGN.md §3): the output positions are in the torso-facing
local frame — origin = ankle XY midpoint, X = torso +x projected to XY,
Y = left, Z = up.  Torso twist relative to the ankle line lives inside
the stored foot coordinates.  Velocities are rotated into the same frame.

Point definitions live in ``landmarks.py`` (feet are ankle centers, not
foot-body origins).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation as _R

from envs.humanoid21.meta import Humanoid21Meta
from point5.contracts import (
    POINT_INDEX,
    POINT_NAMES,
    Point5Error,
    Point5State,
    ankle_mid_xy,
    to_local,
)

from .mechanics import LANDMARKS


def _torso_facing(view: Mapping[str, Any], suffix: str) -> np.ndarray:
    """Unit XY direction of the torso's forward (+x) axis."""
    tq = view.get("body_xquat", {}).get(f"torso{suffix}")
    if tq is None:
        raise Point5Error("view lacks torso quaternion; cannot define "
                          "the local frame")
    tq = np.asarray(tq, dtype=np.float64)
    fwd = _R.from_quat([tq[1], tq[2], tq[3], tq[0]]).as_matrix()[:, 0][:2]
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        raise Point5Error("torso axis vertical; facing undefined")
    return fwd / n


def point5_from_view(view: Mapping[str, Any], robot_id: str, *,
                     with_vel: bool = True) -> Point5State:
    """Convert a ``get_derived_state`` robot view into a ``Point5State``.

    Reads ``body_xpos`` + ``body_xquat`` (required) and
    ``body_linvel_world`` (required when ``with_vel=True``), all keyed by
    full body names such as ``head_a``. Missing keys or non-finite data
    raise ``Point5Error`` — never silently fabricate a point.
    """
    suffix = Humanoid21Meta.ROBOT_SUFFIXES.get(robot_id)
    if suffix is None:
        raise Point5Error(f"unknown robot_id {robot_id!r}")
    bodies = view.get("body_xpos")
    quats = view.get("body_xquat")
    if bodies is None or quats is None:
        raise Point5Error("view lacks 'body_xpos'/'body_xquat'")
    vels = None
    if with_vel:
        vels = view.get("body_linvel_world")
        if vels is None:
            raise Point5Error("view lacks 'body_linvel_world'")

    world = np.empty((len(POINT_NAMES), 3), dtype=np.float64)
    wvel = np.empty_like(world) if with_vel else None
    for i, name in enumerate(POINT_NAMES):
        body, off = LANDMARKS[name]
        full = f"{body}{suffix}"
        try:
            xpos = np.asarray(bodies[full], dtype=np.float64)
            xquat = np.asarray(quats[full], dtype=np.float64)
        except KeyError:
            raise Point5Error(f"view missing body {full!r}") from None
        rot = _R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]])
        world[i] = xpos + rot.as_matrix() @ off
        if wvel is not None:
            try:
                wvel[i] = np.asarray(vels[full], dtype=np.float64)
            except KeyError:
                raise Point5Error(
                    f"body_linvel_world missing {full!r}") from None
    if not np.all(np.isfinite(world)) or (
            wvel is not None and not np.all(np.isfinite(wvel))):
        raise Point5Error(f"{robot_id} has non-finite point data")

    # ---- v2 local frame: origin = ankle XY mid, X = torso facing ----
    origin = ankle_mid_xy(world)
    fwd = _torso_facing(view, suffix)
    yaw_deg = float(np.degrees(np.arctan2(fwd[1], fwd[0])))

    pos = to_local(world, origin, yaw_deg)
    vel = None
    if wvel is not None:
        vel = to_local(wvel, np.zeros(2), yaw_deg)   # rotation only

    return Point5State(
        pos=pos, vel=vel,
        metadata={"robot_id": robot_id, "source": "derived_view",
                  "frame_origin_xy": origin.tolist(),
                  "frame_yaw_deg": yaw_deg})
