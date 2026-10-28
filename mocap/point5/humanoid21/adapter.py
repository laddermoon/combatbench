"""Humanoid21 adapter: convert derived-state views into Point5State.

Single function, designed for use inside observer plugins. The input is the
per-robot view produced by ``Humanoid21Simulator.get_derived_state()``
(e.g. ``accessor.get_derived_state(['robot_a'])['robot_a']``) — plain dicts
keyed by full body names. No simulator object required.

Point definitions live in ``landmarks.py`` (feet are ankle centers, not
foot-body origins).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as _R

from envs.humanoid21.meta import Humanoid21Meta
from mocap.point5.contracts import (
    POINT_INDEX,
    POINT_NAMES,
    Point5Error,
    Point5State,
)

from .mechanics import LANDMARKS


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

    pos = np.empty((len(POINT_NAMES), 3), dtype=np.float64)
    vel = np.empty_like(pos) if with_vel else None
    for i, name in enumerate(POINT_NAMES):
        body, off = LANDMARKS[name]
        full = f"{body}{suffix}"
        try:
            xpos = np.asarray(bodies[full], dtype=np.float64)
            xquat = np.asarray(quats[full], dtype=np.float64)
        except KeyError as e:
            raise Point5Error(f"view missing body {full!r}") from None
        rot = _R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]])
        pos[i] = xpos + rot.as_matrix() @ off
        if vel is not None:
            try:
                vel[i] = np.asarray(vels[full], dtype=np.float64)
            except KeyError:
                raise Point5Error(
                    f"body_linvel_world missing {full!r}") from None
    if not np.all(np.isfinite(pos)) or (
            vel is not None and not np.all(np.isfinite(vel))):
        raise Point5Error(f"{robot_id} has non-finite point data")

    torso_yaw = _torso_yaw(view, suffix, pos)
    return Point5State(
        pos=pos, vel=vel, torso_yaw=torso_yaw, frame="world",
        metadata={"robot_id": robot_id, "source": "derived_view"})


def _torso_yaw(view: Mapping[str, Any], suffix: str,
               pos: np.ndarray) -> Optional[float]:
    """Degress of body yaw per DESIGN.md 4.6: baseline = projected
    left->right foot line; angle of torso +x vs stance normal."""
    tq = view.get("body_xquat", {}).get(f"torso{suffix}")
    if tq is None:
        return None
    tq = np.asarray(tq, dtype=np.float64)
    fwd = _R.from_quat([tq[1], tq[2], tq[3], tq[0]]).as_matrix()[:, 0]
    fr = pos[POINT_INDEX["foot_right"], :2]
    fl = pos[POINT_INDEX["foot_left"], :2]
    b = fr - fl
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return None                     # coincident feet: baseline undefined
    b /= nb
    n = np.array([-b[1], b[0]])         # baseline rotated +90 deg = forward
    t = fwd[:2]
    if np.linalg.norm(t) < 1e-9:
        return None                     # torso axis vertical: no yaw
    return float(np.degrees(np.arctan2(t @ b, t @ n)))
