"""Humanoid21 adapter: convert derived-state views into Point5State.

Single function, designed for use inside observer plugins. The input is the
per-robot view produced by ``Humanoid21Simulator.get_derived_state()``
(e.g. ``accessor.get_derived_state(['robot_a'])['robot_a']``) — plain dicts
keyed by full body names. No simulator object required.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from envs.humanoid21.meta import Humanoid21Meta
from mocap.point5.contracts import (
    POINT_NAMES,
    Point5Error,
    Point5State,
)


def point5_from_view(view: Mapping[str, Any], robot_id: str, *,
                     with_vel: bool = True) -> Point5State:
    """Convert a ``get_derived_state`` robot view into a ``Point5State``.

    Reads ``body_xpos`` (required) and ``body_linvel_world`` (required when
    ``with_vel=True``), both keyed by full body names such as ``head_a``.
    Missing keys or non-finite data raise ``Point5Error`` — never silently
    fabricate a point.
    """
    suffix = Humanoid21Meta.ROBOT_SUFFIXES.get(robot_id)
    if suffix is None:
        raise Point5Error(f"unknown robot_id {robot_id!r}")
    bodies = view.get("body_xpos")
    if bodies is None:
        raise Point5Error("view lacks 'body_xpos'")
    vels = None
    if with_vel:
        vels = view.get("body_linvel_world")
        if vels is None:
            raise Point5Error("view lacks 'body_linvel_world'")

    pos = np.empty((len(POINT_NAMES), 3), dtype=np.float64)
    vel = np.empty_like(pos) if with_vel else None
    for i, name in enumerate(POINT_NAMES):
        full = f"{name}{suffix}"
        try:
            pos[i] = np.asarray(bodies[full], dtype=np.float64)
        except KeyError:
            raise Point5Error(f"body_xpos missing {full!r}") from None
        if vel is not None:
            try:
                vel[i] = np.asarray(vels[full], dtype=np.float64)
            except KeyError:
                raise Point5Error(
                    f"body_linvel_world missing {full!r}") from None
    if not np.all(np.isfinite(pos)) or (
            vel is not None and not np.all(np.isfinite(vel))):
        raise Point5Error(f"{robot_id} has non-finite point data")

    return Point5State(
        pos=pos, vel=vel, frame="world",
        metadata={"robot_id": robot_id, "source": "derived_view"})
