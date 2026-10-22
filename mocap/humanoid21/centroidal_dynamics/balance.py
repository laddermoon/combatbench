"""Momentum-balance calculations for recorded physics-step windows."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .contracts import (
    BalanceStep,
    CentroidalState,
    LoadSnapshot,
    SOURCE_XFRC_APPLIED,
    UnsupportedGeneralizedLoad,
)


def _contact_loads(snapshot: Optional[LoadSnapshot]):
    if snapshot is None:
        return ()
    return tuple(l for l in snapshot.spatial_loads
                 if l.source_type.startswith("contact_"))


def _step_loads(snapshot: Optional[LoadSnapshot]):
    """Loads known to act over the step rather than as endpoint impulses."""
    if snapshot is None:
        return ()
    return tuple(l for l in snapshot.spatial_loads
                 if l.source_type == SOURCE_XFRC_APPLIED)


def _check_generalized(snapshot: Optional[LoadSnapshot]) -> None:
    if snapshot is not None and snapshot.generalized_loads:
        raise UnsupportedGeneralizedLoad(
            "momentum balance cannot consume qfrc_applied loads without a "
            "validated spatial-wrench conversion")


def momentum_balance(
    state0: CentroidalState,
    state1: CentroidalState,
    snapshot0: LoadSnapshot,
    snapshot1: Optional[LoadSnapshot] = None,
    *,
    mode: str = "trapezoid",
) -> BalanceStep:
    """Compute endpoint or trapezoidal momentum-balance residuals.

    - ``mode="pre"`` uses contact forces at ``t0``.
    - ``mode="post"`` uses contact forces at ``t1`` and requires
      ``snapshot1``.
    - ``mode="trapezoid"`` averages contact endpoint forces. ``xfrc_applied``
      is always taken from ``snapshot0`` because it acts during the step and is
      cleared by ``physical_step()``.
    """
    if state0.robot_id != state1.robot_id:
        raise ValueError("state0/state1 robot_id mismatch")
    if snapshot0.robot_id != state0.robot_id:
        raise ValueError("snapshot0 robot_id mismatch")
    if snapshot1 is not None and snapshot1.robot_id != state0.robot_id:
        raise ValueError("snapshot1 robot_id mismatch")
    dt = state1.time - state0.time
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"state times must satisfy t1>t0, got dt={dt}")
    _check_generalized(snapshot0)
    _check_generalized(snapshot1)

    gravity = np.asarray(snapshot0.gravity, dtype=np.float64)
    lin_impulse = state0.mass * gravity * dt
    ang_impulse = np.zeros(3, dtype=np.float64)

    if mode == "pre":
        endpoints = ((snapshot0, state0),)
        contact_weight = 1.0
    elif mode == "post":
        if snapshot1 is None:
            raise ValueError("mode='post' requires snapshot1")
        endpoints = ((snapshot1, state1),)
        contact_weight = 1.0
    elif mode == "trapezoid":
        if snapshot1 is None:
            raise ValueError("mode='trapezoid' requires snapshot1")
        endpoints = ((snapshot0, state0), (snapshot1, state1))
        contact_weight = 0.5
    else:
        raise ValueError(f"unknown balance mode {mode!r}")

    for snap, ref_state in endpoints:
        for load in _contact_loads(snap):
            lin_impulse += contact_weight * load.force_world * dt
            ang_impulse += contact_weight * load.moment_about(
                ref_state.com_pos) * dt

    for load in _step_loads(snapshot0):
        lin_impulse += load.force_world * dt
        ang_impulse += load.moment_about(state0.com_pos) * dt

    lin_residual = (state1.lin_mom - state0.lin_mom) - lin_impulse
    ang_residual = (state1.ang_mom - state0.ang_mom) - ang_impulse
    return BalanceStep(
        robot_id=state0.robot_id,
        t0=state0.time,
        t1=state1.time,
        state0=state0,
        state1=state1,
        lin_impulse=lin_impulse,
        ang_impulse=ang_impulse,
        lin_residual=lin_residual,
        ang_residual=ang_residual,
        diagnostics={
            "mode": mode,
            "contact_loads0": len(_contact_loads(snapshot0)),
            "contact_loads1": len(_contact_loads(snapshot1)),
            "step_loads0": len(_step_loads(snapshot0)),
            "gravity_impulse_norm":
                float(np.linalg.norm(state0.mass * gravity * dt)),
        },
    )
