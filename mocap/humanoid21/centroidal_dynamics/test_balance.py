"""P3 momentum-balance unit tests."""
import numpy as np
import pytest

from mocap.humanoid21.centroidal_dynamics.balance import momentum_balance
from mocap.humanoid21.centroidal_dynamics.contracts import (
    CentroidalState,
    ExternalLoad,
    GeneralizedLoad,
    LoadSnapshot,
    SOURCE_CONTACT_GROUND,
    SOURCE_XFRC_APPLIED,
    UnsupportedGeneralizedLoad,
)


def _state(t, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0),
           com=(0.0, 0.0, 1.0)):
    return CentroidalState(
        robot_id="robot_a", time=t, mass=10.0,
        com_pos=np.asarray(com), lin_mom=np.asarray(lin),
        ang_mom=np.asarray(ang))


def _snapshot(loads=(), generalized=()):
    return LoadSnapshot(
        robot_id="robot_a", time=0.0,
        spatial_loads=tuple(loads),
        generalized_loads=tuple(generalized),
        gravity=np.array([0.0, 0.0, -9.81]))


def test_gravity_and_xfrc_impulse_are_accounted():
    dt = 0.002
    s0 = _state(0.0)
    s1 = _state(dt, lin=np.array([0.4, 0.0, -0.1962]))
    load = ExternalLoad(
        source_type=SOURCE_XFRC_APPLIED, robot_id="robot_a",
        point_world=np.array([0.0, 0.0, 1.0]),
        force_world=np.array([200.0, 0.0, 0.0]),
        torque_world=np.array([0.0, 0.2, 0.0]))
    # Need identical snapshots because trapezoid requires both endpoints.
    snap = _snapshot([load])
    b = momentum_balance(s0, s1, snap, snap, mode="pre")
    assert np.allclose(b.lin_impulse,
                       np.array([0.4, 0.0, -0.1962]))
    assert np.allclose(b.ang_impulse, np.array([0.0, 0.0004, 0.0]))
    assert np.allclose(b.lin_residual, np.zeros(3))


def test_trapezoid_averages_only_endpoint_contact_forces():
    dt = 0.002
    s0 = _state(0.0)
    s1 = _state(dt)
    c0 = ExternalLoad(
        source_type=SOURCE_CONTACT_GROUND, robot_id="robot_a",
        point_world=np.zeros(3), force_world=np.array([0.0, 0.0, 100.0]))
    c1 = ExternalLoad(
        source_type=SOURCE_CONTACT_GROUND, robot_id="robot_a",
        point_world=np.zeros(3), force_world=np.array([0.0, 0.0, 300.0]))
    b = momentum_balance(s0, s1, _snapshot([c0]), _snapshot([c1]),
                         mode="trapezoid")
    # 200 N average over 2 ms, plus gravity.
    assert np.allclose(b.lin_impulse,
                       np.array([0.0, 0.0, 0.4 - 10.0 * 9.81 * dt]))


def test_generalized_load_cannot_be_silently_balanced():
    s0 = _state(0.0)
    s1 = _state(0.002)
    gen = GeneralizedLoad(robot_id="robot_a",
                          dof_indices=np.array([0]),
                          generalized_force=np.array([1.0]))
    with pytest.raises(UnsupportedGeneralizedLoad):
        momentum_balance(s0, s1, _snapshot(generalized=[gen]),
                         _snapshot(), mode="trapezoid")


def test_balance_requires_positive_dt():
    with pytest.raises(ValueError, match="t1>t0"):
        momentum_balance(_state(0.0), _state(0.0),
                         _snapshot(), _snapshot())
