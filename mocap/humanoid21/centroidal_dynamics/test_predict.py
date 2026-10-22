"""P5 discrete centroidal predictor tests."""
import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.centroidal_dynamics.contracts import (
    CentroidalState,
    ExternalLoad,
    GeneralizedLoad,
    LoadSnapshot,
    PredictionDomainViolation,
    SOURCE_CONTACT_WALL,
    SOURCE_XFRC_APPLIED,
)
from mocap.humanoid21.centroidal_dynamics.model_spec import build_model_spec
from mocap.humanoid21.centroidal_dynamics.predict import (
    DiscreteCentroidalPredictor,
)
from mocap.humanoid21.centroidal_dynamics.probe import PhysicsStepProbe


@pytest.fixture()
def sim():
    s = Humanoid21Simulator()
    s.reset(seed=0)
    try:
        yield s
    finally:
        s.close()


def _state(spec, t=0.0):
    robot = spec.robots["robot_a"]
    return CentroidalState(
        robot_id="robot_a", time=t, mass=robot.total_mass,
        com_pos=np.array([0.0, 0.0, 1.0]),
        lin_mom=np.zeros(3), ang_mom=np.zeros(3),
        metadata={"model_fingerprint": spec.fingerprint})


def _snapshot(spec, t=0.0, loads=(), generalized=()):
    return LoadSnapshot(
        robot_id="robot_a", time=t, spatial_loads=tuple(loads),
        generalized_loads=tuple(generalized),
        gravity=np.asarray(spec.gravity),
        metadata={"model_fingerprint": spec.fingerprint})


def test_predictor_integrates_gravity_and_xfrc(sim):
    spec = build_model_spec(sim)
    pred = DiscreteCentroidalPredictor(spec)
    s0 = _state(spec)
    dt = spec.timestep
    load = ExternalLoad(
        source_type=SOURCE_XFRC_APPLIED, robot_id="robot_a",
        point_world=np.array([0.0, 0.0, 1.0]),
        force_world=np.array([10.0, 0.0, 0.0]),
        torque_world=np.array([0.0, 2.0, 0.0]))
    result = pred.predict(s0, [_snapshot(spec, loads=[load])])
    s1 = result.final_state
    expected_lin = (s0.mass * np.asarray(spec.gravity)
                    + load.force_world) * dt
    assert np.allclose(s1.lin_mom, expected_lin)
    assert np.allclose(s1.ang_mom, load.torque_world * dt)
    assert np.allclose(s1.com_pos,
                       s0.com_pos + s1.lin_mom / s0.mass * dt)
    assert s1.time == pytest.approx(dt)


def test_predictor_error_against_live_step_equals_pre_balance(sim):
    probe = PhysicsStepProbe(sim)
    actual = probe.probe_step("robot_a")
    pred = DiscreteCentroidalPredictor(probe.model_spec)
    predicted, _, _, _ = pred.step(actual.state0, actual.snapshot0)
    lin_err = predicted.lin_mom - actual.state1.lin_mom
    ang_err = predicted.ang_mom - actual.state1.ang_mom
    pre = actual.balances["pre"]
    assert np.linalg.norm(lin_err) == pytest.approx(
        np.linalg.norm(pre.lin_residual), abs=1e-12)
    assert np.linalg.norm(ang_err) == pytest.approx(
        np.linalg.norm(pre.ang_residual), abs=1e-12)
    assert np.linalg.norm(lin_err) < 0.05
    assert np.linalg.norm(ang_err) < 0.01


def test_predictor_rejects_non_ground_contact_by_default(sim):
    spec = build_model_spec(sim)
    pred = DiscreteCentroidalPredictor(spec)
    s0 = _state(spec)
    load = ExternalLoad(
        source_type=SOURCE_CONTACT_WALL, robot_id="robot_a",
        point_world=np.zeros(3), force_world=np.ones(3))
    with pytest.raises(PredictionDomainViolation,
                       match="non-ground contact"):
        pred.step(s0, _snapshot(spec, loads=[load]))


def test_predictor_rejects_qfrc_and_time_mismatch(sim):
    spec = build_model_spec(sim)
    pred = DiscreteCentroidalPredictor(spec)
    s0 = _state(spec)
    gen = GeneralizedLoad("robot_a", np.array([0]), np.array([1.0]))
    with pytest.raises(PredictionDomainViolation, match="qfrc"):
        pred.step(s0, _snapshot(spec, generalized=[gen]))
    with pytest.raises(PredictionDomainViolation, match="does not match"):
        pred.step(s0, _snapshot(spec, t=0.01))
