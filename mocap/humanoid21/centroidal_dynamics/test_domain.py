"""P4 domain and contact-capability tests."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.centroidal_dynamics.contracts import (
    ExternalLoad,
    GeneralizedLoad,
    LoadSnapshot,
    SOURCE_CONTACT_GROUND,
)
from mocap.humanoid21.centroidal_dynamics.domain import (
    CentroidalDomainChecker,
    _point_to_convex_margin,
)
from mocap.humanoid21.centroidal_dynamics.external_loads import (
    ExternalLoadNormalizer,
)
from mocap.humanoid21.centroidal_dynamics.state import CentroidalStateExtractor


@pytest.fixture()
def sim():
    s = Humanoid21Simulator()
    s.reset(seed=0)
    try:
        yield s
    finally:
        s.close()


def _state_and_snapshot(sim):
    extractor = CentroidalStateExtractor(sim)
    normalizer = ExternalLoadNormalizer(sim, extractor.model_spec)
    snapshot = normalizer.snapshot("robot_a")
    state = extractor.state("robot_a")
    return state, snapshot


def _codes(report):
    return {i.code for i in report.issues}


def test_reset_state_passes_conservative_domain(sim):
    state, snapshot = _state_and_snapshot(sim)
    report = CentroidalDomainChecker(sim).check(state, snapshot)
    assert report.valid, report.issues
    assert report.margins["support_polygon_margin"] > 0.0
    assert report.margins["tilt_margin_rad"] > 0.0
    assert report.margins["contact_moment_about_com_bound"] > 0.0


def test_excessive_tilt_is_invalid(sim):
    root = sim._meta['robots']['robot_a']['root_qpos_adr']
    q = sim.data.qpos[root + 3:root + 7].copy()
    pitch = R.from_euler("y", 0.8) * R.from_quat(
        [q[1], q[2], q[3], q[0]])
    out = pitch.as_quat()
    sim.data.qpos[root + 3:root + 7] = [out[3], out[0], out[1], out[2]]
    mujoco.mj_forward(sim.model, sim.data)
    state, snapshot = _state_and_snapshot(sim)
    report = CentroidalDomainChecker(sim).check(state, snapshot)
    assert not report.valid
    assert "excessive_tilt" in _codes(report)


def test_friction_violation_is_invalid(sim):
    state, snapshot = _state_and_snapshot(sim)
    checker = CentroidalDomainChecker(sim)
    spec = checker.model_spec
    foot = next(g for g in spec.robots["robot_a"].bodies
                if g.name == "foot_right_a")
    foot_geom = next(g for g in spec.geoms
                     if g.body_id == foot.body_id and
                     g.name.startswith("foot"))
    load = ExternalLoad(
        source_type=SOURCE_CONTACT_GROUND, robot_id="robot_a",
        point_world=np.array([0.0, 0.0, 0.0]),
        force_world=np.array([1000.0, 0.0, 10.0]),
        body_id=foot.body_id,
        body_name=foot.name,
        geom_pair=(spec.ground_geom_id, foot_geom.geom_id),
        metadata={"contact_id": 0,
                  "target_geom_id": foot_geom.geom_id,
                  "normal_world": np.array([0.0, 0.0, 1.0])})
    snap = LoadSnapshot("robot_a", state.time, (load,),
                        gravity=np.array([0.0, 0.0, -9.81]),
                        metadata=state.metadata)
    report = checker.check(state, snap)
    assert not report.valid
    assert "friction_cone_violation" in _codes(report)


def test_generalized_load_is_invalid_for_first_phase(sim):
    state, snapshot = _state_and_snapshot(sim)
    gen = GeneralizedLoad("robot_a", np.array([0]), np.array([1.0]))
    bad = LoadSnapshot("robot_a", state.time, snapshot.spatial_loads,
                       generalized_loads=(gen,),
                       gravity=snapshot.gravity,
                       metadata=snapshot.metadata)
    report = CentroidalDomainChecker(sim).check(state, bad)
    assert not report.valid
    assert "unsupported_generalized_load" in _codes(report)


def test_convex_margin_sign():
    square = np.array([[-1.0, -1.0], [1.0, -1.0],
                       [1.0, 1.0], [-1.0, 1.0]])
    assert _point_to_convex_margin(square, np.zeros(2)) > 0.0
    assert _point_to_convex_margin(square, np.array([2.0, 0.0])) < 0.0
