"""P0 model-contract tests for Humanoid21 centroidal dynamics."""
import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.centroidal_dynamics.model_spec import (
    build_model_spec,
)


@pytest.fixture(scope="module")
def sim():
    s = Humanoid21Simulator()
    s.reset(seed=0)
    try:
        yield s
    finally:
        s.close()


def test_model_spec_robot_geometry_and_mass(sim):
    spec = build_model_spec(sim)
    assert spec.timestep == pytest.approx(0.002)
    assert np.allclose(spec.gravity, [0.0, 0.0, -9.81])
    assert spec.nactuator == 42
    assert set(spec.robots) == {"robot_a", "robot_b"}

    for robot_id, robot in spec.robots.items():
        assert len(robot.body_ids) == 16
        assert len(robot.controlled_joint_ids) == 21
        assert len(robot.actuators) == 21
        assert robot.total_mass == pytest.approx(40.84402084)
        assert robot.root_body_name == f"torso_{robot_id[-1]}"
        assert len(robot.body_ids) == len(robot.bodies)
        assert all(b.body_id in robot.body_ids for b in robot.bodies)
        assert all(a.actuator_id >= 0 for a in robot.actuators)


def test_model_spec_fingerprint_is_stable_and_content_addressed(sim):
    a = build_model_spec(sim)
    b = build_model_spec(sim)
    assert a.fingerprint == b.fingerprint
    assert len(a.fingerprint) == 64
    payload = a.payload()
    assert "fingerprint" not in payload
    assert payload["robots"]["robot_a"]["total_mass"] > 0.0


def test_model_spec_records_contact_contracts(sim):
    spec = build_model_spec(sim)
    by_name = {g.name: g for g in spec.geoms}
    assert "ground" in by_name
    assert by_name["ground"].affiliation == 0
    for side in ("right", "left"):
        for part in ("foot1", "foot2"):
            geom = by_name[f"{part}_{side}_a"]
            assert geom.affiliation == 1
            assert geom.condim == 3
            assert geom.priority == 2
