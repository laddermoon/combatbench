"""P2 external-load normalization tests."""
import mujoco
import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.centroidal_dynamics.contracts import (
    SOURCE_CONTACT_GROUND,
    SOURCE_XFRC_APPLIED,
    OutOfDomainContact,
    UnknownExternalSource,
    UnsupportedGeneralizedLoad,
)
from mocap.humanoid21.centroidal_dynamics.external_loads import (
    ExternalLoadNormalizer,
)


@pytest.fixture(scope="module")
def sim():
    s = Humanoid21Simulator()
    s.reset(seed=0)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(scope="module")
def normalizer(sim):
    return ExternalLoadNormalizer(sim)


def _contact_force_world_on_geom2(sim, contact_id):
    contact = sim.data.contact[contact_id]
    wrench = np.zeros(6)
    mujoco.mj_contactForce(sim.model, sim.data, contact_id, wrench)
    frame = np.asarray(contact.frame).reshape(3, 3)
    return frame.T @ wrench[:3]


def test_ground_contacts_match_contacts_vec_sign(sim, normalizer):
    sim.reset(seed=0)
    snap = normalizer.snapshot("robot_a")
    contacts = sim.get_derived_state(["contacts"])["contacts"]
    assert contacts["ncon"] > 0
    contact_loads = [l for l in snap.spatial_loads
                     if l.source_type == SOURCE_CONTACT_GROUND]
    assert len(contact_loads) > 0
    for load in contact_loads:
        cid = load.metadata["contact_id"]
        expected_geom2 = contacts["force_world"][cid]
        target_geom = load.metadata["target_geom_id"]
        expected = (-expected_geom2 if target_geom == contacts["geom1"][cid]
                    else expected_geom2)
        assert np.allclose(load.force_world, expected, atol=1e-5, rtol=1e-6)
        assert np.allclose(load.point_world, contacts["position"][cid],
                           atol=1e-6, rtol=0.0)
        assert np.allclose(load.torque_world, np.zeros(3))


def test_xfrc_is_body_com_world_wrench_and_cleared_by_step(sim, normalizer):
    sim.reset(seed=0)
    body_id = sim._meta['robots']['robot_a']['root_body_id']
    force = np.array([10.0, 20.0, 30.0])
    torque = np.array([1.0, 2.0, 3.0])
    sim.data.ctrl[:] = 0.0
    sim.data.qfrc_applied[:] = 0.0
    sim.data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    qfrc0 = sim.data.qfrc_smooth.copy()
    sim.apply_external_force("torso", force, torque, robot_id="robot_a")

    snap = normalizer.snapshot("robot_a", refresh=False)
    applied = [l for l in snap.spatial_loads
               if l.source_type == SOURCE_XFRC_APPLIED]
    assert len(applied) == 1
    load = applied[0]
    assert load.body_id == body_id
    assert load.body_name == "torso_a"
    assert np.allclose(load.point_world, sim.data.xipos[body_id])
    assert np.allclose(load.force_world, force)
    assert np.allclose(load.torque_world, torque)

    mujoco.mj_forward(sim.model, sim.data)
    jac_p = np.zeros((3, sim.model.nv))
    jac_r = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBodyCom(sim.model, sim.data, jac_p, jac_r, body_id)
    expected_delta = jac_p.T @ force + jac_r.T @ torque
    assert np.allclose(sim.data.qfrc_smooth - qfrc0, expected_delta,
                       atol=1e-8, rtol=1e-8)
    assert np.allclose(sim.data.qfrc_applied, np.zeros(sim.model.nv))

    sim.physical_step()
    assert np.allclose(sim.data.xfrc_applied, 0.0)
    snap_after = normalizer.snapshot("robot_a")
    assert not [l for l in snap_after.spatial_loads
                if l.source_type == SOURCE_XFRC_APPLIED]


def test_nonzero_target_qfrc_is_explicitly_unsupported(sim, normalizer):
    sim.reset(seed=0)
    dof = sim._meta['robots']['robot_a']['root_qvel_adr']
    sim.data.qfrc_applied[dof] = 2.5
    with pytest.raises(UnsupportedGeneralizedLoad):
        normalizer.snapshot("robot_a")
    snap = normalizer.snapshot("robot_b", strict_generalized=False)
    assert len(snap.generalized_loads) == 0
    sim.reset(seed=0)


def test_unmapped_contact_geom_fails_loud(sim, normalizer):
    sim.reset(seed=0)
    assert sim.data.ncon > 0
    geom_id = int(sim.data.contact[0].geom1)
    geom_map = sim._meta['geom_id_to_aff']
    old = geom_map.pop(geom_id)
    try:
        with pytest.raises(UnknownExternalSource):
            normalizer.snapshot("robot_a")
    finally:
        geom_map[geom_id] = old


def test_contact_torque_is_out_of_domain(sim, normalizer, monkeypatch):
    sim.reset(seed=0)
    assert sim.data.ncon > 0
    import mocap.humanoid21.centroidal_dynamics.external_loads as mod
    real = mujoco.mj_contactForce

    def fake_contact_force(model, data, contact_id, out):
        out[:] = 0.0
        out[0] = 1.0
        out[3] = 0.1

    monkeypatch.setattr(mod.mujoco, "mj_contactForce", fake_contact_force)
    try:
        with pytest.raises(OutOfDomainContact):
            normalizer.snapshot("robot_a")
    finally:
        monkeypatch.setattr(mod.mujoco, "mj_contactForce", real)
