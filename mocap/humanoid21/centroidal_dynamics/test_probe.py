"""P3 live physics-substep probe tests."""
import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.centroidal_dynamics.contracts import (
    SOURCE_XFRC_APPLIED,
    UnsupportedGeneralizedLoad,
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


def test_probe_step_records_pre_post_state_loads_and_residuals(sim):
    probe = PhysicsStepProbe(sim)
    result = probe.probe_step("robot_a")
    assert result.dt == pytest.approx(sim.model.opt.timestep)
    assert result.state0.time < result.state1.time
    assert set(result.balances) == {"pre", "post", "trapezoid"}
    assert result.snapshot0.metadata["ncon"] > 0
    assert result.snapshot1.metadata["ncon"] > 0
    for balance in result.balances.values():
        assert np.all(np.isfinite(balance.lin_residual))
        assert np.all(np.isfinite(balance.ang_residual))
        assert np.linalg.norm(balance.lin_residual) < 1.0


def test_probe_samples_xfrc_before_physical_step_clears_it(sim):
    probe = PhysicsStepProbe(sim)
    sim.apply_external_force("torso", np.array([4.0, 3.0, 2.0]),
                             np.array([0.1, 0.0, 0.0]),
                             robot_id="robot_a")
    result = probe.probe_step("robot_a")
    applied = [l for l in result.snapshot0.spatial_loads
               if l.source_type == SOURCE_XFRC_APPLIED]
    assert len(applied) == 1
    assert np.allclose(applied[0].force_world, [4.0, 3.0, 2.0])
    assert not [l for l in result.snapshot1.spatial_loads
                if l.source_type == SOURCE_XFRC_APPLIED]


def test_probe_rejects_target_qfrc_load(sim):
    probe = PhysicsStepProbe(sim)
    dof = sim._meta['robots']['robot_a']['root_qvel_adr']
    sim.data.qfrc_applied[dof] = 1.0
    with pytest.raises(UnsupportedGeneralizedLoad):
        probe.probe_step("robot_a")
    sim.reset(seed=0)


def test_probe_steps_preserves_existing_action_contract(sim):
    probe = PhysicsStepProbe(sim)
    results = probe.probe_steps(
        "robot_a", 2,
        action={"robot_a": np.zeros(21), "robot_b": None})
    assert len(results) == 2
    assert results[0].index == 0 and results[1].index == 1
    assert np.allclose(sim._target_pos_norm["robot_a"], np.zeros(21))
