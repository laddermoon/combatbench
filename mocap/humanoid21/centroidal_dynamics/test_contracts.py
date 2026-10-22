"""Contract-level unit tests for centroidal dynamics."""
import numpy as np
import pytest

from mocap.humanoid21.centroidal_dynamics.contracts import (
    CentroidalState,
    DomainIssue,
    DomainReport,
    ExternalLoad,
)


def _state(**overrides):
    kw = {
        "robot_id": "robot_a",
        "time": 0.0,
        "mass": 40.0,
        "com_pos": np.zeros(3),
        "lin_mom": np.zeros(3),
        "ang_mom": np.zeros(3),
    }
    kw.update(overrides)
    return CentroidalState(**kw)


def test_centroidal_state_validates_robot_mass_and_vectors():
    st = _state(lin_mom=np.array([40.0, 0.0, 0.0]))
    assert np.allclose(st.com_vel, [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="robot_id"):
        _state(robot_id="robot_c")
    with pytest.raises(ValueError, match="mass"):
        _state(mass=0.0)
    with pytest.raises(ValueError, match="com_pos"):
        _state(com_pos=np.zeros(4))


def test_external_load_moment_is_about_requested_point():
    load = ExternalLoad(
        source_type="contact_ground", robot_id="robot_a",
        point_world=np.array([1.0, 0.0, 0.0]),
        force_world=np.array([0.0, 0.0, 10.0]))
    # Point force about a point 1 m along x produces y moment.
    assert np.allclose(load.moment_about(np.zeros(3)),
                       [0.0, -10.0, 0.0])


def test_domain_report_rejects_valid_with_invalid_issue():
    issue = DomainIssue("invalid", "bad", "bad input")
    with pytest.raises(ValueError, match="valid report"):
        DomainReport(valid=True, issues=(issue,))
