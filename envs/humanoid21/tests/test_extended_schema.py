"""Coverage tests for the extended DATASPEC fields (§2.1 / §2.2 / §4.1.1 / §4.3).

These tests guarantee that observers can rely on the contract:
- per-agent body/joint name alignment
- keypoint name maps
- body masses aligned with body_names
- derived per-body arrays keyed by the same names
- structured ``contacts`` list with full geometry + world-frame force
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator


@pytest.fixture(scope="module")
def sim():
    simulator = Humanoid21Simulator()
    simulator.reset()
    return simulator


class TestStaticSchema:
    def test_global_fields(self, sim):
        static = sim.get_static_data()
        assert isinstance(static["dt"], float) and static["dt"] > 0.0
        assert static["ground_geom_name"] == "ground"

    def test_per_agent_fields_present(self, sim):
        static = sim.get_static_data()
        for agent in ("robot_a", "robot_b"):
            a = static[agent]
            for key in (
                "dof_names",
                "body_names",
                "body_masses_by_name",
                "joint_names",
                "controlled_joint_names",
                "root_joint_name",
                "keypoint_body_names",
                "keypoint_joint_names",
                "joint_limits",
            ):
                assert key in a, f"{agent} missing static field {key!r}"

    def test_body_masses_align_with_body_names(self, sim):
        static = sim.get_static_data()
        for agent in ("robot_a", "robot_b"):
            a = static[agent]
            assert set(a["body_names"]) == set(a["body_masses_by_name"].keys())
            for name in a["body_names"]:
                assert a["body_masses_by_name"][name] > 0.0

    def test_keypoint_maps(self, sim):
        static = sim.get_static_data()
        for agent, suffix in (("robot_a", "_a"), ("robot_b", "_b")):
            kb = static[agent]["keypoint_body_names"]
            kj = static[agent]["keypoint_joint_names"]
            # Body role → full name
            for role in ("torso", "head", "foot_left", "foot_right", "hand_left", "hand_right"):
                assert kb[role].endswith(suffix)
                assert kb[role] in static[agent]["body_names"]
            # Joint role → full name
            for role in ("ankle_x_left", "ankle_x_right", "ankle_y_left", "ankle_y_right"):
                assert kj[role].endswith(suffix)
                assert kj[role] in static[agent]["joint_names"]

    def test_controlled_joints_are_subset(self, sim):
        static = sim.get_static_data()
        for agent in ("robot_a", "robot_b"):
            a = static[agent]
            controlled = a["controlled_joint_names"]
            assert len(controlled) == 21
            for name in controlled:
                assert name in a["joint_names"]


class TestDerivedPerAgent:
    def test_body_arrays_cover_body_names(self, sim):
        static = sim.get_static_data()
        derived = sim.get_derived_state(['robot_a', 'robot_b'])
        for agent in ("robot_a", "robot_b"):
            names = set(static[agent]["body_names"])
            for field, expected_shape in (
                ("body_xpos", (3,)),
                ("body_xipos", (3,)),
                ("body_xquat", (4,)),
                ("body_linvel_world", (3,)),
                ("body_angvel_world", (3,)),
            ):
                mapping = derived[agent][field]
                assert set(mapping.keys()) == names, f"{agent}.{field} names mismatch"
                for arr in mapping.values():
                    assert isinstance(arr, np.ndarray)
                    assert arr.shape == expected_shape
                    assert arr.dtype == np.float32

    def test_joint_anchors_cover_joint_names(self, sim):
        static = sim.get_static_data()
        derived = sim.get_derived_state(['robot_a', 'robot_b'])
        for agent in ("robot_a", "robot_b"):
            assert set(derived[agent]["joint_world_anchor"].keys()) == set(
                static[agent]["joint_names"]
            )

    def test_copy_semantics_body_xipos(self, sim):
        """Mutating the returned ndarray must NOT leak back into the simulator."""
        derived1 = sim.get_derived_state(['robot_a', 'robot_b'])
        torso_name = sim.get_static_data()["robot_a"]["keypoint_body_names"]["torso"]
        derived1["robot_a"]["body_xipos"][torso_name][:] = 999.0

        sim._data_cache.clear()
        derived2 = sim.get_derived_state(['robot_a', 'robot_b'])
        assert not np.allclose(derived2["robot_a"]["body_xipos"][torso_name], 999.0)


class TestStructuredContacts:
    def test_contacts_shape_and_fields(self, sim):
        derived = sim.get_derived_state(['contacts'])
        cv = derived["contacts"]
        assert isinstance(cv, dict)
        assert cv['ncon'] > 0
        ncon = cv['ncon']
        # Check SoA arrays have correct length
        for key in ('geom1', 'geom2', 'body1', 'body2', 'aff1', 'aff2', 'force_mag'):
            assert cv[key].shape == (ncon,), f"{key} shape mismatch: {cv[key].shape}"
        for key in ('force_world', 'position', 'normal'):
            assert cv[key].shape == (ncon, 3), f"{key} shape mismatch: {cv[key].shape}"
        assert cv['frame'].shape == (ncon, 3, 3)

    def test_foot_ground_contact_has_upward_force(self, sim):
        """Standing pose: the foot–ground contact must push the foot upward."""
        static = sim.get_static_data()
        derived = sim.get_derived_state(['contacts'])
        cv = derived["contacts"]
        ground_geom_id = static.get('ground_geom_id')
        body_id_to_name = static.get('body_id_to_name', {})
        foot = static["robot_a"]["keypoint_body_names"]["foot_left"]

        total = np.zeros(3, dtype=np.float64)
        for i in range(cv['ncon']):
            g1, g2 = cv['geom1'][i], cv['geom2'][i]
            b1, b2 = cv['body1'][i], cv['body2'][i]
            fw = cv['force_world'][i]
            if g1 == ground_geom_id and body_id_to_name.get(int(b2), '') == foot:
                total += fw
            elif g2 == ground_geom_id and body_id_to_name.get(int(b1), '') == foot:
                total -= fw
        # Standing: Z component should be positive (support force)
        assert total[2] > 0.0, f"expected upward support force, got {total}"
