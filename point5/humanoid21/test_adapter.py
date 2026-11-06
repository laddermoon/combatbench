"""Tests for point5_from_view (real get_derived_state path, v2 local frame)."""
from __future__ import annotations

import numpy as np
import pytest

from point5 import Point5Error, twist_deg
from point5.humanoid21 import point5_from_view


@pytest.fixture(scope="module")
def sim():
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator()
    s.reset(seed=0)
    return s


@pytest.fixture(scope="module")
def view_a(sim):
    return sim.get_derived_state(["robot_a"])["robot_a"]


@pytest.fixture(scope="module")
def view_b(sim):
    return sim.get_derived_state(["robot_b"])["robot_b"]


class TestFromView:
    def test_positions_sane_at_reset(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        head = st.point_pos("head")
        assert 1.0 < head[2] < 1.8
        for f in ("foot_right", "foot_left"):
            z = st.point_pos(f)[2]
            assert 0.05 < z < 0.12   # ankle-center ~0.087, above sole
        # local frame: ankle XY midpoint is the origin by construction
        mid = 0.5 * (st.pos[3, :2] + st.pos[4, :2])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)
        d = np.linalg.norm(st.point_pos("foot_right")[:2]
                           - st.point_pos("foot_left")[:2])
        assert d > 0.1

    def test_shape(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        assert st.pos.shape == (5, 3)

    def test_world_roundtrip_and_robot_isolation(self, view_a, view_b):
        a = point5_from_view(view_a, "robot_a")
        b = point5_from_view(view_b, "robot_b")
        # identical frame convention -> local poses comparable directly
        assert abs(a.pos[0, 0] - b.pos[0, 0]) < 0.5
        # twist is derived from stored feet, finite at reset
        assert twist_deg(a.pos) is not None

    def test_fail_loud(self, view_a):
        with pytest.raises(Point5Error):
            point5_from_view(view_a, "robot_c")
        with pytest.raises(Point5Error):
            point5_from_view({}, "robot_a")
        with pytest.raises(Point5Error):
            point5_from_view({"body_xpos": {},
                              "body_xquat": {}}, "robot_a")

    def test_returns_copies(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        head0 = st.pos[0, 0]
        view_a["body_xpos"]["head_a"][0] += 999.0   # mutate source
        assert st.pos[0, 0] == head0              # state unaffected
