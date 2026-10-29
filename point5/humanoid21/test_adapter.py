"""Tests for point5_from_view (real get_derived_state path, v2 local frame)."""
from __future__ import annotations

import numpy as np
import pytest

from point5 import Point5Error, ankle_mid_xy, to_world, twist_deg
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
        np.testing.assert_allclose(ankle_mid_xy(st.pos), [0., 0.],
                                   atol=1e-9)
        d = np.linalg.norm(st.point_pos("foot_right")[:2]
                           - st.point_pos("foot_left")[:2])
        assert d > 0.1

    def test_shape_and_vel(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        assert st.pos.shape == (5, 3)
        assert st.vel is not None and st.vel.shape == (5, 3)
        assert st.metadata["robot_id"] == "robot_a"
        assert st.metadata["source"] == "derived_view"

    def test_vel_optional(self, view_a):
        assert point5_from_view(view_a, "robot_a",
                                with_vel=False).vel is None

    def test_world_roundtrip_and_robot_isolation(self, view_a, view_b):
        a = point5_from_view(view_a, "robot_a")
        b = point5_from_view(view_b, "robot_b")
        # local poses are frame-relative; metadata carries the world frame
        wa = to_world(a.pos, a.metadata["frame_origin_xy"],
                      a.metadata["frame_yaw_deg"])
        wb = to_world(b.pos, b.metadata["frame_origin_xy"],
                      b.metadata["frame_yaw_deg"])
        assert abs(wa[0, 0] - wb[0, 0]) > 1.5   # worlds differ (heads)
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
        with pytest.raises(Point5Error):
            point5_from_view({"body_xpos": view_a["body_xpos"],
                              "body_xquat": view_a["body_xquat"]},
                             "robot_a", with_vel=True)  # no linvel field

    def test_returns_copies(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        st.pos[0, 0] = 999.0
        assert view_a["body_xpos"]["head_a"][0] != 999.0
