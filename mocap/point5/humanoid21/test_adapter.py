"""Tests for point5_from_view (real get_derived_state path)."""
from __future__ import annotations

import numpy as np
import pytest

from mocap.point5 import Point5Error, is_planted
from mocap.point5.humanoid21 import point5_from_view


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
        assert 1.2 < head[2] < 1.8
        for f in ("foot_right", "foot_left"):
            assert is_planted(st.point_pos(f), tol=0.05)
        assert abs(st.point_pos("foot_right")[1]
                   - st.point_pos("foot_left")[1]) > 0.1

    def test_shape_and_vel(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        assert st.pos.shape == (5, 3)
        assert st.vel is not None and st.vel.shape == (5, 3)
        assert st.metadata["robot_id"] == "robot_a"
        assert st.metadata["source"] == "derived_view"

    def test_vel_optional(self, view_a):
        assert point5_from_view(view_a, "robot_a",
                                with_vel=False).vel is None

    def test_robot_isolation(self, view_a, view_b):
        a = point5_from_view(view_a, "robot_a")
        b = point5_from_view(view_b, "robot_b")
        assert abs(a.point_pos("head")[0]
                   - b.point_pos("head")[0]) > 1.5

    def test_fail_loud(self, view_a):
        with pytest.raises(Point5Error):
            point5_from_view(view_a, "robot_c")
        with pytest.raises(Point5Error):
            point5_from_view({}, "robot_a")
        with pytest.raises(Point5Error):
            point5_from_view({"body_xpos": {}}, "robot_a")
        with pytest.raises(Point5Error):
            point5_from_view({"body_xpos": view_a["body_xpos"]},
                             "robot_a", with_vel=True)  # no linvel field

    def test_returns_copies(self, view_a):
        st = point5_from_view(view_a, "robot_a")
        st.pos[0, 0] = 999.0
        assert view_a["body_xpos"]["head_a"][0] != 999.0
