"""Tests for the Humanoid21 point5 adapter (requires the real simulator)."""
from __future__ import annotations

import numpy as np
import pytest

from mocap.point5 import Point5Error, is_planted
from mocap.point5.humanoid21 import measure_point5


@pytest.fixture(scope="module")
def sim():
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator()
    s.reset(seed=0)
    return s


class TestMeasure:
    def test_positions_sane_at_reset(self, sim):
        st = measure_point5(sim, "robot_a")
        head = st.point_pos("head")
        assert 1.2 < head[2] < 1.8              # head well above ground
        for f in ("foot_right", "foot_left"):
            assert is_planted(st.point_pos(f), tol=0.05)
        assert abs(st.point_pos("foot_right")[1]
                   - st.point_pos("foot_left")[1]) > 0.1  # feet apart

    def test_shape_time_metadata(self, sim):
        st = measure_point5(sim, "robot_a")
        assert st.pos.shape == (5, 3)
        assert st.vel is not None and st.vel.shape == (5, 3)
        assert st.time == pytest.approx(0.0)
        assert st.metadata["robot_id"] == "robot_a"
        assert st.metadata["source"] == "humanoid21_sim"

    def test_vel_optional(self, sim):
        assert measure_point5(sim, "robot_a", with_vel=False).vel is None

    def test_robot_isolation(self, sim):
        a = measure_point5(sim, "robot_a")
        b = measure_point5(sim, "robot_b")
        assert abs(a.point_pos("head")[0]
                   - b.point_pos("head")[0]) > 1.5   # roots on opposite sides

    def test_unknown_robot(self, sim):
        with pytest.raises(Point5Error):
            measure_point5(sim, "robot_c")

    def test_returns_copies(self, sim):
        st = measure_point5(sim, "robot_a")
        st.pos[0, 0] = 999.0
        assert measure_point5(sim, "robot_a").pos[0, 0] != 999.0
