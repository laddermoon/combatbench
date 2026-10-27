"""Tests for the Humanoid21 point5 adapter (requires the real simulator)."""
from __future__ import annotations

import numpy as np
import pytest

from mocap.point5 import (
    POINT_INDEX,
    POINT_NAMES,
    Keyframe,
    Point5Task,
    PointSpec,
    RootOffset,
    SelfPointOffset,
    WorldPoint,
    build_reference,
    is_planted,
)
from mocap.point5.humanoid21 import Humanoid21Point5


@pytest.fixture(scope="module")
def sim():
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator()
    s.reset(seed=0)
    return s


@pytest.fixture(scope="module")
def adapter(sim):
    return Humanoid21Point5(sim)


class TestMeasure:
    def test_positions_sane_at_reset(self, adapter):
        st = adapter.measure("robot_a")
        head = st.point_pos("head")
        assert 1.2 < head[2] < 1.8              # head well above ground
        for f in ("foot_right", "foot_left"):
            assert is_planted(st.point_pos(f), tol=0.05)
        assert abs(st.point_pos("foot_right")[1]
                   - st.point_pos("foot_left")[1]) > 0.1  # feet apart

    def test_shape_time_metadata(self, adapter):
        st = adapter.measure("robot_a")
        assert st.pos.shape == (5, 3)
        assert st.vel is not None and st.vel.shape == (5, 3)
        assert st.time == pytest.approx(0.0)
        assert st.metadata["robot_id"] == "robot_a"

    def test_vel_optional(self, adapter):
        assert adapter.measure("robot_a", with_vel=False).vel is None

    def test_robot_isolation(self, adapter):
        a = adapter.measure("robot_a")
        b = adapter.measure("robot_b")
        assert abs(a.point_pos("head")[0]
                   - b.point_pos("head")[0]) > 1.5   # roots on opposite sides
        np.testing.assert_allclose(
            a.point_pos("foot_right")[1], a.point_pos("foot_left")[1],
            atol=0.2)

    def test_returns_copies(self, adapter):
        st = adapter.measure("robot_a")
        st.pos[0, 0] = 999.0
        assert adapter.measure("robot_a").pos[0, 0] != 999.0


class TestResolver:
    def test_root_offset_facing_direction(self, adapter):
        # robot_a faces +x, robot_b faces -x (euler 180deg)
        a = adapter.resolver("robot_a")
        b = adapter.resolver("robot_b")
        pa = a.resolve(RootOffset([1.0, 0, 0])) - adapter.root_pose("robot_a")[0]
        pb = b.resolve(RootOffset([1.0, 0, 0])) - adapter.root_pose("robot_b")[0]
        assert pa[0] > 0.9 and pb[0] < -0.9

    def test_self_point_offset(self, adapter):
        r = adapter.resolver("robot_a")
        st = adapter.measure("robot_a")
        got = r.resolve(SelfPointOffset("head", [0, 0, 0.1]))
        np.testing.assert_allclose(
            got, st.point_pos("head") + np.array([0, 0, 0.1]))

    def test_resolver_uses_supplied_snapshot(self, adapter):
        st = adapter.measure("robot_a")
        moved = type(st)(pos=st.pos + np.array([10, 0, 0]), time=0.0)
        r = adapter.resolver("robot_a", initial=moved,
                             root_pose=(np.zeros(3), np.eye(3)))
        got = r.resolve(SelfPointOffset("foot_left", np.zeros(3)))
        np.testing.assert_allclose(
            got, st.point_pos("foot_left") + np.array([10, 0, 0]))


class TestEndToEnd:
    def test_task_against_live_resolver(self, adapter):
        # "hand_right reaches 0.3m forward of root" at a reset pose
        reach = Keyframe(specs={
            n: PointSpec(WorldPoint(
                adapter.measure("robot_a").point_pos(n)))
            for n in POINT_NAMES})
        strike = Keyframe(specs={
            **{n: PointSpec(WorldPoint(
                adapter.measure("robot_a").point_pos(n)))
               for n in POINT_NAMES if n != "hand_right"},
            "hand_right": PointSpec(RootOffset([0.3, 0, 1.2])),
        }, label="strike")
        task = Point5Task(name="reach", keyframes=(reach, strike),
                          durations=(0.4,))
        ref = build_reference(task, adapter.resolver("robot_a"))
        mid = ref.sample_time(0.2)
        p0 = ref.target_at(0)[POINT_INDEX["hand_right"]]
        p1 = ref.target_at(1)[POINT_INDEX["hand_right"]]
        np.testing.assert_allclose(
            mid.point_pos("hand_right"), 0.5 * p0 + 0.5 * p1)
