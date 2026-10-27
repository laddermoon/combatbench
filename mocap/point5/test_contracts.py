"""Tests for point5 contracts: vocabulary, specs, task, anchors, serde."""
from __future__ import annotations

import numpy as np
import pytest

from .contracts import (
    GROUND_Z,
    POINT_NAMES,
    Keyframe,
    Point5State,
    Point5Task,
    PointSpec,
    RootOffset,
    SelfPointOffset,
    StaticStartResolver,
    TaskValidationError,
    WorldPoint,
    is_planted,
)


def _spec(p=(0, 0, 0), **kw) -> PointSpec:
    return PointSpec(target=WorldPoint(np.asarray(p, dtype=float)), **kw)


def _keyframe(**kw) -> Keyframe:
    return Keyframe(specs={n: _spec((i, 0, 0)) for i, n in enumerate(POINT_NAMES)},
                    **kw)


def _initial() -> Point5State:
    return Point5State(pos=np.array(
        [[0, 0, 1.5], [0.3, -0.2, 1.2], [0.3, 0.2, 1.2],
         [0, -0.1, 0.0], [0, 0.1, 0.0]]))


class TestVocabulary:
    def test_five_fixed_names(self):
        assert POINT_NAMES == ("head", "hand_right", "hand_left",
                               "foot_right", "foot_left")

    def test_keyframe_must_cover_all_five(self):
        with pytest.raises(TaskValidationError):
            Keyframe(specs={"head": _spec()})

    def test_keyframe_rejects_unknown_point(self):
        specs = {n: _spec() for n in POINT_NAMES}
        specs["torso"] = specs.pop("head")
        with pytest.raises(TaskValidationError):
            Keyframe(specs=specs)

    def test_spec_requires_anchor_object(self):
        with pytest.raises(TaskValidationError):
            PointSpec(target=np.zeros(3))  # raw vector is not an Anchor

    def test_spec_validation(self):
        with pytest.raises(TaskValidationError):
            _spec(tolerance=0.0)
        with pytest.raises(TaskValidationError):
            _spec(weight=-0.1)


class TestState:
    def test_pos_shape(self):
        with pytest.raises(TaskValidationError):
            Point5State(pos=np.zeros((4, 3)))
        with pytest.raises(TaskValidationError):
            Point5State(pos=np.full((5, 3), np.nan))

    def test_vel_optional_but_shape_checked(self):
        st = Point5State(pos=np.zeros((5, 3)))
        assert st.vel is None
        assert st.point_vel("head") is None
        st2 = Point5State(pos=np.zeros((5, 3)), vel=np.ones((5, 3)))
        assert st2.point_vel("head").shape == (3,)
        with pytest.raises(TaskValidationError):
            Point5State(pos=np.zeros((5, 3)), vel=np.zeros((5, 2)))

    def test_point_access_and_bad_name(self):
        st = _initial()
        np.testing.assert_allclose(st.point_pos("head"), [0, 0, 1.5])
        with pytest.raises(TaskValidationError):
            st.point_pos("torso")


class TestTask:
    def test_needs_two_keyframes(self):
        with pytest.raises(TaskValidationError):
            Point5Task(name="t", keyframes=(_keyframe(),), durations=())

    def test_durations_length(self):
        kf = _keyframe()
        with pytest.raises(TaskValidationError):
            Point5Task(name="t", keyframes=(kf, kf, kf), durations=(1.0,))

    def test_durations_positive(self):
        kf = _keyframe()
        for bad in (0.0, -1.0, np.nan):
            with pytest.raises(TaskValidationError):
                Point5Task(name="t", keyframes=(kf, kf), durations=(bad,))

    def test_duration_and_segments(self):
        kf = _keyframe()
        t = Point5Task(name="t", keyframes=(kf, kf, kf), durations=(0.3, 0.5))
        assert t.duration == pytest.approx(0.8)
        assert t.n_segments == 2


class TestAnchors:
    def test_world_point(self):
        r = StaticStartResolver(_initial(), np.zeros(3), np.eye(3))
        np.testing.assert_allclose(r.resolve(WorldPoint([1, 2, 3])),
                                   [1, 2, 3])

    def test_root_offset_applies_rotation(self):
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # +90deg yaw
        r = StaticStartResolver(_initial(), np.array([1.0, 0, 0]), rot)
        got = r.resolve(RootOffset([0.5, 0, 0]))
        np.testing.assert_allclose(got, [1.0, 0.5, 0.0], atol=1e-12)

    def test_self_point_offset_uses_initial(self):
        r = StaticStartResolver(_initial(), np.zeros(3), np.eye(3))
        got = r.resolve(SelfPointOffset("hand_right", [0.1, 0, 0.2]))
        np.testing.assert_allclose(got, [0.4, -0.2, 1.4])

    def test_bad_resolver_inputs(self):
        with pytest.raises(TaskValidationError):
            StaticStartResolver(_initial(), np.zeros(3), np.eye(2))
        with pytest.raises(TaskValidationError):
            SelfPointOffset("torso", np.zeros(3))


class TestSerde:
    def test_task_roundtrip(self):
        kf0 = Keyframe(specs={
            "head": PointSpec(WorldPoint([0, 0, 1.5]), tolerance=0.1),
            "hand_right": PointSpec(RootOffset([0.5, 0, 0]), weight=2.0),
            "hand_left": PointSpec(SelfPointOffset("head", [0, 0, -0.1])),
            "foot_right": _spec(), "foot_left": _spec()},
            label="strike")
        kf1 = _keyframe()
        task = Point5Task(name="jab", keyframes=(kf0, kf1),
                          durations=(0.4,), metadata={"author": "x"})
        back = Point5Task.from_dict(task.to_dict())
        assert back.name == "jab" and back.durations == (0.4,)
        assert back.keyframes[0].label == "strike"
        s = back.keyframes[0].spec("hand_right")
        assert isinstance(s.target, RootOffset) and s.weight == 2.0
        s2 = back.keyframes[0].spec("hand_left")
        assert isinstance(s2.target, SelfPointOffset)
        np.testing.assert_allclose(s2.target.offset, [0, 0, -0.1])


class TestGroundConvention:
    def test_is_planted(self):
        assert is_planted([0, 0, GROUND_Z])
        assert is_planted([0, 0, 0.01])
        assert not is_planted([0, 0, 0.05])
        assert is_planted([0, 0, -0.01])
