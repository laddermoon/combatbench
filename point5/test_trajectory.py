"""Tests for reference synthesis: resolution, linear interp, sampling."""
from __future__ import annotations

import numpy as np
import pytest

from .contracts import (
    POINT_NAMES,
    Keyframe,
    Point5State,
    Point5Task,
    PointSpec,
    RootOffset,
    SampleOutOfRange,
    SelfPointOffset,
    StaticStartResolver,
    TaskValidationError,
    WorldPoint,
)
from .trajectory import (
    Point5Reference,
    adaptive_sample,
    build_reference,
    frames_payload,
    resolve_task,
)


def _kf(pos_fn, label=""):
    return Keyframe(specs={n: PointSpec(target=WorldPoint(np.asarray(
        pos_fn(i, n), dtype=float))) for i, n in enumerate(POINT_NAMES)},
                    label=label)


def _world_resolver() -> StaticStartResolver:
    initial = Point5State(pos=np.zeros((5, 3)))
    return StaticStartResolver(initial, np.zeros(3), np.eye(3))


class TestResolution:
    def test_resolve_world_targets(self):
        task = Point5Task(name="t", durations=(1.0,), keyframes=(
            _kf(lambda i, n: (0, 0, i)),
            _kf(lambda i, n: (1, 0, i))))
        ref = build_reference(task, _world_resolver())
        np.testing.assert_allclose(ref.target_at(0)[0], [0, 0, 0])
        np.testing.assert_allclose(ref.target_at(1)[4], [1, 0, 4])

    def test_static_anchors_resolve_against_start(self):
        initial = Point5State(pos=np.array(
            [[0, 0, 1.5], [0.3, -0.2, 1.2], [0.3, 0.2, 1.2],
             [0, -0.1, 0.0], [0, 0.1, 0.0]]))
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        r = StaticStartResolver(initial, np.array([2.0, 0, 0]), rot)
        kf = Keyframe(specs={
            "head": PointSpec(RootOffset([0.5, 0, 1.0])),
            "hand_right": PointSpec(SelfPointOffset("head", [0.1, 0, 0])),
            "hand_left": PointSpec(WorldPoint([9, 9, 9])),
            "foot_right": PointSpec(WorldPoint([0, 0, 0])),
            "foot_left": PointSpec(WorldPoint([0, 0, 0]))})
        task = Point5Task(name="t", keyframes=(kf, kf), durations=(1.0,))
        ref = build_reference(task, r)
        P = ref.target_at(0)
        np.testing.assert_allclose(P[0], [2.0, 0.5, 1.0])     # root frame rot
        np.testing.assert_allclose(P[1], [0.1, 0, 1.5])       # own head + off
        np.testing.assert_allclose(P[2], [9, 9, 9])           # world

    def test_resolver_nonfinite_fails(self):
        class Bad:
            def resolve(self, a):
                return np.array([np.nan, 0, 0])
        kf = _kf(lambda i, n: (0, 0, 0))
        with pytest.raises(TaskValidationError):
            resolve_task(Point5Task(name="t", keyframes=(kf, kf),
                                    durations=(1.0,)), Bad())


class TestSampling:
    def _two_seg_task(self):
        # head: (0,0,0) -> (2,0,0) -> (2,2,0); others static
        def mk(head, label):
            def f(i, n):
                return head if n == "head" else (0, 0, i)
            return _kf(f, label)
        return Point5Task(name="walk", durations=(0.4, 0.6), keyframes=(
            mk((0, 0, 0), "start"), mk((2, 0, 0), "mid"), mk((2, 2, 0), "end")))

    def test_linear_midpoint(self):
        ref = build_reference(self._two_seg_task(), _world_resolver())
        st = ref.sample(0, 0.5)
        np.testing.assert_allclose(st.point_pos("head"), [1, 0, 0])
        st = ref.sample(1, 0.5)
        np.testing.assert_allclose(st.point_pos("head"), [2, 1, 0])

    def test_boundaries_exact(self):
        ref = build_reference(self._two_seg_task(), _world_resolver())
        np.testing.assert_allclose(ref.sample(0, 0.0).point_pos("head"),
                                   [0, 0, 0])
        np.testing.assert_allclose(ref.sample(1, 1.0).point_pos("head"),
                                   [2, 2, 0])

    def test_sample_time_maps_segments(self):
        ref = build_reference(self._two_seg_task(), _world_resolver())
        np.testing.assert_allclose(ref.sample_time(0.2).point_pos("head"),
                                   [1, 0, 0])
        np.testing.assert_allclose(ref.sample_time(0.4).point_pos("head"),
                                   [2, 0, 0])  # exact boundary
        np.testing.assert_allclose(ref.sample_time(0.7).point_pos("head"),
                                   [2, 1, 0])
        assert ref.sample_time(0.8).time == pytest.approx(0.8)

    def test_out_of_range_raises(self):
        ref = build_reference(self._two_seg_task(), _world_resolver())
        with pytest.raises(SampleOutOfRange):
            ref.sample_time(-0.01)
        with pytest.raises(SampleOutOfRange):
            ref.sample_time(1.01)
        with pytest.raises(SampleOutOfRange):
            ref.sample(2, 0.5)
        with pytest.raises(SampleOutOfRange):
            ref.sample(0, 1.5)
        # exact endpoints are fine
        ref.sample_time(0.0)
        ref.sample_time(1.0)

    def test_no_velocity_anywhere(self):
        ref = build_reference(self._two_seg_task(), _world_resolver())
        assert ref.sample(0, 0.5).vel is None
        assert ref.sample_time(0.5).vel is None

    def test_static_segment_via_equal_targets(self):
        # same target at both ends => constant segment without any hold flag
        ref = build_reference(self._two_seg_task(), _world_resolver())
        a = ref.sample(0, 0.2).point_pos("hand_right")
        b = ref.sample(0, 0.9).point_pos("hand_right")
        np.testing.assert_allclose(a, b)

    def test_metadata_labels(self):
        ref = build_reference(self._two_seg_task(), _world_resolver())
        assert ref.sample(0, 0.2).metadata["keyframe_label"] == "start"
        assert ref.sample(0, 0.8).metadata["keyframe_label"] == "mid"
        assert ref.sample(1, 0.5).metadata["seg"] == 1


class TestFrames:
    def test_dense_sampling(self):
        ref = build_reference(
            Point5Task(name="t", durations=(0.5,), keyframes=(
                _kf(lambda i, n: (0, 0, i)),
                _kf(lambda i, n: (1, 0, i)))),
            _world_resolver())
        fr = ref.frames(0.1)
        assert fr.pos.shape[1:] == (5, 3)
        assert fr.times[-1] == pytest.approx(0.5)
        np.testing.assert_allclose(fr.pos[-1][0], [1, 0, 0])
        assert len(fr.states) == fr.pos.shape[0]

    def test_frames_bad_dt(self):
        ref = build_reference(
            Point5Task(name="t", durations=(0.5,), keyframes=(
                _kf(lambda i, n: (0, 0, i)),
                _kf(lambda i, n: (1, 0, i)))), _world_resolver())
        with pytest.raises(TaskValidationError):
            ref.frames(0.0)


class TestAdaptiveSample:
    def _task(self):
        # head travels 0.25m in seg0, then static 0.4s in seg1
        def mk(head):
            def f(i, n):
                return head if n == "head" else (0, 0, i)
            return _kf(f)
        return Point5Task(name="t", durations=(0.2, 0.4), keyframes=(
            mk((0, 0, 0)), mk((0.25, 0, 0)), mk((0.25, 0, 0))))

    def test_per_frame_step_bounded(self):
        ref = build_reference(self._task(), _world_resolver())
        fr = adaptive_sample(ref, max_step=0.01)
        d = np.linalg.norm(np.diff(fr.pos, axis=0), axis=2)  # (T-1,5)
        assert d.size and float(d.max()) <= 0.01 + 1e-9

    def test_static_segment_collapses(self):
        ref = build_reference(self._task(), _world_resolver())
        fr = adaptive_sample(ref, max_step=0.01)
        # seg1 is static: emits only its two boundaries (kf1 at s=0, kf2 at s=1)
        seg1 = fr.states[fr.segs.tolist().index(1):]
        assert len(seg1) == 2
        np.testing.assert_allclose(seg1[0].pos, seg1[1].pos)

    def test_boundary_not_duplicated(self):
        ref = build_reference(self._task(), _world_resolver())
        fr = adaptive_sample(ref, max_step=0.01)
        assert len(np.unique(fr.times)) == len(fr.times)

    def test_max_step_validation(self):
        ref = build_reference(self._task(), _world_resolver())
        with pytest.raises(TaskValidationError):
            adaptive_sample(ref, max_step=0.0)


class TestFramesPayload:
    def test_payload_format_and_jsonable(self):
        import json
        ref = build_reference(self._task(), _world_resolver())
        payload = frames_payload(ref, adaptive_sample(ref))
        assert payload["format"] == "point5.frames.v1"
        assert payload["frames"][0]["pos"][0] == [0.0, 0.0, 0.0]
        json.dumps(payload)  # must be JSON-safe

    def _task(self):
        return Point5Task(name="t", durations=(0.5,), keyframes=(
            _kf(lambda i, n: (0, 0, i)),
            _kf(lambda i, n: (0.1, 0, i))))


class TestExportHtml:
    def test_writes_player(self, tmp_path):
        from point5 import export_html
        ref = build_reference(
            Point5Task(name="t", durations=(0.5,), keyframes=(
                _kf(lambda i, n: (0, 0, i)),
                _kf(lambda i, n: (0.2, 0, i)))),
            _world_resolver())
        out = export_html(ref, tmp_path / "t.html")
        text = out.read_text()
        assert "point5.frames.v1" in text
        assert '"hand_right"' in text
        assert "<canvas" in text


class TestExportMp4:
    def test_writes_video(self, tmp_path):
        from point5 import export_mp4
        ref = build_reference(
            Point5Task(name="t", durations=(0.5,), keyframes=(
                _kf(lambda i, n: (0, 0, i)),
                _kf(lambda i, n: (0.2, 0, i)))),
            _world_resolver())
        out = export_mp4(ref, tmp_path / "t.mp4", fps=10)
        head = out.read_bytes()[:16]
        assert b"ftyp" in head          # MP4 container magic
        assert out.stat().st_size > 2000


class TestTorsoYaw:
    def test_yaw_linear_interpolation(self):
        kf0 = Keyframe(specs={n: PointSpec(WorldPoint(np.zeros(3)))
                              for n in POINT_NAMES}, torso_yaw=0.0)
        kf1 = Keyframe(specs={n: PointSpec(WorldPoint(np.ones(3)))
                              for n in POINT_NAMES}, torso_yaw=90.0)
        ref = build_reference(
            Point5Task("yaw", (kf0, kf1), (1.0,)), _world_resolver())
        assert ref.sample(0, 0.5).torso_yaw == pytest.approx(45.0)

    def test_yaw_none_when_either_end_missing(self):
        kf0 = Keyframe(specs={n: PointSpec(WorldPoint(np.zeros(3)))
                              for n in POINT_NAMES}, torso_yaw=0.0)
        kf1 = Keyframe(specs={n: PointSpec(WorldPoint(np.ones(3)))
                              for n in POINT_NAMES})
        ref = build_reference(
            Point5Task("yaw2", (kf0, kf1), (1.0,)), _world_resolver())
        assert ref.sample(0, 0.5).torso_yaw is None

    def test_yaw_serde_roundtrip(self):
        kf = Keyframe(specs={n: PointSpec(WorldPoint(np.zeros(3)))
                             for n in POINT_NAMES}, torso_yaw=30.0)
        assert Keyframe.from_dict(kf.to_dict()).torso_yaw == \
            pytest.approx(30.0)
        kf2 = Keyframe(specs={n: PointSpec(WorldPoint(np.zeros(3)))
                              for n in POINT_NAMES})
        assert Keyframe.from_dict(kf2.to_dict()).torso_yaw is None

    def test_state_yaw_validation(self):
        with pytest.raises(TaskValidationError):
            Point5State(pos=np.zeros((5, 3)), torso_yaw=float("nan"))
        st = Point5State(pos=np.zeros((5, 3)), torso_yaw=15.0)
        assert st.torso_yaw == pytest.approx(15.0)
