"""trajectory.py v2 tests."""
import numpy as np
import pytest

from point5.contracts import Keyframe, P5Clip, Transfer
from point5.trajectory import (
    frames_payload,
    keyframe_frames,
    keyframe_world,
    sample_clip,
)


def _pose(z=0.05):
    return np.array([[0., 0., 1.3], [-0.1, -0.2, 0.8], [0.1, 0.2, 0.8],
                     [0., -0.15, z], [0., 0.15, z]])


def _clip(dx=0.4, dyaw=30.0):
    return P5Clip("c", (
        Keyframe(pos=_pose()),
        Keyframe(pos=_pose() + np.array([0.1, 0., 0.]),
                 transfer=Transfer(dx, 0.0, dyaw, 0.5)),
        Keyframe(pos=_pose() + np.array([0.2, 0., 0.]),
                 transfer=Transfer(dx, 0.0, 0.0, 0.5)),
    ))


def test_keyframe_frames_chain():
    frames = keyframe_frames(_clip())
    assert len(frames) == 3
    np.testing.assert_allclose(frames[0][0], [0., 0.])
    assert frames[0][1] == 0.0
    np.testing.assert_allclose(frames[1][0], [0.4, 0.])
    assert frames[1][1] == 30.0
    # frame2 origin = frame1 origin + R(30deg)@[0.4,0]
    c = np.cos(np.deg2rad(30)); s = np.sin(np.deg2rad(30))
    np.testing.assert_allclose(frames[2][0], [0.4 + 0.4 * c, 0.4 * s])
    assert frames[2][1] == 30.0


def test_keyframe_world_shapes():
    w = keyframe_world(_clip())
    assert w.shape == (3, 5, 3)


def test_sample_bound():
    samples = sample_clip(_clip(), max_step=0.01)
    world = np.stack([s.world_pos() for s in samples])
    step = np.linalg.norm(np.diff(world, axis=0), axis=2).max()
    assert step <= 0.01 + 1e-9
    assert samples[0].time == 0.0
    assert samples[-1].time == pytest.approx(1.0)


def test_sample_static_segment_collapses():
    clip = P5Clip("s", (Keyframe(pos=_pose()),
                        Keyframe(pos=_pose(),
                                 transfer=Transfer(0., 0., 0., 0.5))))
    # same pose but zero-motion transfer still must exist on frame 2
    # -> segment has zero displacement; still >=1 sample emitted
    samples = sample_clip(clip, max_step=0.05)
    assert len(samples) == 2


def test_frames_payload():
    samples = sample_clip(_clip(), max_step=0.05)
    pl = frames_payload(_clip(), samples)
    assert pl["format"] == "point5.frames.v2"
    assert pl["n_frames"] == len(samples)
    assert "origin" in pl["frames"][0] and "yaw" in pl["frames"][0]
