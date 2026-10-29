"""contracts.py v2 semantics tests."""
import numpy as np
import pytest

from point5.contracts import (
    Keyframe,
    P5Clip,
    Point5State,
    TaskValidationError,
    Transfer,
    ankle_mid_xy,
    compose_frame,
    is_degenerate,
    to_local,
    to_world,
    twist_deg,
)


def _pose(feet=((0., -0.15, 0.05), (0., 0.15, 0.05)),
          head=(0., 0., 1.3), hr=(-0.1, -0.2, 0.8), hl=(0.1, 0.2, 0.8)):
    return np.array([head, hr, hl, feet[0], feet[1]])


def test_state_accepts_5x3():
    st = Point5State(pos=_pose())
    assert st.pos.shape == (5, 3)
    assert st.point_pos("head").tolist() == [0., 0., 1.3]


def test_state_rejects_bad_shape():
    with pytest.raises(TaskValidationError):
        Point5State(pos=np.zeros((4, 3)))


def test_ankle_mid_and_twist():
    st = _pose()
    np.testing.assert_allclose(ankle_mid_xy(st), [0., 0.])
    assert twist_deg(st) == pytest.approx(0.0)   # feet straddle Y axis


def test_twist_from_tilted_ankle_line():
    # feet rotated 30deg: ankle-line normal tilts off +X
    a = np.deg2rad(30.0)
    c, s = np.cos(a), np.sin(a)
    p = _pose()
    for i in (3, 4):
        p[i, :2] = p[i, :2] @ np.array([[c, -s], [s, c]]).T
    tw = twist_deg(p)
    assert tw is not None and abs(tw) == pytest.approx(30.0)


def test_degenerate_feet():
    p = _pose(feet=((0., 0., 0.05), (0., 0.005, 0.05)))
    assert is_degenerate(p)
    assert twist_deg(p) is None


def test_world_local_roundtrip():
    p = _pose()
    w = to_world(p, [1.0, 2.0], 37.0)
    np.testing.assert_allclose(to_local(w, [1.0, 2.0], 37.0), p, atol=1e-12)


def test_compose_frame():
    o, y = compose_frame(np.zeros(2), 0.0,
                         Transfer(dx=0.3, dy=0.1, dyaw=45.0, dur=0.5))
    np.testing.assert_allclose(o, [0.3, 0.1])
    assert y == 45.0
    o2, y2 = compose_frame(o, y, Transfer(0.1, 0.0, 0.0, 0.5))
    np.testing.assert_allclose(o2, [0.3 + 0.1 * np.cos(np.pi / 4)
                                    - 0.0, 0.1 + 0.1 * np.sin(np.pi / 4)],
                               atol=1e-9)


def test_transfer_validation():
    with pytest.raises(TaskValidationError):
        Transfer(0.0, 0.0, 0.0, dur=0.0)


def test_clip_validation():
    k0 = Keyframe(pos=_pose())
    k1 = Keyframe(pos=_pose(), transfer=Transfer(0.1, 0., 0., 0.4))
    clip = P5Clip("t", (k0, k1))
    assert clip.n_segments == 1 and clip.duration == 0.4
    # first frame must not carry transfer
    with pytest.raises(TaskValidationError):
        P5Clip("t", (k1, k0))
    # later frames must carry transfer
    with pytest.raises(TaskValidationError):
        P5Clip("t", (k0, Keyframe(pos=_pose())))


def test_serde_roundtrip():
    clip = P5Clip("c", (Keyframe(pos=_pose()),
                        Keyframe(pos=_pose(), label="b",
                                 transfer=Transfer(0.1, 0.2, 30.0, 0.5))))
    clip2 = P5Clip.from_dict(clip.to_dict())
    np.testing.assert_allclose(clip2.keyframes[1].pos, clip.keyframes[1].pos)
    assert clip2.keyframes[1].transfer.dyaw == 30.0
