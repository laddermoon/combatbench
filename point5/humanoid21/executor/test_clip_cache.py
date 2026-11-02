"""E2 verification: anchoring, world-target preservation, cache ops."""
import numpy as np
import pytest

from point5.contracts import Keyframe, P5Clip, Transfer, to_local
from point5.humanoid21.executor.clip_cache import (
    InstructionCache, build_stand_punch_clip, _rel_frame)
from point5.humanoid21.stand_punch import stand_p5, plan_stand_punch
from point5.contracts import Point5State, ankle_mid_xy
from point5.trajectory import keyframe_world
from point5.humanoid21.executor.contracts import AnchorError


def _cur(dx=0.0, yaw=0.0):
    pos = stand_p5()
    origin = ankle_mid_xy(pos) + np.array([dx, 0.0])
    p = Point5State(pos=to_local(pos, origin, yaw))
    object.__setattr__(p, "metadata", dict(p.metadata) | {
        "frame_origin_xy": origin.tolist(), "frame_yaw_deg": yaw})
    return p


def _clip(cur):
    return build_stand_punch_clip(cur, np.array([1.05, 0, 1.15]),
                                  "right")


def test_builder_prepend():
    cur = _cur()
    c = _clip(cur)
    assert [k.label for k in c.keyframes] == \
        ["start", "coil", "swing", "land", "strike"]
    np.testing.assert_allclose(c.keyframes[0].pos, cur.pos, atol=0)


def test_submit_anchor_check():
    cur, cache = _cur(), InstructionCache()
    cache.submit(_clip(cur), cur)                    # ok
    bad = _clip(cur)
    kf = bad.keyframes[0]
    moved = Keyframe(pos=kf.pos + np.array(
        [[0, 0, 0], [0.01, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        label=kf.label)
    bad2 = P5Clip(name="b", keyframes=(moved,) + bad.keyframes[1:])
    with pytest.raises(AnchorError):
        cache.submit(bad2, cur)


def test_world_targets_preserved_on_reanchor():
    """Simulated tracking error must not move future world targets."""
    cur = _cur()
    cache = InstructionCache()
    sub = cache.submit(_clip(cur), cur)
    w_before = sub.world_pts.copy()
    shifted = _cur(dx=0.08, yaw=12.0)          # drifted actual state
    cache.update_head(shifted)
    np.testing.assert_allclose(cache.sub.world_pts, w_before,
                               atol=1e-12)
    tr = cache.remaining_transfer()
    assert tr is not None
    # remaining transfer composes head frame -> next world frame
    assert abs(np.deg2rad(tr.dyaw) - np.deg2rad(
        sub.world_frames[1][1] - 12.0)) < 1e-6


def test_se2_rel_frame():
    a = (np.array([1.0, 2.0]), 30.0)
    b = (np.array([1.0 + np.cos(np.pi / 6) * 0.5, 2.0 +
                   np.sin(np.pi / 6) * 0.5]), 80.0)
    tr = _rel_frame(a, b)
    assert tr.dx == pytest.approx(0.5) and abs(tr.dy) < 1e-9
    assert tr.dyaw == pytest.approx(50.0)


def test_clear_and_keep():
    cur = _cur()
    cache = InstructionCache()
    cache.submit(_clip(cur), cur)
    cache.clear()
    assert cache.sub is None and cache.remaining_transfer() is None
