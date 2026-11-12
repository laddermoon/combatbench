"""Tests for cpose.py — CPose 协议核心。"""
from __future__ import annotations

import numpy as np
import pytest

from point5.cpose import CPose, CPoseClip


def _mk(**kw):
    base = dict(left_hand=[0.2, 0.1, -0.6], right_hand=[0.2, -0.1, -0.6],
                left_ankle=[0.0, 0.1, -0.9], right_ankle=[0.0, -0.1, -0.9],
                torso_pry=[0.0, 0.0, 0.0])
    base.update(kw)
    return CPose(**base)


class TestCPose:
    def test_pelvis_defaults_zero(self):
        p = _mk()
        np.testing.assert_allclose(p.pelvis_pry, 0)
        np.testing.assert_allclose(p.pelvis_pos, 0)

    def test_torso_pry_required(self):
        kw = dict(left_hand=[0.2, 0.1, -0.6], right_hand=[0.2, -0.1, -0.6],
                  left_ankle=[0.0, 0.1, -0.9], right_ankle=[0.0, -0.1, -0.9])
        with pytest.raises(TypeError):
            CPose(**kw)

    def test_ext_bounds(self):
        with pytest.raises(ValueError):
            _mk(left_hand=[0.0, 0.0, 0.0])          # 最小不为 0
        with pytest.raises(ValueError):
            _mk(left_ankle=[0.0, 0.0, -1.05])       # 最大为 1
        _mk(left_ankle=[0.0, 0.0, -1.0])            # ext=1 合法

    def test_pry_stored_as_given(self):
        # 协议不改写输入：非标准分支原样存储
        p = _mk(torso_pry=[2.0, 0.3, 0.4])
        np.testing.assert_allclose(p.torso_pry, [2.0, 0.3, 0.4])

    def test_field_shape_validation(self):
        with pytest.raises(ValueError):
            _mk(left_hand=[0.1, 0.2])               # 形状错
        with pytest.raises(ValueError):
            _mk(torso_pry=[0.0, np.inf, 0.0])       # 非有限

    def test_limb_accessors(self):
        p = _mk(left_hand=[0.6, 0.0, -0.5])
        np.testing.assert_allclose(p.limb("left_hand"), [0.6, 0.0, -0.5])
        np.testing.assert_allclose(p.limb_extension("left_hand"),
                                   np.linalg.norm([0.6, 0.0, -0.5]))

    def test_limb_direction_known(self):
        # (azimuth, polar)：azimuth 0=前 +pi/2=左；polar 0=向头 pi=下垂
        p = _mk(left_hand=[0.5, 0.0, 0.0])
        az, pol = p.limb_direction("left_hand")
        np.testing.assert_allclose([az, pol], [0.0, np.pi / 2])

        p = _mk(left_hand=[0.0, 0.5, 0.0])
        az, pol = p.limb_direction("left_hand")
        np.testing.assert_allclose([az, pol], [np.pi / 2, np.pi / 2])

        p = _mk(left_ankle=[0.0, 0.0, -0.8])
        az, pol = p.limb_direction("left_ankle")
        np.testing.assert_allclose(pol, np.pi)


class TestClip:
    def test_sequence(self):
        frames = tuple(_mk() for _ in range(3))
        clip = CPoseClip(frames)
        assert len(clip) == 3
        assert clip[1] is frames[1]
        assert list(clip) == list(frames)
