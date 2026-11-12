"""Tests for cpose.py — CPose 协议核心。"""
from __future__ import annotations

import numpy as np
import pytest

from point5.cpose import (CPose, CPoseClip, LIMB_NAMES, mat_to_pry,
                          pry_to_mat)


def _mk(**kw):
    base = dict(left_hand=[0.2, 0.1, -0.6], right_hand=[0.2, -0.1, -0.6],
                left_ankle=[0.0, 0.1, -0.9], right_ankle=[0.0, -0.1, -0.9])
    base.update(kw)
    return CPose(**base)


class TestPryMat:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            pry = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-3, 3),
                            rng.uniform(-np.pi, np.pi)])
            back = mat_to_pry(pry_to_mat(pry))
            np.testing.assert_allclose(
                pry_to_mat(back), pry_to_mat(pry), atol=1e-12)

    def test_known_rotation(self):
        # yaw=pi/2 纯转：+x 前向 -> +y
        R = pry_to_mat([0.0, 0.0, np.pi / 2])
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        # pitch=+pi/2 前倾到底：+x -> -z
        R = pry_to_mat([np.pi / 2, 0.0, 0.0])
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 0, -1], atol=1e-12)

    def test_gimbal_branch(self):
        # |pitch|=pi/2 奇异：返回合法标准分支且旋转一致
        for p in (np.pi / 2, -np.pi / 2):
            back = mat_to_pry(pry_to_mat([p, 0.7, 1.1]))
            np.testing.assert_allclose(
                pry_to_mat(back), pry_to_mat([p, 0.7, 1.1]), atol=1e-12)


class TestCPose:
    def test_defaults_zero(self):
        p = _mk()
        np.testing.assert_allclose(p.torso_pry, 0)
        np.testing.assert_allclose(p.pelvis_pry, 0)
        np.testing.assert_allclose(p.pelvis_pos, 0)
        np.testing.assert_allclose(p.pelvis_mat, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(p.torso_mat, np.eye(3), atol=1e-12)

    def test_ext_bounds(self):
        with pytest.raises(ValueError):
            _mk(left_hand=[0.0, 0.0, 0.0])          # 最小不为 0
        with pytest.raises(ValueError):
            _mk(left_ankle=[0.0, 0.0, -1.05])       # 最大为 1
        _mk(left_ankle=[0.0, 0.0, -1.0])            # ext=1 合法

    def test_pry_canonicalized(self):
        # 非标准分支 pitch(>pi/2) 被改写为等价标准分支
        p = _mk(torso_pry=[2.0, 0.3, 0.4])
        assert np.cos(p.torso_pry[0]) >= 0.0
        np.testing.assert_allclose(pry_to_mat(p.torso_pry),
                                   pry_to_mat([2.0, 0.3, 0.4]), atol=1e-12)

    def test_torso_mat_composition(self):
        p = _mk(pelvis_pry=[0.0, 0.0, 0.5], torso_pry=[0.1, 0.0, 0.2])
        np.testing.assert_allclose(
            p.torso_mat, pry_to_mat([0, 0, 0.5]) @ pry_to_mat([0.1, 0, 0.2]),
            atol=1e-12)

    def test_exts_dirs_vecs(self):
        p = _mk()
        assert p.vecs.shape == (4, 3)
        assert p.dirs.shape == (4, 3)
        np.testing.assert_allclose(p.exts,
                                   np.linalg.norm(p.vecs, axis=1))
        np.testing.assert_allclose(
            np.linalg.norm(p.dirs, axis=1), 1.0, atol=1e-12)

    def test_vec_host(self):
        # pelvis yaw=pi/2：踝向量 +x -> 所在帧 +y；手向量随 torso
        p = _mk(left_ankle=[0.5, 0.0, -0.5], left_hand=[0.6, 0.0, -0.5],
                pelvis_pry=[0.0, 0.0, np.pi / 2], torso_pry=[0.0, 0.0, 0.0])
        np.testing.assert_allclose(p.vec_host("left_ankle"),
                                   [0.0, 0.5, -0.5], atol=1e-12)
        np.testing.assert_allclose(p.vec_host("left_hand"),
                                   [0.0, 0.6, -0.5], atol=1e-12)

    def test_hand_rides_torso_yaw(self):
        # torso 相对 yaw=pi/2 而 pelvis 不动：手转向、踝不转
        p = _mk(left_hand=[0.6, 0.0, -0.5], left_ankle=[0.5, 0.0, -0.5],
                torso_pry=[0.0, 0.0, np.pi / 2])
        np.testing.assert_allclose(p.vec_host("left_hand"),
                                   [0.0, 0.6, -0.5], atol=1e-12)
        np.testing.assert_allclose(p.vec_host("left_ankle"),
                                   [0.5, 0.0, -0.5], atol=1e-12)


class TestClip:
    def test_sequence(self):
        frames = tuple(_mk() for _ in range(3))
        clip = CPoseClip(frames)
        assert len(clip) == 3
        assert clip[1] is frames[1]
        assert list(clip) == list(frames)
