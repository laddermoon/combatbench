"""Tests for the pure mechanical model (mechanics.py)."""
from __future__ import annotations

import numpy as np
import pytest

from point5.humanoid21.mechanics import (
    JOINT_NAMES, LANDMARK_ORDER, Skeleton,
    norm_to_rad, rad_to_norm, spring_energy, solve_p5_pose,
    solve_support_pose)

_ANKLE_CENTER = ("ankle_y_right", "ankle_y_left")


@pytest.fixture(scope="module")
def sk():
    return Skeleton()


def _planted_zero_target(sk):
    """五点目标：无载姿态，脚槽放踝心（ankle_y 锚点），P5 语义。"""
    q0 = np.zeros(21)
    lo, hi = 0.2, 1.2
    for _ in range(25):
        m = (lo + hi) / 2
        a, _ = sk.anchors_jac([0, 0, m], [1, 0, 0, 0], q0, _ANKLE_CENTER)
        if a[:, 2].min() < 0.087:
            lo = m
        else:
            hi = m
    m = (lo + hi) / 2
    lm = sk.landmarks([0, 0, m], [1, 0, 0, 0], q0)
    a, _ = sk.anchors_jac([0, 0, m], [1, 0, 0, 0], q0, _ANKLE_CENTER)
    lm[3], lm[4] = a
    return lm


class TestStructure:
    def test_counts(self, sk):
        assert len(JOINT_NAMES) == 21
        assert len(LANDMARK_ORDER) == 5
        assert len(sk.order) == 16

    def test_norm_roundtrip(self):
        q = np.linspace(-1, 1, 21)
        np.testing.assert_allclose(rad_to_norm(norm_to_rad(q)), q)

    def test_zero_pose_is_unloaded(self, sk):
        assert spring_energy(np.zeros(21)) == pytest.approx(0.0)


class TestSprings:
    def test_laws(self):
        # spring table sanity: stiff linear core, quadratic absorbers
        from point5.humanoid21.mechanics import SPRINGS
        assert SPRINGS["abdomen_y"][0] == "linear"
        assert SPRINGS["knee_right"][0] == "quadratic"
        assert SPRINGS["elbow_right"][0] == "linear"

    def test_quadratic_soft_midrange(self):
        i = JOINT_NAMES.index("knee_right")
        q1 = np.zeros(21); q1[i] = 0.1
        q2 = np.zeros(21); q2[i] = 0.9
        assert spring_energy(q1) < spring_energy(q2) / 100.0


class TestP5Pose:
    def test_zero_target_recovers(self, sk):
        tgt = _planted_zero_target(sk)
        res = solve_p5_pose(tgt, skeleton=sk)
        assert res["satisfied"]
        assert res["point_errors"].max() < 0.04

    def test_deterministic(self, sk):
        tgt = _planted_zero_target(sk)
        a = solve_p5_pose(tgt, skeleton=sk)
        b = solve_p5_pose(tgt, skeleton=sk)
        np.testing.assert_allclose(a["q_norm"], b["q_norm"], atol=1e-6)

    def test_unreachable_reports(self, sk):
        tgt = _planted_zero_target(sk)
        tgt[0, 2] += 3.0                       # head 3m up: impossible
        res = solve_p5_pose(tgt, skeleton=sk)
        assert not res["satisfied"]
        assert not res["stage1"]["converged"]

    def test_crouch_legs_absorb(self, sk):
        """Lowering head: legs (hip/knee) should absorb, torso ~stays."""
        tgt = _planted_zero_target(sk)
        tgt[0, 2] -= 0.4
        res = solve_p5_pose(tgt, skeleton=sk)
        assert res["satisfied"]
        qn = res["q_norm"]
        i_abd = JOINT_NAMES.index("abdomen_y")
        i_knee = JOINT_NAMES.index("knee_right")
        i_hip = JOINT_NAMES.index("hip_y_right")
        # knee/hip flex substantially; abdomen stays near 0
        assert abs(qn[i_abd]) < 0.15
        assert abs(qn[i_knee]) > 0.3 or abs(qn[i_hip]) > 0.3


def _support_target(sk, q=None, root_z=1.28):
    """骨架自洽目标：已知 q -> FK -> com/双踝心（纯骨架，无 MuJoCo）。"""
    if q is None:
        q = np.zeros(21)
    com, _, a, *_ = sk.support_jac([0, 0, root_z], [1, 0, 0, 0],
                                   q, _ANKLE_CENTER)
    return com, a[0], a[1]


class TestSupportPose:
    def test_zero_target_recovers(self, sk):
        com, ar, al = _support_target(sk)
        res = solve_support_pose(com, ar, al, skeleton=sk)
        assert res["satisfied"]
        np.testing.assert_allclose(res["com"], com, atol=0.03)
        np.testing.assert_allclose(res["ankles"][0], ar, atol=0.05)
        np.testing.assert_allclose(res["ankles"][1], al, atol=0.05)

    def test_deterministic(self, sk):
        com, ar, al = _support_target(sk)
        a = solve_support_pose(com, ar, al, skeleton=sk)
        b = solve_support_pose(com, ar, al, skeleton=sk)
        np.testing.assert_allclose(a["q_norm"], b["q_norm"], atol=1e-6)

    def test_unreachable_reports(self, sk):
        com, ar, al = _support_target(sk)
        res = solve_support_pose(com + [0, 0, 3.0], ar, al, skeleton=sk)
        assert not res["satisfied"]
        assert not res["stage1"]["converged"]

    def test_bent_legs_recovers(self, sk):
        q = np.zeros(21)
        jn = {n: i for i, n in enumerate(JOINT_NAMES)}
        q[jn["knee_right"]] = q[jn["knee_left"]] = -0.6
        q[jn["hip_y_right"]] = q[jn["hip_y_left"]] = 0.35
        q[jn["ankle_y_right"]] = q[jn["ankle_y_left"]] = 0.25
        com, ar, al = _support_target(sk, q, root_z=1.12)
        res = solve_support_pose(com, ar, al, skeleton=sk)
        assert res["satisfied"]
        np.testing.assert_allclose(res["com"], com, atol=0.03)
        # 目标项生效：|头-com| 显著大于塌缩值（~0.2）
        assert res["head_com_dist"] > 0.45


class TestCPose:
    """CPose <-> Pose 适配（纯骨架，无 MuJoCo）。约定见 mechanics
    "CPose 适配" 段。"""

    def test_ext_within_one(self, sk):
        """LMax 语义：任意姿态提取的肢向量伸展度 <=1。"""
        from point5.humanoid21.mechanics import pose_to_cpose
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.uniform(-0.6, 0.6, 21)
            cp = pose_to_cpose(q, [0, 0, 1.0], [1, 0, 0, 0], skeleton=sk)
            for n in ("left_hand", "right_hand",
                      "left_ankle", "right_ankle"):
                assert np.linalg.norm(getattr(cp, n)) <= 1.0 + 1e-9, n

    def test_roundtrip_fields(self, sk):
        """c -> pose -> c' 字段级 roundtrip——CPose 未表达自由度
        （髋自旋/踝）由 infer 先验吸收，字段本身须自洽还原。"""
        from point5.humanoid21.mechanics import (
            cpose_to_pose, pose_to_cpose)
        from point5.humanoid21.robot_state import _zyx_to_quat_wxyz
        from point5.humanoid21.mechanics import JNT_RANGE
        rng = np.random.default_rng(1)
        jn = {n: i for i, n in enumerate(JOINT_NAMES)}
        for _ in range(20):
            # 限位内采样 → 目标可达；CPose 未表达的自由度取先验值
            q = JNT_RANGE[:, 0] + rng.random(21) * np.diff(
                JNT_RANGE, axis=1)[:, 0]
            for side in ("left", "right"):
                q[jn[f"hip_x_{side}"]] = 0.0
            cp1 = pose_to_cpose(q, [0.1, -0.2, 1.0], [1, 0, 0, 0],
                                skeleton=sk)
            res = cpose_to_pose(cp1, skeleton=sk)
            cp2 = pose_to_cpose(res["qpos"], res["root_pos"],
                                _zyx_to_quat_wxyz(res["root_zyx"]),
                                skeleton=sk)
            for n in ("left_hand", "right_hand", "left_ankle",
                      "right_ankle"):
                np.testing.assert_allclose(         # 闭式 IK 分支等价差
                    getattr(cp2, n), getattr(cp1, n), atol=3e-5,
                    err_msg=n)
            for n in ("torso_pry", "pelvis_pry", "pelvis_pos"):
                np.testing.assert_allclose(
                    getattr(cp2, n), getattr(cp1, n), atol=1e-9,
                    err_msg=n)

    def test_pelvis_pose_recovered(self, sk):
        """root 反推：infer 结果的 pelvis 位姿 == CPose 指定值。"""
        from point5 import CPose
        from point5.humanoid21.mechanics import (
            cpose_to_pose, pose_to_cpose, _zyx_to_mat)
        from point5.humanoid21.robot_state import _zyx_to_quat_wxyz
        base = pose_to_cpose(np.zeros(21), [0, 0, 1.0], [1, 0, 0, 0],
                             skeleton=sk)
        cp = CPose(left_hand=base.left_hand, right_hand=base.right_hand,
                   left_ankle=base.left_ankle, right_ankle=base.right_ankle,
                   torso_pry=base.torso_pry,
                   pelvis_pry=np.array([0.12, -0.08, 0.4]),
                   pelvis_pos=np.array([0.3, 0.1, 0.85]))
        res = cpose_to_pose(cp, skeleton=sk)
        xf = sk.body_transforms(res["root_pos"],
                                _zyx_to_quat_wxyz(res["root_zyx"]),
                                res["qpos"])
        np.testing.assert_allclose(xf["pelvis"][0], cp.pelvis_pos,
                                   atol=1e-9)
        np.testing.assert_allclose(xf["pelvis"][1],
                                   _zyx_to_mat(cp.pelvis_pry), atol=1e-9)

    def test_torso_pry_one_to_one(self, sk):
        """躯干 1:1 映射：abdomen_y/x/z = (-pitch,-roll,-yaw)。"""
        from point5.humanoid21.mechanics import (
            cpose_to_pose, pose_to_cpose)
        jn = {n: i for i, n in enumerate(JOINT_NAMES)}
        q = np.zeros(21)
        q[jn["abdomen_y"]], q[jn["abdomen_x"]], q[jn["abdomen_z"]] = \
            -0.4, -0.3, -0.2
        cp = pose_to_cpose(q, [0, 0, 1.0], [1, 0, 0, 0], skeleton=sk)
        np.testing.assert_allclose(cp.torso_pry, [0.4, 0.3, 0.2])
        res = cpose_to_pose(cp, skeleton=sk)
        for name, want in (("abdomen_y", -0.4), ("abdomen_x", -0.3),
                           ("abdomen_z", -0.2)):
            assert res["qpos"][jn[name]] == pytest.approx(want)
