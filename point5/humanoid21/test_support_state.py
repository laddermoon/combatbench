"""Tests for support_state.py — passive_drift 与支撑几何工具。"""
from __future__ import annotations

import numpy as np
import pytest

from point5.humanoid21.support_state import (
    ANKLE_Z_PLANTED, LIP_G, LIP_MASS, SupportGeometry, SupportState)


def _st(com, lf, rf, lfh=0.0, rfh=0.0, yaw=0.0,
        v=(0, 0, 0), H=(0, 0, 0)):
    return SupportState(com=np.asarray(com, float),
        left_foot=np.asarray(lf, float),
        right_foot=np.asarray(rf, float), torso_yaw=yaw,
        left_foot_height=lfh, right_foot_height=rfh,
        com_vel=np.asarray(v, float), ang_mom=np.asarray(H, float))


class TestPassiveDrift:
    def test_balanced_hinge_no_drift(self):
        r = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).passive_drift()
        assert r.mode == "hinge"
        assert r.assumption_ok
        np.testing.assert_allclose(r.state.com, [0, 0, 0.95], atol=1e-9)
        np.testing.assert_allclose(r.state.com_vel, 0, atol=1e-9)
        assert r.min_normal == pytest.approx(LIP_MASS * LIP_G, rel=0.01)

    def test_eccentric_falls_forward(self):
        r = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).passive_drift()
        assert r.mode == "hinge"
        s = r.state
        assert s.com[0] > 0.10            # 向偏心方向漂
        assert s.com[2] < 0.95            # 倒下：com 下降
        assert s.com_vel[0] > 0.0

    def test_single_pivot_falls_faster_than_hinge(self):
        """自由 2 轴铰约束更少 → 同偏心下漂移更大。"""
        a = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.30, -0.30], rfh=0.25
                ).passive_drift()
        b = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).passive_drift()
        assert a.mode == "pivot" and b.mode == "hinge"
        assert np.linalg.norm(a.state.com_vel) \
            > np.linalg.norm(b.state.com_vel)

    def test_ballistic_free_fall(self):
        r = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09],
                lfh=0.2, rfh=0.2, v=(0.5, 0, 0.3)).passive_drift()
        assert r.mode == "ballistic"
        s = r.state
        np.testing.assert_allclose(
            s.com, [0.5 * 0.05, 0, 0.95 + 0.3 * 0.05
                    - 0.5 * LIP_G * 0.05 ** 2], atol=1e-9)
        np.testing.assert_allclose(
            s.com_vel, [0.5, 0, 0.3 - LIP_G * 0.05], atol=1e-9)
        np.testing.assert_allclose(s.ang_mom, 0)   # 无矩角动量守恒

    def test_radial_violation_flagged(self):
        """v 直指远离支点（腿要变长）→ 违反贴地约束，被标记。"""
        r = _st([0, 0, 0.95], [0, 0], [0.3, -0.3], rfh=0.25,
                v=(0, 0, 1.0)).passive_drift()
        assert r.mode == "pivot"
        assert r.radial_violation == pytest.approx(1.0)
        assert not r.assumption_ok

    def test_tangential_velocity_consistent(self):
        """切向速度合法：不违例，且漂移延续速度方向。"""
        r = _st([0, 0, 0.95], [0, 0], [0.3, -0.3], rfh=0.25,
                v=(0.3, 0, 0)).passive_drift()
        assert r.radial_violation == pytest.approx(0.0)
        assert r.assumption_ok
        assert r.state.com_vel[0] > 0.3   # 重力再加速

    def test_inertia_slows_and_updates_H(self):
        J = np.diag([1.2, 1.5, 0.3])
        r = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).passive_drift(inertia_com=J)
        assert np.linalg.norm(r.state.ang_mom) > 0.0
        # 无惯量版本不退化、方向一致
        r0 = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                 ).passive_drift()
        assert np.sign(r.state.com_vel[0]) == np.sign(r0.state.com_vel[0])

    def test_output_is_supportstate(self):
        r = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).passive_drift()
        assert isinstance(r.state, SupportState)
        np.testing.assert_allclose(r.state.left_foot,
                                   [-0.05, 0.09])

    def test_swing_recontact_ignored_but_reported(self):
        """规则：摆脚再触地不进动力学（模式不切换、脚高原样）；
        诊断：倾倒方向的摆脚轨迹最低间隙 <0 = 本会触地。"""
        st = _st([0.15, 0, 0.95], [0.0, 0.05], [0.45, -0.1],
                 rfh=0.08, v=(0.6, 0, 0))
        r = st.passive_drift(dt=0.3)
        assert r.mode == "pivot"
        assert r.swing_min_clearance < 0.0        # 本会穿地
        # 脚位/足高不受触地影响、原样保留
        assert r.state.right_foot_height == 0.08
        np.testing.assert_allclose(r.state.right_foot, [0.45, -0.1])
        # 反向情形：摆脚在支点后方，前倒时被抬起 → 间隙不降
        r2 = _st([0.15, 0, 0.95], [0.0, 0.05], [-0.3, -0.15],
                 rfh=0.08, v=(0.6, 0, 0)).passive_drift(dt=0.3)
        assert r2.swing_min_clearance > 0.0

    def test_swing_clearance_nan_non_pivot(self):
        st = _st([0.1, 0, 0.95], [-0.05, 0.09], [0.05, -0.09])
        assert np.isnan(st.passive_drift().swing_min_clearance)


def _aim(l, r):
    return SupportGeometry(com=np.zeros(3),
        left_foot=np.asarray(l, float),
        right_foot=np.asarray(r, float), torso_yaw=0.0)


def _cur():
    return _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09])


class TestPathTo:
    def test_direct_reach_two_items(self):
        seq = _cur().path_to(_aim((-0.05, 0.09), (0.45, -0.09)))
        assert len(seq) == 2
        np.testing.assert_allclose(seq[0][1], [0.05, -0.09])
        np.testing.assert_allclose(seq[1][1], [0.45, -0.09])

    def test_far_multi_step(self):
        seq = _cur().path_to(_aim((1.95, 0.09), (2.05, -0.09)))
        assert len(seq) > 2
        # 首项=当前 末项=aim
        np.testing.assert_allclose(seq[0][0], [-0.05, 0.09])
        np.testing.assert_allclose(seq[-1][0], [1.95, 0.09])
        np.testing.assert_allclose(seq[-1][1], [2.05, -0.09])
        # 每转换：恰好一脚动、落点在不动脚 step_radius 内
        for i in range(1, len(seq)):
            l0, r0 = seq[i - 1]
            l1, r1 = seq[i]
            dl = np.linalg.norm(l1 - l0)
            dr = np.linalg.norm(r1 - r0)
            assert (dl > 1e-9) + (dr > 1e-9) == 1
            reach = (np.linalg.norm(l1 - r0) if dl > 1e-9
                     else np.linalg.norm(r1 - l0))
            assert reach <= 0.6 + 1e-9

    def test_aim_stance_too_wide_raises(self):
        with pytest.raises(ValueError):
            _cur().path_to(_aim((0, 0.6), (0, -0.6)))

    def test_sidestep(self):
        seq = _cur().path_to(_aim((-0.05, 0.49), (0.05, 0.31)))
        assert len(seq) >= 2
        np.testing.assert_allclose(seq[-1][0], [-0.05, 0.49])
