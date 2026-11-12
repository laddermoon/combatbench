"""Tests for support_state.py — drift 与支撑几何工具。"""
from __future__ import annotations

import numpy as np
import pytest

from point5.humanoid21.support_state import (
    ANKLE_Z_PLANTED, LIP_G, LIP_MASS, SupportClass,
    SupportGeometry, SupportState)


def _st(com, lf, rf, lfh=0.0, rfh=0.0, yaw=0.0,
        v=(0, 0, 0), H=(0, 0, 0)):
    return SupportState(com=np.asarray(com, float),
        left_foot=np.asarray(lf, float),
        right_foot=np.asarray(rf, float), torso_yaw=yaw,
        left_foot_height=lfh, right_foot_height=rfh,
        com_vel=np.asarray(v, float), ang_mom=np.asarray(H, float))


class TestSupportClass:
    """support_class：按贴地脚数分类（分治 dispatch key）。"""

    def test_four_cases(self):
        assert _st([0, 0, 1], [0, 0.1], [0, -0.1]).support_class() \
            == SupportClass.DOUBLE
        assert _st([0, 0, 1], [0, 0.1], [0, -0.1],
                   rfh=0.15).support_class() == SupportClass.SINGLE_LEFT
        assert _st([0, 0, 1], [0, 0.1], [0, -0.1],
                   lfh=0.15).support_class() == SupportClass.SINGLE_RIGHT
        assert _st([0, 0, 1], [0, 0.1], [0, -0.1], lfh=0.15,
                   rfh=0.15).support_class() == SupportClass.AIRBORNE

    def test_eps_boundary(self):
        """阈值语义 = 与 support_polygon 同源的 foot_height < eps。"""
        st = _st([0, 0, 1], [0, 0.1], [0, -0.1], rfh=0.02)
        assert st.support_class() == SupportClass.SINGLE_LEFT
        assert st.support_class(foot_height_eps=0.03) \
            == SupportClass.DOUBLE


class TestDrift:
    def test_balanced_hinge_no_drift(self):
        r = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).drift()
        assert r.mode == "hinge"
        assert r.assumption_ok
        np.testing.assert_allclose(r.state.com, [0, 0, 0.95], atol=1e-9)
        np.testing.assert_allclose(r.state.com_vel, 0, atol=1e-9)
        assert r.min_normal == pytest.approx(LIP_MASS * LIP_G, rel=0.01)

    def test_eccentric_falls_forward(self):
        r = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).drift()
        assert r.mode == "hinge"
        s = r.state
        assert s.com[0] > 0.10            # 向偏心方向漂
        assert s.com[2] < 0.95            # 倒下：com 下降
        assert s.com_vel[0] > 0.0

    def test_single_pivot_falls_faster_than_hinge(self):
        """自由 2 轴铰约束更少 → 同偏心下漂移更大。"""
        a = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.30, -0.30], rfh=0.25
                ).drift()
        b = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).drift()
        assert a.mode == "pivot" and b.mode == "hinge"
        assert np.linalg.norm(a.state.com_vel) \
            > np.linalg.norm(b.state.com_vel)

    def test_ballistic_free_fall(self):
        r = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09],
                lfh=0.2, rfh=0.2, v=(0.5, 0, 0.3)).drift()
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
                v=(0, 0, 1.0)).drift()
        assert r.mode == "pivot"
        assert r.radial_violation == pytest.approx(1.0)
        assert not r.assumption_ok

    def test_tangential_velocity_consistent(self):
        """切向速度合法：不违例，且漂移延续速度方向。"""
        r = _st([0, 0, 0.95], [0, 0], [0.3, -0.3], rfh=0.25,
                v=(0.3, 0, 0)).drift()
        assert r.radial_violation == pytest.approx(0.0)
        assert r.assumption_ok
        assert r.state.com_vel[0] > 0.3   # 重力再加速

    def test_inertia_slows_and_updates_H(self):
        J = np.diag([1.2, 1.5, 0.3])
        r = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).drift(inertia_com=J)
        assert np.linalg.norm(r.state.ang_mom) > 0.0
        # 无惯量版本不退化、方向一致
        r0 = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                 ).drift()
        assert np.sign(r.state.com_vel[0]) == np.sign(r0.state.com_vel[0])

    def test_output_is_supportstate(self):
        r = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09]
                ).drift()
        assert isinstance(r.state, SupportState)
        np.testing.assert_allclose(r.state.left_foot,
                                   [-0.05, 0.09])

    def test_swing_recontact_ignored_but_reported(self):
        """规则：摆脚再触地不进动力学（模式不切换、脚高原样）；
        诊断：倾倒方向的摆脚轨迹最低间隙 <0 = 本会触地。"""
        st = _st([0.15, 0, 0.95], [0.0, 0.05], [0.45, -0.1],
                 rfh=0.08, v=(0.6, 0, 0))
        r = st.drift(dt=0.3)
        assert r.mode == "pivot"
        assert r.swing_min_clearance < 0.0        # 本会穿地
        # 脚位/足高不受触地影响、原样保留
        assert r.state.right_foot_height == 0.08
        np.testing.assert_allclose(r.state.right_foot, [0.45, -0.1])
        # 反向情形：摆脚在支点后方，前倒时被抬起 → 间隙不降
        r2 = _st([0.15, 0, 0.95], [0.0, 0.05], [-0.3, -0.15],
                 rfh=0.08, v=(0.6, 0, 0)).drift(dt=0.3)
        assert r2.swing_min_clearance > 0.0

    def test_swing_clearance_nan_non_pivot(self):
        st = _st([0.1, 0, 0.95], [-0.05, 0.09], [0.05, -0.09])
        assert np.isnan(st.drift().swing_min_clearance)


class TestDriftAlpha:
    """alpha = 踝矩通道角加速度输入。"""

    def test_alpha_opposes_pivot_fall(self):
        """com 偏 +x → 重力矩 +y 前倒；alpha=(0,-k,0) 顶住 → 落得慢。"""
        st = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.30, -0.30],
                 rfh=0.25)
        r0 = st.drift(dt=0.2)
        r1 = st.drift(dt=0.2, alpha=(0, -2.0, 0))
        assert r0.mode == r1.mode == "pivot"
        assert r1.state.com[2] > r0.state.com[2]      # 高度保住更多
        assert r1.state.com[0] < r0.state.com[0]      # 前漂更少

    def test_alpha_assists_fall(self):
        """同向 alpha 加速倾倒。"""
        st = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.30, -0.30],
                 rfh=0.25)
        r0 = st.drift(dt=0.2)
        r1 = st.drift(dt=0.2, alpha=(0, 2.0, 0))
        assert r1.state.com[2] < r0.state.com[2]

    def test_pivot_yaw_component_dropped(self):
        """踝无 yaw 矩通道：纯 z 分量 alpha ≡ 被动。"""
        st = _st([0.10, 0, 0.95], [-0.05, 0.09], [0.30, -0.30],
                 rfh=0.25)
        r0 = st.drift(dt=0.2)
        r1 = st.drift(dt=0.2, alpha=(0, 0, 30.0))
        np.testing.assert_allclose(r1.state.com, r0.state.com,
                                   atol=1e-12)

    def test_ballistic_alpha_spins_only(self):
        """腾空：绕 com 外力矩只改角动量，com 轨迹不变。"""
        J = np.diag([1.2, 1.5, 0.3])
        st = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09],
                 lfh=0.2, rfh=0.2)
        r0 = st.drift(dt=0.1, inertia_com=J)
        r1 = st.drift(dt=0.1, inertia_com=J, alpha=(0, 0, 2.0))
        np.testing.assert_allclose(r1.state.com, r0.state.com,
                                   atol=1e-12)
        np.testing.assert_allclose(
            r1.state.ang_mom, J @ np.array([0, 0, 2.0]) * 0.1,
            atol=1e-9)

    def test_hinge_takes_axis_component(self):
        """hinge 只吸收沿踝轴分量：双脚沿 y 排开时 e=±ŷ，
        alpha_y 改变演化、alpha_x 无效果。"""
        st = _st([0.10, 0, 0.95], [0, 0.10], [0, -0.10])
        r0 = st.drift(dt=0.2)
        ry = st.drift(dt=0.2, alpha=(0, 30.0, 0))
        rx = st.drift(dt=0.2, alpha=(30.0, 0, 0))
        assert ry.mode == "hinge"
        assert abs(ry.state.com[2] - r0.state.com[2]) > 1e-3
        np.testing.assert_allclose(rx.state.com, r0.state.com,
                                   atol=1e-12)


class TestCanBalance:
    """踝矩可达域判据：ξ 须落在可控 COP 界内。"""

    def test_airborne_false(self):
        st = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09],
                 lfh=0.2, rfh=0.2)
        assert not st.can_balance()

    def test_still_over_pivot(self):
        st = _st([0, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25)
        assert st.can_balance()

    def test_actuator_bound_not_sole(self):
        """com 前偏 0.08：在鞋底几何缘(0.167)内但超执行器界
        (τ_act/mg≈0.050) → False。"""
        st = _st([0.08, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25)
        assert not st.can_balance()
        st2 = _st([0.04, 0.1, 0.95], [0, 0.1], [0.3, -0.3],
                  rfh=0.25)
        assert st2.can_balance()

    def test_lateral_sole_bound(self):
        """侧向 N·margin_side < τ_act → 界 = 0.047 鞋底缘。"""
        assert not _st([0, 0.16, 0.95], [0, 0.1], [0.3, -0.3],
                       rfh=0.25).can_balance()
        assert _st([0, 0.14, 0.95], [0, 0.1], [0.3, -0.3],
                   rfh=0.25).can_balance()

    def test_moving_xi_out(self):
        """com 居中但速度把 ξ 顶出界：v=0.4 → ξ≈0.12 > 0.05。"""
        assert not _st([0, 0.1, 0.95], [0, 0.1], [0.3, -0.3],
                       rfh=0.25, v=(0.4, 0, 0)).can_balance()
        assert _st([0, 0.1, 0.95], [0, 0.1], [0.3, -0.3],
                   rfh=0.25, v=(0.15, 0, 0)).can_balance()

    def test_hinge_loadshare_stronger(self):
        """双脚载重分摊不占踝矩：com 前偏 0.15（单脚必死，
        双脚在鞋底缘 0.167 内）→ True。"""
        st = _st([0.15, 0, 0.95], [0, 0.1], [0, -0.1])
        assert st.can_balance()
        st2 = _st([0.2, 0, 0.95], [0, 0.1], [0, -0.1])
        assert not st2.can_balance()

    def test_hinge_along_axis_free(self):
        """com 沿踝轴偏出双脚端点 ≠ 失衡（锁死方向）——垂直
        分量在界内即 True。"""
        st = _st([0.0, 0.9, 0.95], [0, 0.1], [0, -0.1])
        assert st.can_balance()


class TestBalanceMargin:
    """balance_margin：can_balance 的连续化——符号一致 +
    数值语义（核内深度负、越界量正）。"""

    def test_consistent_with_can_balance(self):
        cases = [
            _st([0, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25),
            _st([0.08, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25),
            _st([0.04, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25),
            _st([0, 0.16, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25),
            _st([0.15, 0, 0.95], [0, 0.1], [0, -0.1]),
            _st([0.2, 0, 0.95], [0, 0.1], [0, -0.1]),
            _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09],
                lfh=0.2, rfh=0.2),
            _st([0, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25,
                v=(0.4, 0, 0))]
        for st in cases:
            assert (st.balance_margin() <= 0.0) == st.can_balance()

    def test_signed_semantics(self):
        """pivot 上静止 → 深度 ≈ −reach_lat（最紧轴是侧向 0.047）。"""
        st = _st([0, 0.1, 0.95], [0, 0.1], [0.3, -0.3], rfh=0.25)
        m = st.balance_margin()
        assert m == pytest.approx(-0.047, abs=1e-3)
        # com 前偏 0.08：前后向越界 0.08−0.05=0.03
        st2 = _st([0.08, 0.1, 0.95], [0, 0.1], [0.3, -0.3],
                  rfh=0.25)
        assert st2.balance_margin() == pytest.approx(0.03, abs=1e-3)

    def test_airborne_inf(self):
        st = _st([0, 0, 0.95], [-0.05, 0.09], [0.05, -0.09],
                 lfh=0.2, rfh=0.2)
        assert st.balance_margin() == float("inf")


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
