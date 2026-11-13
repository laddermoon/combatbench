"""Tests for robot_state.py — take_action 模拟器推演。"""
from __future__ import annotations

import numpy as np
import pytest
import mujoco

from point5.humanoid21.robot_state import RobotState


def _standing_state() -> RobotState:
    """标准站姿 -> RobotState（standard("standing") 同源的
    测试输入底本）。"""
    return RobotState.standing()


@pytest.fixture(scope="module")
def st():
    return _standing_state()


class TestStandardPoses:
    @pytest.mark.parametrize("name", RobotState.STANDARD_POSES)
    def test_constructable_zero_velocity(self, name):
        st = RobotState.standard(name)
        assert st.qpos.shape == (21,) and np.isfinite(st.qpos).all()
        assert np.isfinite(st.root_position).all()
        assert np.isfinite(st.root_orientation).all()
        for f in (st.qvel, st.root_linvel, st.root_angvel,
                  st.ang_mom):
            assert np.all(f == 0.0)

    def test_deterministic(self):
        a, b = (RobotState.standard("squat")
                for _ in range(2))
        assert np.array_equal(a.qpos, b.qpos)
        assert np.array_equal(a.root_position, b.root_position)
        assert np.array_equal(a.root_orientation, b.root_orientation)

    @pytest.mark.parametrize("name", RobotState.STANDARD_POSES)
    def test_named_wrapper_matches_registry(self, name):
        named = getattr(RobotState, name)()
        reg = RobotState.standard(name)
        assert np.array_equal(named.qpos, reg.qpos)
        assert np.array_equal(named.root_position, reg.root_position)

    def test_unknown_name(self):
        with pytest.raises(ValueError):
            RobotState.standard("bogus")

    def test_standing_matches_sim_reset(self):
        """standing 的 qpos/root 与 sim reset（robot_a）一致
        ——仅差双人占位的 x_offset。"""
        from envs.humanoid21.simulator import Humanoid21Simulator
        from envs.humanoid21.meta import Humanoid21Meta
        sim = Humanoid21Simulator()
        sim.reset()
        r = Humanoid21Meta.build_runtime_tables(
            sim.model)["robots"]["robot_a"]
        st = RobotState.standing()
        assert np.allclose(st.qpos,
                           sim.data.qpos[np.asarray(r["qpos_indices"])])
        adr = int(r["root_qpos_adr"])
        assert np.isclose(st.root_position[1:],
                          sim.data.qpos[adr + 1:adr + 3]).all()


class TestCorrespondingAction:
    def test_standing_matches_meta_action(self):
        """standing 的 action 表值 = rad_to_norm(joint_pos=0)。"""
        from envs.humanoid21.meta import Humanoid21Meta
        a = RobotState.standing().corresponding_action()
        np.testing.assert_allclose(
            a, Humanoid21Meta.INITIAL_POSES["standing"]["action"],
            atol=1e-3)

    def test_shape_bounds_and_clip(self):
        a = RobotState.squat().corresponding_action()
        assert a.shape == (21,) and np.isfinite(a).all()
        assert (np.abs(a) <= 1.0).all()
        # 越限 qpos → 目标钉在 ±1
        import dataclasses
        st = dataclasses.replace(
            RobotState.standing(),
            qpos=RobotState.standing().qpos + 100.0)
        assert np.all(st.corresponding_action() == 1.0)


class TestJointReachRange:
    def test_shape_lo_le_hi_within_limits(self, st):
        from point5.humanoid21.robot_state import joint_ranges
        rng = st.joint_reach_range()
        assert rng.shape == (21, 2)
        assert (rng[:, 0] <= rng[:, 1]).all()
        lim = joint_ranges()
        assert (rng[:, 0] >= lim[:, 0] - 1e-6).all()
        assert (rng[:, 1] <= lim[:, 1] + 1e-6).all()

    def test_stance_leg_tiny_range_gravity_asymmetry(self, st):
        """站姿：触地腿单步可达域远小于手臂；膝屈（−，重力
        折叠方向）比伸（+，对抗负载）容易。"""
        rng = st.joint_reach_range()
        knee, elbow = 6, 17
        assert rng[knee, 1] - rng[knee, 0] < 0.05
        assert rng[elbow, 1] - rng[elbow, 0] > 1.0
        assert st.qpos[knee] - rng[knee, 0] \
            > rng[knee, 1] - st.qpos[knee]

    def test_swing_leg_much_wider(self):
        """单腿站：摆动腿可达域 ≫ 支撑腿。"""
        st = RobotState.stand_on_left_leg()
        rng = st.joint_reach_range()
        swing_w = rng[7, 1] - rng[7, 0]      # ankle_y_right
        stance_w = rng[13, 1] - rng[13, 0]   # ankle_y_left
        assert swing_w > 10.0 * stance_w

    def test_clamp_pd_shrinks_or_equal(self, st):
        a, b = st.joint_reach_range(clamp_pd=True), \
            st.joint_reach_range(clamp_pd=False)
        assert (a[:, 0] >= b[:, 0] - 1e-9).all()
        assert (a[:, 1] <= b[:, 1] + 1e-9).all()

    def test_velocity_shifts_range(self, st):
        import dataclasses
        st2 = dataclasses.replace(
            st, qvel=st.qvel + np.eye(21)[6] * 5.0)
        rng = st2.joint_reach_range()
        assert rng[6].min() > -0.2     # 初速把可达区整体推向 +

    def test_dt_must_be_positive(self, st):
        with pytest.raises(ValueError):
            st.joint_reach_range(dt=0.0)


class TestRandom:
    def test_seed_deterministic(self):
        a, b = RobotState.random(7), RobotState.random(7)
        assert np.array_equal(a.qpos, b.qpos)
        assert not np.array_equal(a.qpos, RobotState.random(8).qpos)

    def test_zero_vel_default(self):
        st = RobotState.random(3)
        for f in (st.qvel, st.root_linvel, st.root_angvel,
                  st.ang_mom):
            assert np.all(f == 0.0)

    def test_vel_scale(self):
        st = RobotState.random(3, vel_scale=0.5)
        assert np.any(st.qvel != 0.0)
        assert np.any(st.ang_mom != 0.0)   # FK 真值回填非零


class TestTakeAction:
    def test_deterministic_and_new_object(self, st):
        a = np.zeros(21)
        s1 = st.take_action(a, dt=0.05)
        s2 = st.take_action(a, dt=0.05)
        assert s1 is not st                       # 新对象，非原地
        np.testing.assert_allclose(s1.qpos, s2.qpos, atol=1e-12)
        np.testing.assert_allclose(
            s1.root_position, s2.root_position, atol=1e-12)

    def test_input_state_untouched(self, st):
        q0, p0 = st.qpos.copy(), st.root_position.copy()
        st.take_action(np.ones(21) * 0.3, dt=0.05)
        np.testing.assert_allclose(st.qpos, q0, atol=0)
        np.testing.assert_allclose(st.root_position, p0, atol=0)

    def test_action_changes_outcome(self, st):
        s0 = st.take_action(np.zeros(21), dt=0.05)
        s1 = st.take_action(np.full(21, 0.5), dt=0.05)
        assert not np.allclose(s0.qpos, s1.qpos)
        assert np.isfinite(s1.qpos).all()
        assert np.isfinite(s1.ang_mom).all()

    def test_bad_dt_raises(self, st):
        with pytest.raises(ValueError):
            st.take_action(np.zeros(21), dt=0.0)

    def test_bad_action_shape_raises(self, st):
        with pytest.raises(ValueError):
            st.take_action(np.zeros(10))


class TestRender:
    def test_render_pil_and_save(self, st, tmp_path):
        img = st.render(out_path=tmp_path / "pose.png",
                        width=640, height=480)
        from PIL import Image
        assert isinstance(img, Image.Image)
        assert img.size == (640, 480)
        assert (tmp_path / "pose.png").exists()
        # 非纯黑/纯白——确实渲到了内容
        arr = np.asarray(img)
        assert arr.std() > 5.0

    def test_render_named_camera(self, st):
        img = st.render(camera="ceilingcamera",
                        width=320, height=240)
        assert img.size == (320, 240)


_ANKLE_J = frozenset(
    __import__("point5.humanoid21.mechanics", fromlist=["_JIDX"])
    ._JIDX[n] for n in ("ankle_y_right", "ankle_x_right",
                        "ankle_y_left", "ankle_x_left"))


class TestAnklePolicy:
    """双支撑踝策略——direction 期望重心水平加速度 (world xy)。"""

    def test_only_ankle_joints_move(self, st):
        """非踝关节 = 站姿基准 action（钉体态不棘轮）。"""
        a = st.ankle_policy(np.array([0.5, 0.3]))
        ref = st._base_action()
        assert a.shape == (21,)
        for i in range(21):
            if i not in _ANKLE_J:
                assert a[i] == ref[i]

    def test_direction_zero_compensates_gravity(self, st):
        """d=0 ≠ hold——站姿 com 在双踝中点前 ~2.6cm，零加速
        也须踝矩抗重力前倒：ankle_y 正偏（COP 前移到 com 下）。"""
        from point5.humanoid21.mechanics import _JIDX
        g = st.to_support_geometry()
        a_mid = 0.5 * (g.left_foot + g.right_foot)
        a = st.ankle_policy(np.zeros(2))
        ref = st._base_action()
        for n in ("ankle_y_right", "ankle_y_left"):
            d_y = a[_JIDX[n]] - ref[_JIDX[n]]
            assert d_y > 0.0                    # COP 向 com 侧压
            np.testing.assert_allclose(
                d_y, (g.com[0] - a_mid[0]) * (0.5 * 40.84 * 9.81)
                / 100.0 / 0.8727, rtol=0.05)

    def test_sagittal_direction_sign(self, st):
        """+x 加速需求 -> COP 后置 -> ankle_y 负偏（两角同值，
        ankle_x 仅随侧向需求动）。"""
        from point5.humanoid21.mechanics import _JIDX
        a = st.ankle_policy(np.array([0.5, 0.0]))
        ref = st._base_action()
        for n in ("ankle_y_right", "ankle_y_left"):
            assert a[_JIDX[n]] < ref[_JIDX[n]]
        np.testing.assert_allclose(a[_JIDX["ankle_y_right"]],
                                   a[_JIDX["ankle_y_left"]])
        np.testing.assert_allclose(a[_JIDX["ankle_x_right"]],
                                   ref[_JIDX["ankle_x_right"]],
                                   atol=1e-12)

    def test_lateral_antisymmetric(self, st):
        """+y 加速需求 -> 左右 ankle_x 反号偏（斜轴 kx 镜像：
        COP 移 −y -> τ_wx>0 -> 右脚正、左脚负）。"""
        from point5.humanoid21.mechanics import _JIDX
        a = st.ankle_policy(np.array([0.0, 0.3]))
        ref = st._base_action()
        dl = a[_JIDX["ankle_x_left"]] - ref[_JIDX["ankle_x_left"]]
        dr = a[_JIDX["ankle_x_right"]] - ref[_JIDX["ankle_x_right"]]
        assert dl < 0 and dr > 0
        np.testing.assert_allclose(dl, -dr)

    def test_saturation(self, st):
        """超大 direction -> 踝目标被能力边界 clamp，仍在 [-1,1]。"""
        a = st.ankle_policy(np.array([1e3, -1e3]))
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)

    def test_sim_moves_com_forward(self, st):
        """sim 真值：+x direction 下 COM 比基准站姿更向前。"""
        ref = st._base_action()
        a = st.ankle_policy(np.array([0.5, 0.0]))
        com_hold = st.take_action(ref, dt=0.1) \
            .to_support_geometry().com[0]
        com_pol = st.take_action(a, dt=0.1) \
            .to_support_geometry().com[0]
        assert com_pol > com_hold

    def test_requires_double_support(self, st, monkeypatch):
        """非双支撑 -> NotImplementedError（各形态各策略族）。"""
        from point5.humanoid21.support_state import SupportGeometry
        g = st.to_support_geometry()
        single = SupportGeometry(
            com=g.com, left_foot=g.left_foot,
            right_foot=g.right_foot, torso_yaw=g.torso_yaw,
            left_foot_height=g.left_foot_height,
            right_foot_height=0.5)
        monkeypatch.setattr(st, "to_support_geometry",
                            lambda: single)
        with pytest.raises(NotImplementedError):
            st.ankle_policy(np.array([0.1, 0.0]))


class TestActionForBalanceWithAnklePolicy:
    def test_standing_nearly_base(self, st):
        """站姿 -> 非踝 = 站姿基准；踝仅受 com 偏心补偿小偏。"""
        a = st.action_for_balance_with_ankle_policy()
        ref = st._base_action()
        for i in range(21):
            if i not in _ANKLE_J:
                assert a[i] == ref[i]
        assert np.abs(a[list(_ANKLE_J)] - ref[list(_ANKLE_J)]
                      ).max() < 0.2

    def test_perturbed_ankle_pushes_cop_forward(self, st):
        """前冲态 -> ankle_y 正偏（COP 前顶刹重心）。"""
        import dataclasses
        from point5.humanoid21.mechanics import _JIDX
        pert = dataclasses.replace(
            st, root_linvel=st.root_linvel + np.array([0.5, 0, 0]))
        a = pert.action_for_balance_with_ankle_policy()
        ref = pert._base_action()
        for n in ("ankle_y_right", "ankle_y_left"):
            assert a[_JIDX[n]] > ref[_JIDX[n]]

    def test_closed_loop_stabilizes(self, st):
        """sim 真值：前冲 0.15m/s 闭环 20 步——速度收敛、
        margin 保持核内（阻尼律实测工况）。"""
        import dataclasses
        rs = dataclasses.replace(
            st, root_linvel=st.root_linvel + np.array([0.15, 0, 0]))
        for _ in range(20):
            a = rs.action_for_balance_with_ankle_policy()
            rs = rs.take_action(a)
        g = rs.to_support_state()
        assert abs(g.com_vel[0]) < 0.1
        assert g.balance_margin() < 0.0


class TestArmWheelPolicy:
    """arm_as_reaction_wheel_policy（有界反应轮）原语不变量。

    能力边界已 oracle 证伪（sim 见 robot_state 该方法 docstring），
    这里只固化接口契约。
    """

    @pytest.fixture(scope="class")
    def st(self):
        return RobotState.standing()

    def test_non_arm_equals_standing(self, st):
        """非臂 15 维与站姿快照逐位相等（含踝钉住）。"""
        from point5.humanoid21.robot_state import _ARM_J
        for d in (-10.0, -2.0, 0.0, +2.0, +10.0):
            a = st.arm_as_reaction_wheel_policy(d)
            b = st._base_action()
            for i in range(21):
                if i not in _ARM_J:
                    assert a[i] == b[i]
            assert a.shape == (21,)

    def test_direction_sign_maps_to_swing(self, st):
        """direction<0（刹前倒）→ 手臂向前加速抽走前倒角
        动量 ⇒ 摆角目标 θ_cmd > θ_now；d>0 反向；d=0 →
        全等于站姿。"""
        import dataclasses
        from point5.humanoid21.mechanics import (
            _HALF, _MID, Skeleton)
        from point5.humanoid21.robot_state import _ARM_J
        sk = Skeleton()
        e3 = np.array([0.0, 1.0, 0.0])
        ep3 = np.array([1.0, 0.0, 0.0])
        th0 = st._swing_theta_now(e3, ep3, sk)
        for d, want in ((-10.0, +1), (+10.0, -1)):
            a = st.arm_as_reaction_wheel_policy(d)
            q_arm = st.qpos.copy()
            for i in _ARM_J:
                q_arm[i] = a[i] * _HALF[i] + _MID[i]
            st_tgt = dataclasses.replace(st, qpos=q_arm)
            th_cmd = st_tgt._swing_theta_now(e3, ep3, sk)
            assert np.sign(th_cmd - th0) == want
        a0 = st.arm_as_reaction_wheel_policy(0.0)
        base0 = st._base_action()
        non_arm = [i for i in range(21) if i not in _ARM_J]
        assert np.allclose(a0[non_arm], base0[non_arm])
        # 直臂模式：零指令臂钉在"肘全伸+θ_now"而非站姿弯肘
        q0 = st.qpos.copy()
        for i in _ARM_J:
            q0[i] = a0[i] * _HALF[i] + _MID[i]
        assert np.allclose(q0[[17, 20]], -1.7453, atol=0.15)

    def test_single_support_raises(self, st):
        """非双脚支撑 -> NotImplementedError。"""
        import dataclasses
        g = st.to_support_state()
        st1 = dataclasses.replace(
            st, root_position=st.root_position + np.array([0, 0, 0.5]))
        if st1.to_support_state().support_class().name == "DOUBLE":
            pytest.skip("抬高后仍判双脚——换一个构造")
        with pytest.raises(NotImplementedError):
            st1.arm_as_reaction_wheel_policy(1.0)

    def test_wheel_closed_loop_2s(self, st):
        """sim 真值：比例抡摆闭环 40 步（2s）——保持双脚支
        撑、com 不漂出 ±0.1m（5s 完整工况实测站住，回归
        用半程防超时）。standing 走保肘档（straight=False：
        直臂前悬的静态前偏在站姿上反伤稳定，实测 com 漂
        出 0.23m）；ready 走直臂档（实证 6s 站住）。"""
        rs = st
        for _ in range(40):
            a = rs.action_for_balance_with_wheel_policy(
                straight=False)
            rs = rs.take_action(a)
            g = rs.to_support_state()
            assert g.support_class().name == "DOUBLE"
        assert abs(g.com[0]) < 0.1 and abs(g.com_vel[0]) < 0.3

    def test_wheel_closed_loop_ready(self):
        """ready + 直臂抡摆闭环（默认 _WHEEL_STRAIGHT）——
        钉 ready 基准 40 步保双脚支撑、com 收在踝心附近。"""
        rs = RobotState.ready()
        base = rs.corresponding_action()
        for _ in range(40):
            a = rs.action_for_balance_with_wheel_policy(base=base)
            rs = rs.take_action(a)
            g = rs.to_support_state()
            assert g.support_class().name == "DOUBLE"
        g = rs.to_support_state()
        am = 0.5 * (np.asarray(g.left_foot) + np.asarray(g.right_foot))
        assert abs(g.com[0] - am[0]) < 0.1
