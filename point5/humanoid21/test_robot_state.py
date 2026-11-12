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


class TestActionForBalance:
    def test_standing_returns_near_hold(self, st):
        """站姿已可平衡 → 最优解 ≈ a_hold（不动先验）。"""
        from point5.humanoid21.mechanics import rad_to_norm
        a = st.action_for_balance(dt=0.05, max_evals=30)
        a_hold = np.clip(rad_to_norm(st.qpos), -1, 1)
        assert a.shape == (21,)
        assert np.all(np.abs(a) <= 1.0)
        np.testing.assert_allclose(a, a_hold, atol=0.15)

    def test_perturbed_improves_margin(self, st):
        """核外扰动态（前冲 1 m/s）→ 软罚 best-effort：
        margin 显著改善且 action 明确偏离 hold。"""
        import dataclasses
        pert = dataclasses.replace(
            st, root_linvel=st.root_linvel + np.array([1.0, 0, 0]))
        m0 = pert.to_support_state().balance_margin()
        assert m0 > 0.0                            # 确实在核外
        a = pert.action_for_balance(dt=0.05, max_evals=80)
        m_opt = pert.take_action(a, dt=0.05) \
            .to_support_state().balance_margin()
        assert m_opt < m0 - 0.03                   # 显著改善
        from point5.humanoid21.mechanics import rad_to_norm
        a_hold = np.clip(rad_to_norm(pert.qpos), -1, 1)
        assert np.abs(a - a_hold).max() > 0.05     # 真动了
