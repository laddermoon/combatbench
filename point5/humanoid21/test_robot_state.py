"""Tests for robot_state.py — take_action 模拟器推演。"""
from __future__ import annotations

import numpy as np
import pytest
import mujoco

from point5.humanoid21.robot_state import RobotState


def _standing_state() -> RobotState:
    """真实 sim reset 态 -> RobotState（测试输入底本）。"""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from envs.humanoid21.meta import Humanoid21Meta
    sim = Humanoid21Simulator()
    sim.reset()
    r = Humanoid21Meta.build_runtime_tables(
        sim.model)["robots"]["robot_a"]
    d = sim.data
    adr, vadr = int(r["root_qpos_adr"]), int(r["root_qvel_adr"])
    quat = np.asarray(d.qpos[adr + 3:adr + 7])
    Rm = np.empty(9)
    mujoco.mju_quat2Mat(Rm, quat)
    Rm = Rm.reshape(3, 3)
    zyx = np.array([
        np.arctan2(-Rm[2, 0], np.hypot(Rm[0, 0], Rm[1, 0])),
        np.arctan2(Rm[2, 1], Rm[2, 2]),
        np.arctan2(Rm[1, 0], Rm[0, 0])])
    mujoco.mj_subtreeVel(sim.model, d)
    return RobotState(
        qpos=d.qpos[np.asarray(r["qpos_indices"])].copy(),
        root_position=d.qpos[adr:adr + 3].copy(),
        root_orientation=zyx,
        qvel=d.qvel[np.asarray(r["qvel_indices"])].copy(),
        root_linvel=d.qvel[vadr:vadr + 3].copy(),
        root_angvel=Rm @ d.qvel[vadr + 3:vadr + 6],
        ang_mom=np.asarray(
            d.subtree_angmom[r["root_body_id"]]).copy())


@pytest.fixture(scope="module")
def st():
    return _standing_state()


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
