"""Tests for point5.humanoid21.pose.Pose — 精确无位置姿态表达.

Ground truth comes from a real Humanoid21Simulator: obs decode paths
are checked against subtree_com / foot body xpos where possible.
"""
import numpy as np
import pytest

from point5.contracts import P5
from point5.humanoid21.pose import Pose, _level_rot


@pytest.fixture(scope="module")
def sim():
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator(initial_distance=6.0, initial_pose_a="standing")
    s.reset(0)
    return s


@pytest.fixture(scope="module")
def obs(sim):
    return sim.get_derived_state()["robot_a"]["observation"]


@pytest.fixture(scope="module")
def pose(obs):
    return Pose.from_obs(obs)


# ---------------------------------------------------------------------------
# construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid(self):
        p = Pose(qpos=np.zeros(21), root_z=1.28,
                 g_body=np.array([0., 0., -9.81]))
        assert p.qpos.shape == (21,) and p.g_body.shape == (3,)
        assert p.root_z == 1.28

    @pytest.mark.parametrize("bad", [np.zeros(20), np.zeros(22), np.zeros(7)])
    def test_bad_qpos(self, bad):
        with pytest.raises(AssertionError):
            Pose(qpos=bad, root_z=1.0, g_body=np.zeros(3))

    def test_bad_g_body(self):
        with pytest.raises(AssertionError):
            Pose(qpos=np.zeros(21), root_z=1.0, g_body=np.zeros(4))


# ---------------------------------------------------------------------------
# from_obs
# ---------------------------------------------------------------------------

class TestFromObs:
    def test_qpos_denormalized(self, obs, pose):
        rng = Pose.fk().joint_ranges("robot_a")
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        np.testing.assert_allclose(pose.qpos, obs[0:21] * sc + ref)

    def test_root_z_and_g(self, obs, pose):
        assert pose.root_z == pytest.approx(float(obs[45]))
        np.testing.assert_allclose(pose.g_body, obs[42:45])

    def test_standing_upright(self, pose):
        np.testing.assert_allclose(pose.tilt_angles(), [0., 0.],
                                   atol=0.05)


# ---------------------------------------------------------------------------
# tilt_angles
# ---------------------------------------------------------------------------

class TestTilt:
    def test_upright_zero(self):
        p = Pose(np.zeros(21), 1.0, np.array([0., 0., -9.81]))
        np.testing.assert_allclose(p.tilt_angles(), [0., 0.], atol=1e-9)

    def test_rolled(self):
        # gravity tilted +x in body frame -> positive pitch
        p = Pose(np.zeros(21), 1.0, np.array([3., 0., -9.5]))
        r, pi = p.tilt_angles()
        assert pi > 0.1 and abs(r) < 0.05


# ---------------------------------------------------------------------------
# anchor
# ---------------------------------------------------------------------------

class TestAnchor:
    def test_ankle_mid_origin(self, pose):
        a = pose.anchor("ankle_mid")
        fp = a.foot_pos()
        mid = 0.5 * (fp["right"] + fp["left"])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)

    def test_foot_anchor_lands(self, pose):
        for side in ("left", "right"):
            a = pose.anchor(side, xy=(1.0, 2.0))
            np.testing.assert_allclose(a.foot_pos()[side], [1.0, 2.0],
                                       atol=1e-9)

    def test_com_anchor_lands(self, pose):
        a = pose.anchor("com", xy=(0.5, -0.5))
        np.testing.assert_allclose(a.com()[:2], [0.5, -0.5], atol=1e-9)

    def test_yaw_controls_heading(self, pose):
        # anchoring at yaw=pi/2 must make the torso face +y:
        # hands/feet layout rotates so the ankle-line normal -> +y
        a0 = pose.anchor("ankle_mid", yaw=0.0)
        a90 = pose.anchor("ankle_mid", yaw=np.pi / 2)
        fp0 = a90.foot_pos()["right"] - a90.foot_pos()["left"]
        # in a 90deg-rotated frame the right-left axis points -x->+x
        np.testing.assert_allclose(fp0 / np.linalg.norm(fp0),
                                   [1.0, 0.0], atol=1e-9)
        assert a0.R.shape == (3, 3)

    def test_R_orthonormal(self, pose):
        a = pose.anchor("ankle_mid", yaw=0.7)
        np.testing.assert_allclose(a.R @ a.R.T, np.eye(3), atol=1e-9)

    def test_unknown_point(self, pose):
        with pytest.raises(ValueError):
            pose.anchor("knee")

    def test_p5_returns_p5_type(self, pose):
        a = pose.anchor()
        assert isinstance(a.p5(), P5)
        assert a.p5().pos.shape == (5, 3)


# ---------------------------------------------------------------------------
# p5() / support_landmark / com_height
# ---------------------------------------------------------------------------

class TestOutputs:
    def test_p5_equals_default_anchor(self, pose):
        np.testing.assert_allclose(pose.p5().pos,
                                   pose.anchor("ankle_mid").p5().pos)

    def test_p5_local_frame_conventions(self, pose):
        p = pose.p5().pos
        mid = 0.5 * (p[3, :2] + p[4, :2])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)

    def test_com_height_matches_sim(self, sim, pose):
        true_z = float(sim.data.subtree_com[1][2])
        assert pose.com_height() == pytest.approx(true_z, abs=0.01)

    def test_support_landmark_standing(self, pose):
        lm = pose.support_landmark_on_p5_axis()
        assert lm.feet["right"].force_ratio == pytest.approx(0.5)
        assert lm.feet["left"].force_ratio == pytest.approx(0.5)
        assert lm.com[2] == pytest.approx(0.85, abs=0.02)
        for s in ("right", "left"):
            assert lm.feet[s].pos.shape == (2,)

    def test_support_landmark_single_leg(self):
        from envs.humanoid21.simulator import Humanoid21Simulator
        s = Humanoid21Simulator(initial_distance=6.0,
                                initial_pose_a="stand_on_left_leg")
        s.reset(0)
        obs = s.get_derived_state()["robot_a"]["observation"]
        lm = Pose.from_obs(obs).support_landmark_on_p5_axis()
        assert lm.feet["right"].force_ratio == 0.0     # airborne
        assert lm.feet["left"].force_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# from_p5 (solve_equilibrium inverse)
# ---------------------------------------------------------------------------

class TestFromP5:
    def test_roundtrip_pose(self, pose):
        p2 = Pose.from_p5(pose.p5())
        assert p2.com_height() == pytest.approx(pose.com_height(),
                                                abs=0.05)

    def test_roundtrip_points(self, pose):
        p2 = Pose.from_p5(pose.p5())
        d = np.linalg.norm(p2.p5().pos - pose.p5().pos, axis=1)
        assert d.max() < 0.15          # solver is approximate

    def test_accepts_raw_array(self, pose):
        p2 = Pose.from_p5(pose.p5().pos)
        assert p2.com_height() > 0.5


# ---------------------------------------------------------------------------
# _level_rot helper
# ---------------------------------------------------------------------------

class TestLevelRot:
    def test_already_level(self):
        np.testing.assert_allclose(
            _level_rot(np.array([0., 0., -9.81])), np.eye(3), atol=1e-9)

    def test_levels_tilted_gravity(self):
        g = np.array([3., 0., -9.5])
        np.testing.assert_allclose(
            _level_rot(g) @ (g / np.linalg.norm(g)),
            [0., 0., -1.], atol=1e-9)

    def test_output_orthonormal(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            g = rng.normal(0, 5, 3); g[2] = -abs(g[2]) - 1
            R = _level_rot(g)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
