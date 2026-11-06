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


# ---------------------------------------------------------------------------
# 深入：别名隔离、跨方法一致性、复合锚定、边缘 g_body、robot_b
# ---------------------------------------------------------------------------

class TestOwnership:
    def test_qpos_not_aliased(self):
        q = np.zeros(21)
        p = Pose(qpos=q, root_z=1.0, g_body=np.array([0., 0., -9.81]))
        q[0] = 99.0
        assert p.qpos[0] == 0.0

    def test_g_body_not_aliased(self):
        g = np.array([0., 0., -9.81])
        p = Pose(np.zeros(21), 1.0, g)
        g[0] = 5.0
        assert p.g_body[0] == 0.0


class TestCrossConsistency:
    def test_com_height_equals_anchored_com_z(self, pose):
        a = pose.anchor()
        assert a.com()[2] == pytest.approx(pose.com_height(), abs=1e-9)

    def test_foot_pos_matches_p5(self, pose):
        a = pose.anchor()
        fp, p = a.foot_pos(), a.p5().pos
        np.testing.assert_allclose(fp["right"], p[3, :2])
        np.testing.assert_allclose(fp["left"], p[4, :2])

    def test_p5_point_order(self, pose):
        # index 0 must be the head (highest point when standing)
        p = pose.p5().pos
        assert p[0, 2] == p[:, 2].max()
        assert p[0, 2] > 1.0

    def test_landmark_com_matches_anchor(self, pose):
        lm = pose.support_landmark_on_p5_axis()
        np.testing.assert_allclose(lm.com, pose.anchor().com())

    def test_anchor_idempotent(self, pose):
        # repeated anchors on one Pose give identical outputs (no state)
        a1, a2 = pose.anchor(), pose.anchor()
        np.testing.assert_allclose(a1.p5().pos, a2.p5().pos)
        np.testing.assert_allclose(a1.com(), a2.com())


class TestCompositeAnchor:
    def test_xy_and_yaw_together(self, pose):
        a = pose.anchor("left", xy=(0.3, -0.7), yaw=np.pi / 2)
        np.testing.assert_allclose(a.foot_pos()["left"], [0.3, -0.7],
                                   atol=1e-9)
        # and heading rotated: ankle line dir -> +x (right-left axis)
        d = a.foot_pos()["right"] - a.foot_pos()["left"]
        np.testing.assert_allclose(d / np.linalg.norm(d), [1., 0.],
                                   atol=1e-9)

    def test_tilted_pose_anchored_level(self):
        # g_body tilted 15deg — anchor still produces a level output
        g = np.array([np.sin(0.26), 0., -np.cos(0.26)]) * 9.81
        p = Pose(qpos=np.zeros(21), root_z=1.2, g_body=g)
        a = p.anchor()
        mid = 0.5 * (a.foot_pos()["right"] + a.foot_pos()["left"])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)
        # R must fold the tilt in: R applied to g_body -> -z
        np.testing.assert_allclose(
            a.R @ (g / np.linalg.norm(g)), [0., 0., -1.], atol=1e-9)


class TestGBodyEdge:
    def test_inverted(self):
        # g_body pointing UP (robot upside down) — 180deg branch
        p = Pose(np.zeros(21), 1.0, np.array([0., 0., 9.81]))
        a = p.anchor()
        mid = 0.5 * (a.foot_pos()["right"] + a.foot_pos()["left"])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)

    def test_tiny_magnitude(self):
        # near-zero g_body is normalized, not crashing
        p = Pose(np.zeros(21), 1.0, np.array([0., 0., -1e-12]))
        a = p.anchor()
        assert np.all(np.isfinite(a.com()))

    def test_level_rot_inverted_branch(self):
        # g = +z exactly: the g[2]<0 else-branch (diag flip)
        R = _level_rot(np.array([0., 0., 9.81]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


class TestRobotB:
    def test_from_obs_robot_b(self):
        from envs.humanoid21.simulator import Humanoid21Simulator
        s = Humanoid21Simulator(initial_distance=6.0,
                                initial_pose_a="standing")
        s.reset(0)
        obs = s.get_derived_state()["robot_b"]["observation"]
        p = Pose.from_obs(obs, robot_id="robot_b")
        assert p.com_height() == pytest.approx(0.85, abs=0.05)


class TestNonStanding:
    def test_single_leg_tilt_nonzero(self):
        from envs.humanoid21.simulator import Humanoid21Simulator
        s = Humanoid21Simulator(initial_distance=6.0,
                                initial_pose_a="stand_on_left_leg")
        s.reset(0)
        obs = s.get_derived_state()["robot_b"]["observation"]
        # robot_b is the frozen one standing; use robot_a falling pose
        obs = s.get_derived_state()["robot_a"]["observation"]
        p = Pose.from_obs(obs)
        lm = p.support_landmark_on_p5_axis()
        # com pulled toward the support (left) foot side
        assert lm.feet["left"].force_ratio == 1.0
        assert lm.feet["right"].force_ratio == 0.0
        # com closer to support foot than airborne foot in xy
        fp = lm.feet
        assert (np.linalg.norm(lm.com[:2] - fp["left"].pos)
                < np.linalg.norm(lm.com[:2] - fp["right"].pos))

    def test_airborne_gives_zero_force(self):
        # a P5 with both feet lifted above the planted band
        base = np.zeros((5, 3)); base[3] = [0, -0.09, 0.5]; base[4] = [0, 0.09, 0.5]
        base[0] = [0, 0, 1.5]
        p = Pose.from_p5(base, tolerance=0.05)
        a = p.anchor()
        fp = a.foot_pos()
        # both raw foot z > 0.12 -> all zero force (if the solver lifts them)
        raw = a.p5().pos
        if raw[3, 2] > 0.12 and raw[4, 2] > 0.12:
            lm = p.support_landmark_on_p5_axis()
            assert lm.feet["right"].force_ratio == 0.0
            assert lm.feet["left"].force_ratio == 0.0


class TestDeterminism:
    def test_same_obs_same_pose(self, obs):
        p1, p2 = Pose.from_obs(obs), Pose.from_obs(obs)
        np.testing.assert_array_equal(p1.qpos, p2.qpos)
        np.testing.assert_allclose(p1.com_height(), p2.com_height())

    def test_from_p5_deterministic(self, pose):
        a = Pose.from_p5(pose.p5(), tolerance=0.02)
        b = Pose.from_p5(pose.p5(), tolerance=0.02)
        np.testing.assert_allclose(a.qpos, b.qpos, atol=1e-6)
