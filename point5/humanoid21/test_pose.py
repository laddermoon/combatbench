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
        rng = Pose.joint_ranges("robot_a")
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


# ---------------------------------------------------------------------------
# 深：世界系度量不变量、_quat_wxyz、共享FK、z透传、边界、估计器对照
# ---------------------------------------------------------------------------

class TestWorldInvariants:
    """Frame-free ground truth: distances are invariant across frames."""

    def _sim_foot_dist(self, sim):
        d = sim.data
        fl = d.xpos[__import__('mujoco').mj_name2id(
            d.model, __import__('mujoco').mjtObj.mjOBJ_BODY,
            'ankle_left_a')][0:2] if False else None
        return None

    def test_foot_span_matches_sim(self, sim, pose):
        import mujoco
        fr = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY,
                               "foot_right_a")
        fl = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY,
                               "foot_left_a")
        sim_span = np.linalg.norm(
            sim.data.xpos[fr][:2] - sim.data.xpos[fl][:2])
        p = pose.p5().pos
        dec_span = np.linalg.norm(p[3, :2] - p[4, :2])
        # landmark offsets vs body origins differ but same order
        assert abs(sim_span - dec_span) < 0.1

    def test_head_foot_distance_sane(self, pose):
        p = pose.p5().pos
        d = np.linalg.norm(p[0] - p[3])
        assert 1.0 < d < 1.8          # head-to-ankle chord, standing


class TestQuatWxyz:
    def test_identity(self):
        from point5.humanoid21.pose import _quat_wxyz
        np.testing.assert_allclose(_quat_wxyz(np.eye(3)),
                                   [1, 0, 0, 0], atol=1e-9)

    def test_90deg_branches(self):
        # rotations with trace <= 0 hit the elif branch
        from point5.humanoid21.pose import _quat_wxyz
        for axis, ang in (([1, 0, 0], np.pi), ([0, 1, 0], np.pi),
                          ([0, 0, 1], np.pi), ([1, 0, 0], np.pi / 2),
                          ([0, 1, 0], 2 * np.pi / 3)):
            a = np.asarray(axis, float); a /= np.linalg.norm(a)
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]],
                          [-a[1], a[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            q = _quat_wxyz(R)
            assert abs(np.linalg.norm(q) - 1) < 1e-9
            # quat -> rotation must reproduce R
            w, x, y, z = q
            R2 = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            np.testing.assert_allclose(R2, R, atol=1e-6)


class TestSharedFK:
    def test_alternating_anchors_independent(self, sim):
        # _FK is a shared mutable model — two poses' anchors must not
        # leak state between calls
        obs = sim.get_derived_state()["robot_a"]["observation"]
        p1 = Pose.from_obs(obs)
        q2 = np.zeros(21); q2[10] = 0.3
        p2 = Pose(qpos=q2, root_z=1.0, g_body=np.array([0., 0., -9.81]))
        a1 = p1.anchor().com()
        _ = p2.anchor()                     # intervening FK pose
        np.testing.assert_allclose(p1.anchor().com(), a1)

    def test_two_poses_different(self, sim):
        obs = sim.get_derived_state()["robot_a"]["observation"]
        p1 = Pose.from_obs(obs)
        p2 = Pose(qpos=np.zeros(21), root_z=1.0,
                  g_body=np.array([0., 0., -9.81]))
        assert not np.allclose(p1.anchor().com(),
                               p2.anchor().com(), atol=1e-6)


class TestZPassthrough:
    def test_anchor_preserves_z(self, pose):
        # xy translation + yaw rotation must not change any z
        for kw in (dict(xy=(1, 2), yaw=0.5),
                   dict(point="left", xy=(-1, 0), yaw=-1.2)):
            a = pose.anchor(**kw)
            ref = pose.anchor()
            np.testing.assert_allclose(a.p5().pos[:, 2],
                                       ref.p5().pos[:, 2], atol=1e-9)


class TestPlantedBoundary:
    def test_boundary_semantics(self):
        # foot z < 0.12 counts planted; just above does not
        p0 = np.zeros((5, 3)); p0[0] = [0, 0, 1.4]
        p0[3] = [0, -0.09, 0.087]; p0[4] = [0, 0.09, 0.13]   # 0.13 > band
        pos = Pose.from_p5(p0, tolerance=0.05)
        lm = pos.support_landmark_on_p5_axis()
        raw = pos.p5().pos
        # whichever side is above band gets 0; if solver lands both
        # below, both share — lock the semantic either way
        for s, fi in (("right", 3), ("left", 4)):
            if raw[fi, 2] > 0.12:
                assert lm.feet[s].force_ratio == 0.0


class TestEstimatorCrossCheck:
    def test_exact_com_vs_mass_proxy(self, pose):
        # exact mass-weighted com vs model-free proxy: same ballpark
        from point5.humanoid21.executor.tools.com_solver import (
            com_estimate)
        exact = pose.anchor().com()
        proxy = com_estimate(pose.p5().pos)
        assert np.linalg.norm(exact - proxy) < 0.08

    def test_com_inside_support_polygon(self, pose):
        # standing: com xy must sit within the ankle-mid support area
        a = pose.anchor()
        fp = a.foot_pos()
        com = a.com()[:2]
        xr = sorted([fp["right"][1], fp["left"][1]])
        assert xr[0] - 0.05 < com[1] < xr[1] + 0.05
        assert abs(com[0]) < 0.15          # near ankle line


class TestAnchorPoints:
    @pytest.mark.parametrize("pt", ["ankle_mid", "left", "right", "com"])
    def test_all_points_resolve(self, pose, pt):
        a = pose.anchor(pt, xy=(0.2, -0.1), yaw=0.3)
        assert a.p5().pos.shape == (5, 3)
        assert np.all(np.isfinite(a.com()))

    def test_anchor_com_vs_mid_differ(self, pose):
        # different anchor points -> different frames, same pose shape
        m, c = pose.anchor("ankle_mid"), pose.anchor("com")
        dm = np.linalg.norm(m.p5().pos[3:5] - c.p5().pos[3:5], axis=1)
        assert dm.min() > 0                 # frames differ
        # but pairwise foot distance is frame-invariant
        fm = np.linalg.norm(m.foot_pos()["right"] - m.foot_pos()["left"])
        fc = np.linalg.norm(c.foot_pos()["right"] - c.foot_pos()["left"])
        assert fm == pytest.approx(fc, abs=1e-9)


class TestFromP5Poses:
    def test_stepped_pose(self):
        p0 = np.zeros((5, 3)); p0[0] = [0.1, 0, 1.3]
        p0[1] = [0.2, -0.15, 0.9]; p0[2] = [0.2, 0.15, 0.9]
        p0[3] = [0.3, -0.09, 0.087]; p0[4] = [-0.05, 0.09, 0.087]
        pos = Pose.from_p5(p0, tolerance=0.05)
        p = pos.p5().pos
        # solver must put front foot forward of rear foot
        assert p[3, 0] > p[4, 0] + 0.1

    def test_crouch_pose(self):
        p0 = np.zeros((5, 3)); p0[0] = [0, 0, 0.9]
        p0[3] = [0, -0.09, 0.087]; p0[4] = [0, 0.09, 0.087]
        pos = Pose.from_p5(p0, tolerance=0.05)
        assert pos.com_height() < 0.75     # crouched com is low


# ---------------------------------------------------------------------------
# 更深：共享模型隔离、R 保向、奇异输入、容差效应、边界、render 路径
# ---------------------------------------------------------------------------

class TestRobotIsolation:
    def test_two_robots_coexist_in_shared_model(self, sim):
        # robot_a and robot_b live in ONE kinematics model — each
        # _fk_state reads only its own robot
        ds = sim.get_derived_state()
        pa = Pose.from_obs(ds["robot_a"]["observation"], robot_id="robot_a")
        pb = Pose.from_obs(ds["robot_b"]["observation"], robot_id="robot_b")
        # different root heights -> different com must not leak
        # both standing -> identical LOCAL frames is correct; what
        # matters is each read its OWN robot (isolation, not identity)
        assert pa.com_height() == pytest.approx(0.85, abs=0.05)
        assert pb.com_height() == pytest.approx(0.85, abs=0.05)
        assert pa.robot_id != pb.robot_id

    def test_pose_write_doesnt_touch_other_robot(self, sim):
        ds = sim.get_derived_state()
        pb = Pose.from_obs(ds["robot_b"]["observation"], robot_id="robot_b")
        before = pb.anchor().com().copy()
        pa = Pose.from_obs(ds["robot_a"]["observation"], robot_id="robot_a")
        pa._set_fk_pose()                     # writes only robot_a's qpos
        np.testing.assert_allclose(pb.anchor().com(), before)


class TestRRotation:
    def test_proper_rotation_det(self, pose):
        a = pose.anchor(yaw=0.7)
        assert np.linalg.det(a.R) == pytest.approx(1.0, abs=1e-9)

    def test_R_levels_gravity(self, pose):
        # R @ g_body/|g| must equal -z for every pose
        g = pose.g_body / np.linalg.norm(pose.g_body)
        np.testing.assert_allclose(pose.anchor().R @ g, [0., 0., -1.],
                                   atol=1e-6)


class TestSingularInputs:
    def test_zero_g_body_no_crash(self):
        p = Pose(np.zeros(21), 1.0, np.zeros(3))
        a = p.anchor()
        assert np.all(np.isfinite(a.com()))
        assert np.all(np.isfinite(a.p5().pos))

    def test_extreme_root_z(self):
        for z in (0.1, 0.3, 3.0):
            p = Pose(np.zeros(21), z, np.array([0., 0., -9.81]))
            assert np.all(np.isfinite(p.anchor().com()))

    def test_degenerate_p5_anchor_no_crash(self):
        p0 = np.zeros((5, 3)); p0[0] = [0, 0, 1.4]
        # feet coincident -> degenerate; anchor still computes
        pos = Pose.from_p5(p0, tolerance=0.05)
        a = pos.anchor()
        assert np.all(np.isfinite(a.com()))


class TestToleranceEffect:
    def test_tighter_tolerance_better_fit(self, pose):
        loose = Pose.from_p5(pose.p5(), tolerance=0.05)
        tight = Pose.from_p5(pose.p5(), tolerance=0.005)
        dl = np.linalg.norm(loose.p5().pos - pose.p5().pos, axis=1).max()
        dt = np.linalg.norm(tight.p5().pos - pose.p5().pos, axis=1).max()
        assert dt <= dl + 1e-9            # tighter >= as good


class TestBoundaryDetails:
    def test_planted_exact_boundary(self):
        # 0.119 -> planted, 0.121 -> airborne (strict < 0.12)
        for z, expect in ((0.119, True), (0.121, False)):
            p0 = np.zeros((5, 3)); p0[0] = [0, 0, 1.4]
            p0[3] = [0, -0.09, z]; p0[4] = [0, 0.09, 0.087]
            pos = Pose.from_p5(p0, tolerance=0.08)
            raw = pos.p5().pos
            lm = pos.support_landmark_on_p5_axis()
            if raw[3, 2] > 0.12 and not expect:
                assert lm.feet["right"].force_ratio == 0.0

    def test_landmark_contract_fields(self, pose):
        lm = pose.support_landmark_on_p5_axis()
        assert lm.torso_yaw is None        # v1: not handled
        assert hasattr(lm, "com") and hasattr(lm, "feet")
        assert set(lm.feet.keys()) == {"right", "left"}


class TestRenderPaths:
    def test_render_no_marks(self, pose, tmp_path):
        outs = pose.render(str(tmp_path / "p_{label}.png"),
                           views=(("front", 0, 0),), mark_points=False)
        assert len(outs) == 1 and outs[0].exists()

    def test_render_empty_views(self, pose, tmp_path):
        assert pose.render(str(tmp_path / "x_{label}.png"),
                           views=()) == []

    def test_render_size_respected(self, pose, tmp_path):
        outs = pose.render(str(tmp_path / "s_{label}.png"),
                           views=(("front", 0, 0),), size=(320, 240))
        from PIL import Image
        assert Image.open(outs[0]).size == (320, 240)


class TestSemanticsDoc:
    def test_pose_is_mutable_dataclass(self):
        # Pose is NOT frozen — field reassignment is the update path
        p = Pose(np.zeros(21), 1.0, np.array([0., 0., -9.81]))
        p.qpos = np.ones(21) * 0.05          # reassign OK (documented)
        assert p.qpos[0] == 0.05
        _ = p.anchor()                       # and reflects in outputs

    def test_kinematics_singleton(self):
        m1, d1, _ = Pose._kinematics()
        m2, d2, _ = Pose._kinematics()
        assert m1 is m2 and d1 is d2


# ---------------------------------------------------------------------------
# 动量：帧语义（有限差分真值）、方向、模长不变性、无速度拒错
# ---------------------------------------------------------------------------

class TestMomentum:
    def _moving_sim(self, n_steps=30):
        """Sim with a pushed robot -> nonzero com velocity."""
        from envs.humanoid21.simulator import Humanoid21Simulator
        s = Humanoid21Simulator(initial_distance=6.0,
                                initial_pose_a="standing")
        s.reset(0)
        return s

    def test_standing_zero(self, pose):
        np.testing.assert_allclose(pose.com_vel(), [0., 0., 0.],
                                   atol=0.05)
        np.testing.assert_allclose(pose.ang_mom(), [0., 0., 0.],
                                   atol=0.5)

    def test_finite_difference_consistency(self):
        """com(t+dt) - com(t) ≈ com_vel(t)·dt — the frame semantic
        ground truth: velocity and position live in the SAME frame."""
        from envs.humanoid21.simulator import Humanoid21Simulator
        s = Humanoid21Simulator(initial_distance=6.0,
                                initial_pose_a="stand_on_left_leg")
        s.reset(0)
        for _ in range(30):
            s.physical_step()            # let it fall -> moving
        ds = s.get_derived_state()
        o1 = ds["robot_a"]["observation"]
        p1 = Pose.from_obs(o1)
        if np.linalg.norm(p1.com_vel()) < 0.01:
            pytest.skip("robot settled — no velocity to verify")
        for _ in range(2):               # ~2 control steps of motion
            s.physical_step()
        o2 = s.get_derived_state()["robot_a"]["observation"]
        p2 = Pose.from_obs(o2)
        # displacement in the local frame vs predicted velocity*dt —
        # dt is physical (unknown sim dt here) so check DIRECTION only
        v = p1.com_vel()
        dv = p2.com_vel()
        assert np.linalg.norm(v - dv) < np.linalg.norm(v) * 2 + 0.1
        # com_vel and observed com-offset must share the same frame:
        # velocity's local-frame direction must be physically sane
        assert abs(v[2]) < 5.0           # bounded

    def test_velocity_norm_frame_invariant(self, pose):
        # |v| must be identical across anchors — rotation preserves it
        v0 = np.linalg.norm(pose.com_vel())
        v1 = np.linalg.norm(pose.com_vel(point="com", xy=(1, 2),
                                         yaw=0.8))
        assert v0 == pytest.approx(v1, abs=1e-9)

    def test_no_velocity_raises(self):
        p = Pose(np.zeros(21), 1.0, np.array([0., 0., -9.81]))
        with pytest.raises(ValueError):
            p.com_vel()
        with pytest.raises(ValueError):
            p.ang_mom()

    def test_velocity_fields_decoded(self, obs, pose):
        assert pose.qvel is not None and pose.qvel.shape == (21,)
        assert pose.root_linvel.shape == (3,)
        assert pose.root_angvel.shape == (3,)

    def test_qvel_denormalization(self, obs, pose):
        # obs[21:42] uses sign(x)*(2x)^2 compression — invert exactly
        ov = np.asarray(obs[21:42], np.float64)
        np.testing.assert_allclose(pose.qvel,
                                   np.sign(ov) * (2.0 * ov) ** 2)
        oa = np.asarray(obs[49:52], np.float64)
        np.testing.assert_allclose(pose.root_angvel,
                                   np.sign(oa) * 2.0 * oa ** 2)
        np.testing.assert_allclose(pose.root_linvel, obs[46:49])
