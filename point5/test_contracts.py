"""contracts.py contract tests — 基石级模块的全表面覆盖。

覆盖：常量/索引、异常层级、P5（构造/校验/冻结/序列化/方法）、
几何推导（ankle_line_dir/is_degenerate/twist_deg）、to_local、
Transfer（校验/序列化）、P5Clip（p5[i] 索引视图/transfers[i]
相邻转换/序列化/边界）。
"""
import numpy as np
import pytest

from point5.contracts import (
    CLIP_FORMAT,
    GROUND_Z,
    MIN_FOOT_SEP_XY,
    N_POINTS,
    POINT_INDEX,
    POINT_NAMES,
    P5Clip,
    P5,
    Point5Error,
    STATE_FORMAT,
    TaskValidationError,
    Transfer,
    ankle_line_dir,
    is_degenerate,
    to_local,
    twist_deg,
)


def _pose(feet=((0., -0.15, 0.05), (0., 0.15, 0.05)),
          head=(0., 0., 1.3), hr=(-0.1, -0.2, 0.8), hl=(0.1, 0.2, 0.8)):
    """A valid local P5: ankle-mid origin, torso +x, feet symmetric."""
    return np.array([head, hr, hl, feet[0], feet[1]])


# ---------------------------------------------------------------------------
# constants & index maps
# ---------------------------------------------------------------------------

class TestConstants:
    def test_point_names(self):
        assert POINT_NAMES == ("head", "hand_right", "hand_left",
                               "foot_right", "foot_left")
        assert N_POINTS == 5

    def test_point_index_bijection(self):
        assert POINT_INDEX == {n: i for i, n in enumerate(POINT_NAMES)}
        assert len(set(POINT_INDEX.values())) == N_POINTS

    def test_formats(self):
        assert STATE_FORMAT == "p5.v2"
        assert CLIP_FORMAT == "p5clip.v3"
        assert GROUND_Z == 0.0


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(TaskValidationError, Point5Error)
        assert issubclass(Point5Error, ValueError)


# ---------------------------------------------------------------------------
# P5
# ---------------------------------------------------------------------------

class TestP5:
    def test_accepts_valid(self):
        st = P5(pos=_pose())
        assert st.pos.shape == (5, 3)
        assert st.pos.dtype == np.float64

    @pytest.mark.parametrize("shape", [(4, 3), (5, 2), (6, 3), (3,), (5, 3, 1)])
    def test_rejects_bad_shape(self, shape):
        with pytest.raises(TaskValidationError):
            P5(pos=np.zeros(shape))

    @pytest.mark.parametrize("fill", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite(self, fill):
        p = _pose(); p[0, 0] = fill
        with pytest.raises(TaskValidationError):
            P5(pos=p)

    def test_frozen(self):
        st = P5(pos=_pose())
        with pytest.raises(AttributeError):
            st.pos = np.zeros((5, 3))

    def test_point_pos_all_names(self):
        st = P5(pos=_pose())
        for i, n in enumerate(POINT_NAMES):
            np.testing.assert_allclose(st.point_pos(n), st.pos[i])

    def test_point_pos_unknown(self):
        st = P5(pos=_pose())
        with pytest.raises(TaskValidationError):
            st.point_pos("knee")

    def test_methods_delegation(self):
        st = P5(pos=_pose())
        assert st.twist_deg() == twist_deg(st.pos)
        assert st.is_degenerate() == is_degenerate(st.pos)

    def test_serde_roundtrip(self):
        st = P5(pos=_pose())
        d = st.to_dict()
        assert d["type"] == STATE_FORMAT
        np.testing.assert_allclose(P5.from_dict(d).pos, st.pos)


# ---------------------------------------------------------------------------
# geometry derivations
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_ankle_line_dir_unit(self):
        b = ankle_line_dir(_pose())
        np.testing.assert_allclose(b, [0.0, -1.0])   # right - left
        assert np.linalg.norm(b) == pytest.approx(1.0)

    def test_ankle_line_dir_degenerate(self):
        p = _pose(feet=((0., 0., 0.), (0., 0.005, 0.)))
        assert ankle_line_dir(p) is None

    def test_is_degenerate_boundary(self):
        # n >= MIN_FOOT_SEP_XY is NOT degenerate; strictly below is
        p = _pose(feet=((0., 0., 0.), (0., MIN_FOOT_SEP_XY * 0.5, 0.)))
        assert is_degenerate(p)
        p = _pose(feet=((0., 0., 0.), (0., MIN_FOOT_SEP_XY, 0.)))
        assert not is_degenerate(p)

    def test_twist_square_stance_zero(self):
        assert twist_deg(_pose()) == pytest.approx(0.0)

    def test_twist_rotated_ankle_line(self):
        a = np.deg2rad(30.0)
        p = _pose()
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for i in (3, 4):
            p[i, :2] = p[i, :2] @ R.T
        assert abs(twist_deg(p)) == pytest.approx(30.0)

    def test_twist_degenerate_none(self):
        p = _pose(feet=((0., 0., 0.), (0., 0., 0.)))
        assert twist_deg(p) is None


class TestToLocal:
    def test_identity(self):
        p = _pose()
        np.testing.assert_allclose(to_local(p, [0., 0.], 0.0), p,
                                   atol=1e-12)

    def test_translation(self):
        p = _pose()
        l = to_local(p, [1.0, 2.0], 0.0)
        np.testing.assert_allclose(l[:, :2], p[:, :2] - [1.0, 2.0])

    def test_rotation_90(self):
        # world +x in a frame facing +y -> local -y (right side)
        l = to_local(np.array([[1., 0., 0.]]), [0., 0.], 90.0)
        np.testing.assert_allclose(l[0, :2], [0., -1.], atol=1e-12)

    def test_z_passthrough(self):
        l = to_local(np.array([[1., 0., 5.]]), [0., 0.], 90.0)
        assert l[0, 2] == 5.0


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

class TestTransfer:
    def test_defaults(self):
        t = Transfer(delta_pos=np.zeros((5, 3)))
        assert t.delta_yaw == 0.0
        assert t.delta_pos.shape == (5, 3)

    @pytest.mark.parametrize("shape", [(4, 3), (5, 2), (3,)])
    def test_rejects_bad_delta_pos(self, shape):
        with pytest.raises(TaskValidationError):
            Transfer(delta_pos=np.zeros(shape))

    def test_rejects_nonfinite_yaw(self):
        with pytest.raises(TaskValidationError):
            Transfer(delta_pos=np.zeros((5, 3)), delta_yaw=np.nan)

    def test_frozen(self):
        t = Transfer(delta_pos=np.zeros((5, 3)))
        with pytest.raises(AttributeError):
            t.delta_yaw = 1.0

    def test_serde_roundtrip(self):
        t = Transfer(delta_pos=np.ones((5, 3)) * 0.2, delta_yaw=0.7)
        t2 = Transfer.from_dict(t.to_dict())
        np.testing.assert_allclose(t2.delta_pos, t.delta_pos)
        assert t2.delta_yaw == 0.7


# ---------------------------------------------------------------------------
# P5Clip: p5[i] index view
# ---------------------------------------------------------------------------

class TestP5Frames:
    def _clip(self, deltas, yaws=None):
        yaws = yaws or [0.0] * len(deltas)
        return P5Clip(Base=P5(pos=_pose()),
                      Transfers=[Transfer(delta_pos=d, delta_yaw=y)
                                 for d, y in zip(deltas, yaws)])

    def test_p5_zero_is_base(self):
        c = self._clip([np.ones((5, 3))])
        assert c.p5[0] is c.Base
        np.testing.assert_allclose(c.p5[0].pos, c.Base.pos)

    def test_len(self):
        c = self._clip([np.ones((5, 3)), np.ones((5, 3))])
        assert len(c.p5) == 3

    def test_uniform_translation_reanchors(self):
        # whole pose shifted +0.3x -> re-anchored p5 identical to base
        d = np.zeros((5, 3)); d[:, 0] = 0.3
        c = self._clip([d])
        np.testing.assert_allclose(c.p5[1].pos, c.Base.pos, atol=1e-9)

    def test_step_keeps_shape_offsets(self):
        # only the right foot steps forward -> asymmetry survives re-anchor
        d = np.zeros((5, 3)); d[3, 0] = 0.4
        c = self._clip([d])
        p = c.p5[1].pos
        assert p[3, 0] > 0.05          # right foot ahead of its own origin
        assert p[4, 0] < -0.05         # left foot behind
        mid = 0.5 * (p[3, :2] + p[4, :2])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)

    def test_delta_yaw_rotates_local_frame(self):
        # pose rotated +90deg about origin, delta_yaw=+90 -> local pose
        # identical to base (its own heading becomes +x)
        p0 = _pose()
        a = np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        d = np.zeros((5, 3))
        d[:, :2] = p0[:, :2] @ R.T - p0[:, :2]    # so base+d = rotated pose
        c = self._clip([d], yaws=[np.pi / 2])
        np.testing.assert_allclose(c.p5[1].pos, p0, atol=1e-9)

    def test_returns_p5_type(self):
        c = self._clip([np.ones((5, 3))])
        assert isinstance(c.p5[1], P5)

    @pytest.mark.parametrize("i", [-1, 3, 99])
    def test_out_of_range(self, i):
        c = self._clip([np.ones((5, 3)), np.ones((5, 3))])
        with pytest.raises(IndexError):
            c.p5[i]


# ---------------------------------------------------------------------------
# P5Clip: transfers[i] adjacent transitions
# ---------------------------------------------------------------------------

class TestTransferSeq:
    def _clip(self):
        return P5Clip(Base=P5(pos=_pose()),
                      Transfers=[Transfer(np.ones((5, 3)) * 0.1, 0.3),
                                 Transfer(np.ones((5, 3)) * 0.25, 0.8)])

    def test_zero_is_base_to_frame1(self):
        c = self._clip()
        np.testing.assert_allclose(c.transfers[0].delta_pos,
                                   c.Transfers[0].delta_pos)
        assert c.transfers[0].delta_yaw == 0.3

    def test_adjacent_difference(self):
        c = self._clip()
        t = c.transfers[1]                       # frame1 -> frame2
        np.testing.assert_allclose(t.delta_pos,
                                   np.ones((5, 3)) * (0.25 - 0.1))
        assert t.delta_yaw == pytest.approx(0.8 - 0.3)

    def test_returns_transfer_type(self):
        assert isinstance(self._clip().transfers[1], Transfer)

    def test_len(self):
        assert len(self._clip().transfers) == 2

    @pytest.mark.parametrize("i", [-1, 2, 5])
    def test_out_of_range(self, i):
        with pytest.raises(IndexError):
            self._clip().transfers[i]


# ---------------------------------------------------------------------------
# P5Clip: construction & serialization
# ---------------------------------------------------------------------------

class TestP5Clip:
    def test_empty_transfers(self):
        c = P5Clip(Base=P5(pos=_pose()))
        assert len(c.p5) == 1
        assert len(c.transfers) == 0

    def test_list_coerced_to_tuple(self):
        c = P5Clip(Base=P5(pos=_pose()),
                   Transfers=[Transfer(np.zeros((5, 3)))])
        assert isinstance(c.Transfers, tuple)

    def test_frozen(self):
        c = P5Clip(Base=P5(pos=_pose()))
        with pytest.raises(AttributeError):
            c.Base = P5(pos=_pose())

    def test_serde_roundtrip(self):
        c = P5Clip(Base=P5(pos=_pose()),
                   Transfers=(Transfer(np.ones((5, 3)) * 0.1, 0.3),
                              Transfer(np.ones((5, 3)) * 0.25, 0.8)))
        d = c.to_dict()
        assert d["type"] == CLIP_FORMAT
        c2 = P5Clip.from_dict(d)
        np.testing.assert_allclose(c2.Base.pos, c.Base.pos)
        np.testing.assert_allclose(c2.p5[1].pos, c.p5[1].pos)
        np.testing.assert_allclose(c2.transfers[1].delta_pos,
                                   c.transfers[1].delta_pos)

    def test_serde_rejects_wrong_format(self):
        c = P5Clip(Base=P5(pos=_pose()))
        d = c.to_dict(); d["type"] = "p5clip.v2"
        with pytest.raises(TaskValidationError):
            P5Clip.from_dict(d)


# ---------------------------------------------------------------------------
# module purity: simulator-unaware
# ---------------------------------------------------------------------------

class TestPurity:
    def test_no_simulator_imports(self):
        import ast
        src = open(__import__('point5.contracts', fromlist=['x'])
                   .__file__).read()
        mods = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
        for m in mods:
            for banned in ("mujoco", "envs", "baseline"):
                assert not m.startswith(banned), f"imports {m}"
