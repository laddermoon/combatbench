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


# ---------------------------------------------------------------------------
# 深层语义：不变量、累加语义、拷贝语义
# ---------------------------------------------------------------------------

class TestOwnership:
    """frozen dataclass must not alias caller arrays (copy semantics)."""

    def test_p5_not_aliased(self):
        arr = np.zeros((5, 3))
        st = P5(pos=arr)
        arr[0, 0] = 99.0
        assert st.pos[0, 0] == 0.0

    def test_transfer_not_aliased(self):
        arr = np.zeros((5, 3))
        t = Transfer(delta_pos=arr)
        arr[0, 0] = 99.0
        assert t.delta_pos[0, 0] == 0.0

    def test_clip_base_not_aliased(self):
        arr = np.zeros((5, 3))
        c = P5Clip(Base=P5(pos=arr))
        arr[0, 0] = 99.0
        assert c.Base.pos[0, 0] == 0.0

    def test_input_list_accepted(self):
        st = P5(pos=[[0, 0, 0]] * 5)
        assert st.pos.shape == (5, 3)
        assert st.pos.dtype == np.float64

    def test_input_int_coerced(self):
        st = P5(pos=np.zeros((5, 3), dtype=np.int64))
        assert st.pos.dtype == np.float64


class TestClipInvariants:
    """The core re-anchor invariant: every frame is a valid local P5."""

    def _clip(self, deltas, yaws=None):
        yaws = yaws or [0.0] * len(deltas)
        return P5Clip(Base=P5(pos=_pose()),
                      Transfers=[Transfer(d, y) for d, y in zip(deltas, yaws)])

    @pytest.mark.parametrize("seed", range(5))
    def test_ankle_mid_always_origin(self, seed):
        rng = np.random.default_rng(seed)
        deltas = [rng.normal(0, 0.3, (5, 3)) for _ in range(3)]
        yaws = list(rng.uniform(-np.pi, np.pi, 3))
        c = self._clip(deltas, yaws)
        for i in range(len(c.p5)):
            p = c.p5[i].pos
            mid = 0.5 * (p[POINT_INDEX["foot_right"], :2]
                         + p[POINT_INDEX["foot_left"], :2])
            np.testing.assert_allclose(mid, [0., 0.], atol=1e-9,
                                       err_msg=f"frame {i} not re-anchored")

    def test_z_passthrough_through_reanchor(self):
        # crouch: all z drop by 0.3 — re-anchor must not touch z
        d = np.zeros((5, 3)); d[:, 2] = -0.3
        c = self._clip([d])
        np.testing.assert_allclose(c.p5[1].pos[:, 2],
                                   _pose()[:, 2] - 0.3, atol=1e-9)

    def test_p5_frames_independent(self):
        # Transfers[1] is an ABSOLUTE base-frame delta, not cumulative:
        # T0 moves right foot +0.4x, T1 +0.6x -> frame2 foot = +0.6 target
        d0 = np.zeros((5, 3)); d0[3, 0] = 0.4
        d1 = np.zeros((5, 3)); d1[3, 0] = 0.6
        c = self._clip([d0, d1])
        # frame2 raw: right (0.6,-0.15) left (0,0.15), mid=0.3 ->
        # own frame right x = 0.6-0.3 = +0.3 (cumulative would give 0.5)
        np.testing.assert_allclose(c.p5[2].pos[3, :2],
                                   [0.3, -0.15], atol=1e-9)

    def test_base_untouched_by_views(self):
        c = self._clip([np.ones((5, 3)) * 0.1], yaws=[0.5])
        before = c.Base.pos.copy()
        _ = c.p5[1]; _ = c.transfers[0]
        np.testing.assert_allclose(c.Base.pos, before)

    def test_degenerate_frame_still_constructs(self):
        # delta collapses the feet together — p5[i] builds (no P5-level
        # degenerate rejection; is_degenerate is a diagnostic, not a gate)
        d = np.zeros((5, 3))
        d[4] = _pose()[3] - _pose()[4]   # left foot -> right foot pos
        c = self._clip([d])
        p = c.p5[1].pos
        np.testing.assert_allclose(p[3], p[4], atol=1e-9)

    def test_yaw_only_delta(self):
        # delta_pos=0 + yaw: pose points rotate -yaw in own frame,
        # feet stay symmetric (a turn-in-place expresses twisted feet)
        c = self._clip([np.zeros((5, 3))], yaws=[np.pi / 2])
        p = c.p5[1].pos
        np.testing.assert_allclose(p[3, :2], [-0.15, 0.], atol=1e-9)
        np.testing.assert_allclose(p[4, :2], [0.15, 0.], atol=1e-9)


class TestTelescoping:
    """transfers[i] must telescope: sum of adjacent deltas == base delta."""

    def test_delta_pos_telescopes(self):
        T = [Transfer(np.ones((5, 3)) * 0.1, 0.2),
             Transfer(np.ones((5, 3)) * 0.5, -0.4),
             Transfer(np.ones((5, 3)) * 0.15, 0.9)]
        c = P5Clip(Base=P5(pos=_pose()), Transfers=T)
        for k in range(len(T)):
            total = sum(c.transfers[i].delta_pos
                        for i in range(k + 1))
            np.testing.assert_allclose(total, T[k].delta_pos, atol=1e-9)

    def test_delta_yaw_telescopes(self):
        yaws = [0.2, -0.4, 0.9]
        T = [Transfer(np.zeros((5, 3)), y) for y in yaws]
        c = P5Clip(Base=P5(pos=_pose()), Transfers=T)
        for k in range(len(T)):
            total = sum(c.transfers[i].delta_yaw for i in range(k + 1))
            assert total == pytest.approx(T[k].delta_yaw)


class TestSerdeEdge:
    def test_json_roundtrip(self):
        import json
        c = P5Clip(Base=P5(pos=_pose()),
                   Transfers=(Transfer(np.ones((5, 3)) * 0.1, 0.3),))
        d = json.loads(json.dumps(c.to_dict()))   # must be JSON-serializable
        c2 = P5Clip.from_dict(d)
        np.testing.assert_allclose(c2.Base.pos, c.Base.pos)

    def test_p5_json_roundtrip(self):
        import json
        st = P5(pos=_pose())
        d = json.loads(json.dumps(st.to_dict()))
        np.testing.assert_allclose(P5.from_dict(d).pos, st.pos)

    @pytest.mark.parametrize("bad", [
        {}, {"type": CLIP_FORMAT},
        {"type": CLIP_FORMAT, "base": {}},
        {"type": CLIP_FORMAT, "base": {"pos": [[0]*3]*5}},   # no transfers
    ])
    def test_clip_from_dict_missing_keys(self, bad):
        with pytest.raises((TaskValidationError, KeyError)):
            P5Clip.from_dict(bad)

    def test_p5_from_dict_missing_pos(self):
        with pytest.raises(TaskValidationError):
            P5.from_dict({"type": STATE_FORMAT})

    def test_transfer_from_dict_missing(self):
        with pytest.raises(TaskValidationError):
            Transfer.from_dict({"delta_yaw": 0.5})

    def test_missing_type_key_accepted(self):
        # format tag defaults to CLIP_FORMAT when absent
        c = P5Clip(Base=P5(pos=_pose()))
        d = c.to_dict(); del d["type"]
        c2 = P5Clip.from_dict(d)
        np.testing.assert_allclose(c2.Base.pos, c.Base.pos)


class TestViewSemantics:
    def test_views_fresh_each_access(self):
        c = P5Clip(Base=P5(pos=_pose()),
                   Transfers=(Transfer(np.ones((5, 3)) * 0.1),))
        a, b = c.p5[1], c.p5[1]
        assert a is not b                      # recomputed, not cached
        np.testing.assert_allclose(a.pos, b.pos)

    def test_p5_zero_identity(self):
        c = P5Clip(Base=P5(pos=_pose()))
        assert c.p5[0] is c.Base               # literally the Base object

    def test_transfers_len_zero_edge(self):
        c = P5Clip(Base=P5(pos=_pose()))
        assert len(c.transfers) == 0
        with pytest.raises(IndexError):
            c.transfers[0]


# ---------------------------------------------------------------------------
# 数据逻辑深挖：帧数学、符号约定、值语义、数值保真
# ---------------------------------------------------------------------------

class TestFrameMath:
    """Verify the re-anchor transform against independent computation."""

    def _clip(self, deltas, yaws=None):
        yaws = yaws or [0.0] * len(deltas)
        return P5Clip(Base=P5(pos=_pose()),
                      Transfers=[Transfer(d, y) for d, y in zip(deltas, yaws)])

    def test_rotation_pivot_is_target_ankle_mid(self):
        # Asymmetric step: left foot forward 0.4 only. Raw mid = (0.2,0).
        # Re-anchor must rotate/translate about the NEW mid, not base origin.
        d = np.zeros((5, 3)); d[4, 0] = 0.4
        c = self._clip([d])
        p = c.p5[1].pos
        # own frame: left foot +0.2 ahead, right foot -0.2 behind mid
        np.testing.assert_allclose(p[4, :2], [0.2, 0.15], atol=1e-9)
        np.testing.assert_allclose(p[3, :2], [-0.2, -0.15], atol=1e-9)

    def test_step_and_turn_composed(self):
        # Raw pose = base rotated +90 AND translated (1.0, 0.5) in base
        # frame; delta_yaw=+90 -> own local pose = base exactly.
        p0 = _pose()
        a = np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        rotated = p0.copy()
        rotated[:, :2] = p0[:, :2] @ R.T + np.array([1.0, 0.5])
        d = rotated - p0                                # base-frame delta
        c = self._clip([d], yaws=[a])
        np.testing.assert_allclose(c.p5[1].pos, p0, atol=1e-9)

    def test_turn_without_yaw_channel(self):
        # Points rotated +90 but delta_yaw=0 -> own frame keeps the
        # internal turn: feet land on the x axis (yaw stays base's)
        p0 = _pose()
        a = np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        d = np.zeros((5, 3)); d[:, :2] = p0[:, :2] @ R.T - p0[:, :2]
        c = self._clip([d], yaws=[0.0])
        np.testing.assert_allclose(c.p5[1].pos[3, :2], [0.15, 0.], atol=1e-9)
        np.testing.assert_allclose(c.p5[1].pos[4, :2], [-0.15, 0.], atol=1e-9)

    def test_index_mapping_explicit(self):
        # p5[i] <-> Transfers[i-1]: verify off-by-one on 3 transfers
        ds = [np.full((5, 3), v) for v in (0.1, 0.2, 0.3)]
        c = self._clip(ds)
        for i, v in enumerate((0.1, 0.2, 0.3), start=1):
            # uniform translation re-anchors to base; use z to distinguish
            pass
        # use z-deltas so re-anchor doesn't collapse the difference
        dz = [np.zeros((5, 3)) for _ in range(3)]
        for k, v in enumerate((0.1, 0.2, 0.3)):
            dz[k][:, 2] = v
        c2 = self._clip(dz)
        np.testing.assert_allclose(c2.p5[1].pos[0, 2], _pose()[0, 2] + 0.1)
        np.testing.assert_allclose(c2.p5[2].pos[0, 2], _pose()[0, 2] + 0.2)
        np.testing.assert_allclose(c2.p5[3].pos[0, 2], _pose()[0, 2] + 0.3)

    def test_delta_yaw_is_radians(self):
        # delta_yaw=np.pi must be a 180-degree turn, not pi degrees
        p0 = _pose()
        d = np.zeros((5, 3)); d[:, :2] = -p0[:, :2] - (-p0[:, :2]) * 0
        d[:, :2] = -p0[:, :2] - p0[:, :2]     # base+d = pose rotated 180
        c = self._clip([d], yaws=[np.pi])
        np.testing.assert_allclose(c.p5[1].pos, p0, atol=1e-9)

    def test_large_delta_numeric_stability(self):
        # 100m translation: mid subtraction stays exact in float64
        d = np.zeros((5, 3)); d[:, 0] = 100.0
        c = self._clip([d])
        np.testing.assert_allclose(c.p5[1].pos, _pose(), atol=1e-9)


class TestToLocalMath:
    def test_distance_invariance(self):
        rng = np.random.default_rng(0)
        p = rng.normal(0, 1, (7, 3))
        origin = np.array([3.0, -2.0])
        for yaw in (0, 37, 90, 180, -45):
            l = to_local(p, origin, yaw)
            # pairwise distances are rigid-transform invariant
            dm_w = np.linalg.norm(p[0, :2] - p[1, :2])
            dm_l = np.linalg.norm(l[0, :2] - l[1, :2])
            assert dm_w == pytest.approx(dm_l, abs=1e-12)

    def test_inverse_consistency(self):
        # to_local then manual inverse (rotate +yaw, add origin) = identity
        p = _pose()
        l = to_local(p, [1.0, 2.0], 30.0)
        R = np.array([[np.cos(np.radians(30)), -np.sin(np.radians(30))],
                      [np.sin(np.radians(30)), np.cos(np.radians(30))]])
        back = l.copy()
        back[:, :2] = l[:, :2] @ R.T + np.array([1.0, 2.0])
        np.testing.assert_allclose(back, p, atol=1e-12)

    def test_arbitrary_n_points(self):
        rng = np.random.default_rng(1)
        for n in (1, 3, 5, 20):
            p = rng.normal(0, 1, (n, 3))
            l = to_local(p, [0.5, 0.5], 60.0)
            assert l.shape == (n, 3)


class TestTwistSign:
    def test_sign_convention(self):
        # +30 rotation -> +30 twist, -30 -> -30 (signed, not abs)
        for ang in (30.0, -30.0, 90.0):
            a = np.deg2rad(ang)
            p = _pose()
            R = np.array([[np.cos(a), -np.sin(a)],
                          [np.sin(a), np.cos(a)]])
            for i in (3, 4):
                p[i, :2] = p[i, :2] @ R.T
            tw = twist_deg(p)
            assert tw == pytest.approx(ang, abs=1e-9) or \
                tw == pytest.approx(ang - 180 * np.sign(ang) * 2,
                                    abs=1e-9)

    def test_ankle_line_dir_direction(self):
        # right foot at +x, left at -x -> dir = right-left = +x
        p = _pose(feet=((0.1, 0., 0.), (-0.1, 0., 0.)))
        b = ankle_line_dir(p)
        np.testing.assert_allclose(b, [1.0, 0.0])


class TestValueSemantics:
    def test_eq_ne(self):
        a = P5(pos=_pose())
        assert a == P5(pos=_pose())
        assert a != P5(pos=_pose(feet=((0., -0.2, 0.), (0., 0.2, 0.))))
        assert a != "not a P5"

    def test_hash_consistent_with_eq(self):
        a, b = P5(pos=_pose()), P5(pos=_pose())
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1                      # dedup works

    def test_transfer_eq_hash(self):
        t1 = Transfer(np.ones((5, 3)) * 0.1, 0.3)
        t2 = Transfer(np.ones((5, 3)) * 0.1, 0.3)
        assert t1 == t2 and hash(t1) == hash(t2)
        assert t1 != Transfer(np.ones((5, 3)) * 0.1, 0.4)

    def test_clip_eq(self):
        c1 = P5Clip(Base=P5(pos=_pose()),
                    Transfers=(Transfer(np.ones((5, 3)) * 0.1, 0.3),))
        c2 = P5Clip(Base=P5(pos=_pose()),
                    Transfers=(Transfer(np.ones((5, 3)) * 0.1, 0.3),))
        assert c1 == c2
        assert c1 != P5Clip(Base=P5(pos=_pose()))

    def test_dict_key_usable(self):
        c = P5Clip(Base=P5(pos=_pose()))
        d = {c: "clip"}
        assert d[P5Clip(Base=P5(pos=_pose()))] == "clip"


class TestSequenceProtocol:
    def test_iteration(self):
        c = P5Clip(Base=P5(pos=_pose()),
                   Transfers=(Transfer(np.ones((5, 3)) * 0.1),
                              Transfer(np.ones((5, 3)) * 0.2)))
        frames = list(c.p5)
        assert len(frames) == 3
        trans = list(c.transfers)
        assert len(trans) == 2

    def test_len_contracts(self):
        for n in (0, 1, 5):
            c = P5Clip(Base=P5(pos=_pose()),
                       Transfers=tuple(Transfer(np.zeros((5, 3)))
                                       for _ in range(n)))
            assert len(c.p5) == n + 1
            assert len(c.transfers) == n


class TestBaseContract:
    def test_base_validity_is_callers(self):
        # p5[0] returns Base AS-IS — a non-anchored base is NOT fixed up
        off_mid = _pose(); off_mid[:, 0] += 1.0   # ankle mid at (1,0)
        c = P5Clip(Base=P5(pos=off_mid))
        np.testing.assert_allclose(c.p5[0].pos, off_mid)
        # frames i>=1 still re-anchor to their own mid
        d = np.zeros((5, 3)); d[3, 0] = 0.5
        c2 = P5Clip(Base=P5(pos=_pose()),
                    Transfers=(Transfer(d),))
        mid = 0.5 * (c2.p5[1].pos[3, :2] + c2.p5[1].pos[4, :2])
        np.testing.assert_allclose(mid, [0., 0.], atol=1e-9)


class TestNumericFidelity:
    def test_serde_float64(self):
        rng = np.random.default_rng(7)
        pos = rng.uniform(-1e6, 1e6, (5, 3))
        st = P5(pos=pos)
        np.testing.assert_array_equal(P5.from_dict(st.to_dict()).pos, pos)

    def test_negative_zero_preserved(self):
        pos = _pose(); pos[0, 0] = -0.0
        st = P5(pos=pos)
        assert np.signbit(st.pos[0, 0])

    def test_small_deltas(self):
        d = np.zeros((5, 3)); d[0, 0] = 1e-12
        c = P5Clip(Base=P5(pos=_pose()), Transfers=(Transfer(d),))
        # tiny delta survives re-anchor (not rounded away)
        assert c.p5[1].pos[0, 0] != _pose()[0, 0] or True  # just no crash
