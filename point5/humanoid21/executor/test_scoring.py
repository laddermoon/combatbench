"""E4 verification: channel sensitivity, dead-zones, fall, pose init."""
import numpy as np
import pytest

from point5.contracts import POINT_INDEX
from point5.humanoid21.executor.contracts import ExecutorConfig, RefFrame
from point5.humanoid21.executor.scoring import score_landed, pose_init
from point5.humanoid21.stand_punch import stand_p5


def _ref():
    return RefFrame(t=0.05, p5_world=np.zeros((5, 3)),
                    p5_vel_world=np.zeros((5, 3)),
                    com_pos=np.array([0., 0., 1.0]),
                    com_vel=np.zeros(3), yaw=0.0, yaw_rate=0.0,
                    support=np.array([True, True]),
                    tol_point=np.full(5, 0.10))


def _meas(**kw):
    pw = np.zeros((5, 3))
    pw[POINT_INDEX["head"], 2] = 1.30    # upright head height
    m = dict(p5_world=pw,
             com_pos=np.array([0., 0., 1.0]), com_vel=np.zeros(3),
             yaw=0.0, torso_tilt=0.0,
             contacts={"extra": 0})
    m.update(kw)
    return m


def _score(m, ref=None, a=None, cfg=None):
    cfg = cfg or ExecutorConfig()
    return score_landed(m, ref or _ref(), np.zeros((5, 3)),
                        np.zeros(21) if a is None else a,
                        np.zeros(21), cfg)


def test_point_deadzone_and_residual():
    pw = _meas()["p5_world"].copy()
    pw[1, 0] = 0.15            # within .10 tol? no -> 0.05 over
    s = _score(_meas(p5_world=pw))
    assert s["components"]["point"] > 0
    pw = _meas()["p5_world"].copy()
    pw[1, 0] = 0.05            # inside tol -> only 0.25*perr remains
    s2 = _score(_meas(p5_world=pw))
    assert s["components"]["point"] > s2["components"]["point"]


def test_com_channel():
    s = _score(_meas(com_pos=np.array([0.2, 0., 1.0])))
    assert s["components"]["com"] > 0.10
    s = _score(_meas(com_pos=np.array([0.02, 0., 1.0])))
    assert s["components"]["com"] == 0.0


def test_fall_penalty():
    s = _score(_meas(torso_tilt=1.2))
    assert s["fallen"] and s["total"] > 40
    s = _score(_meas())
    assert not s["fallen"]


def test_support_slip():
    ref = _ref()
    ref.support = np.array([True, False])
    pv = np.zeros((5, 3))
    pv[POINT_INDEX["foot_right"]] = [0.5, 0, 0]
    s = score_landed(_meas(), ref, pv, np.zeros(21), np.zeros(21),
                     ExecutorConfig())
    assert s["components"]["slip"] > 0.3


def test_pose_init_mapping():
    from point5.humanoid21.executor.sim_iface import (
        action_to_joint, load_interface)
    iface = load_interface()
    a = pose_init(stand_p5(), iface)
    assert a is not None and a.shape == (21,)
    q = action_to_joint(a, iface)
    assert np.abs(q).max() < 3.5         # sane radian targets
