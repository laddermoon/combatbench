"""E3 verification: support phases, timing logic, RefFrame stream."""
import numpy as np
import pytest

from point5.contracts import Point5State, ankle_mid_xy, to_local
from point5.humanoid21.stand_punch import stand_p5
from point5.humanoid21.executor.clip_cache import build_stand_punch_clip
from point5.humanoid21.executor.contracts import (
    ExecutorConfig, PlanError)
from point5.humanoid21.executor.reference import (
    plan_reference, segment_support, needed_duration, window)


def _cur():
    pos = stand_p5()
    o = ankle_mid_xy(pos)
    p = Point5State(pos=pos)
    object.__setattr__(p, "metadata", dict(p.metadata) | {
        "frame_origin_xy": o.tolist(), "frame_yaw_deg": 0.0})
    return p


def _fp(cur):
    return (np.asarray(cur.metadata["frame_origin_xy"], np.float64),
            float(cur.metadata["frame_yaw_deg"]))


def _plan(cfg=None):
    cur = _cur()
    clip = build_stand_punch_clip(cur, np.array([1.05, 0, 1.15]), "right")
    cfg = cfg or ExecutorConfig()
    return plan_reference(clip, _fp(cur),
                          np.array([0.0, 0.0, 1.0]), np.zeros(3), cfg)


def test_support_phases():
    rp = _plan()
    # segments: start->coil double, coil->swing single(anchor),
    # swing->land single then landing, land->strike double
    sups = [tuple(s.support.tolist()) for s in rp.segments]
    movers = [s.mover for s in rp.segments]
    assert sups[0] == (True, True)
    assert any(s == (False, True) or s == (True, False) for s in sups)
    assert sups[-1] == (True, True)
    assert len(set(m for m in movers if m is not None)) >= 1


def test_events_and_times():
    rp = _plan()
    labels = [(r.kf_idx, r.event) for r in rp.frames if r.event]
    assert [l for _, l in labels] == ["coil", "swing", "land", "strike"]
    ts = [t for _, t in rp.key_times]
    assert ts == sorted(ts) and ts[-1] == rp.total_s <= 6.0


def test_velocity_consistency():
    """Analytic ref velocity matches finite difference of positions."""
    rp = _plan()
    for i in (10, 20, 30):
        a, b = rp.frames[i], rp.frames[i + 1]
        fd = (b.p5_world - a.p5_world) / (b.t - a.t)
        v = (a.p5_vel_world + b.p5_vel_world) / 2
        assert np.linalg.norm(fd - v) < 0.6   # within curve accel slack


def test_tighter_capability_needs_more_time():
    """Slower capability cap -> longer or equal duration."""
    cur = _cur()
    clip = build_stand_punch_clip(cur, np.array([1.05, 0, 1.15]),
                                  "right")
    fast = plan_reference(clip, _fp(cur), np.zeros(3), np.zeros(3),
                          ExecutorConfig()).total_s
    slow = plan_reference(clip, _fp(cur), np.zeros(3), np.zeros(3),
                          ExecutorConfig(v_com=0.4, task_budget_s=12.0)).total_s
    assert slow > fast


def test_budget_rejects_absurd():
    cfg = ExecutorConfig(v_hand=0.05)      # absurdly slow cap
    with pytest.raises(PlanError):
        _plan(cfg)


def test_window_pads_end():
    rp = _plan()
    cfg = ExecutorConfig()
    w = window(rp, rp.total_s - 0.05, 4, cfg)
    assert len(w) == 4 and w[-1] is rp.frames[-1]
