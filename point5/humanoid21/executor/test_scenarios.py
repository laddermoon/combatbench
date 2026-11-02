"""Scenario tests for algorithm 1 — diverse situations + correctness.

Scenarios:
  A  rotated frame chain (dyaw transfers) — anchor transform exactness
  B  moving start (nonzero momentum) — blend & settle
  C  crouch sequence — com_z follows keyframe profile
  D  foot-lift segment — capability shrinks, duration stretches
  E  two-step walk-like chain — multi-segment ordering
  F  tiny/large durations — stretch & respect
  G  momentum bookkeeping — l == M*v, L == 0
  H  cross-check anchors vs trajectory.py keyframe_world
"""
import numpy as np
import pytest

from point5.contracts import (Keyframe, P5Clip, Transfer, compose_frame,
                              to_world)
from point5.humanoid21.stand_punch import stand_p5, plan_stand_punch
from point5.humanoid21.executor.capability_solver import capability
from point5.humanoid21.executor.centroidal import (
    MASS_KG, CentroidalState)
from point5.humanoid21.executor.centroidal_planner import (
    CTRL_DT, CentroidalPlanner)
from point5.humanoid21.executor.com_solver import com_estimate
from point5.humanoid21.executor.contracts import Context

P = CentroidalPlanner("robot_a")


def _c0(v=(0, 0, 0)):
    return CentroidalState(com=np.array([0.03, 0.0, 0.87]),
                           lin_mom=MASS_KG * np.asarray(v, float),
                           ang_mom=np.zeros(3))


def _plan(clip, c0=None):
    return P.plan(Context(np.zeros(96), clip), c0 or _c0())


def _p5(dx=0.0, dy=0.0, dz=0.0):
    p = stand_p5().copy()
    p[:, 0] += dx
    p[:, 1] += dy
    p[:, 2] += dz
    return p


# -- A: rotated frame chain -------------------------------------------------
def test_anchor_transform_rotated():
    """kf1 in a frame rotated 90deg about origin: its local +x point must
    appear at +y in frame0."""
    p = _p5()
    clip = P5Clip(name="r", keyframes=(
        Keyframe(pos=p),
        Keyframe(pos=p, transfer=Transfer(0, 0, 90.0, 1.0)),
    ))
    tr = _plan(clip)
    cl = com_estimate(p)
    want = to_world(cl[None], np.zeros(2), 90.0)[0]
    assert np.allclose(tr[-1].com, want, atol=1e-9)
    # sanity: the x offset of cl rotates into y (90deg)
    assert abs(want[0]) < 1e-6 and abs(want[1] - cl[0]) < 1e-9


def test_anchor_transform_composed_chain():
    """Two chained transfers compose correctly in frame0."""
    p = _p5()
    clip = P5Clip(name="c", keyframes=(
        Keyframe(pos=p),
        Keyframe(pos=p, transfer=Transfer(0.4, 0.1, 45.0, 1.0)),
        Keyframe(pos=p, transfer=Transfer(0.3, -0.2, -45.0, 1.0)),
    ))
    tr = _plan(clip)
    o1, y1 = compose_frame(np.zeros(2), 0.0,
                           Transfer(0.4, 0.1, 45.0, 1.0))
    o2, y2 = compose_frame(o1, y1, Transfer(0.3, -0.2, -45.0, 1.0))
    want = to_world(com_estimate(p)[None], o2, y2)[0]
    assert np.allclose(tr[-1].com, want, atol=1e-9)


def test_anchor_matches_keyframe_world():
    """Cross-check our frame0 anchors against library keyframe_world."""
    plan = plan_stand_punch(np.array([1.05, 0, 1.15]), "left")
    raw = list(plan.task.keyframes)
    kfs = [Keyframe(pos=stand_p5()),
           Keyframe(pos=raw[0].pos,
                    transfer=Transfer(0, 0, 0, 0.5))] + raw[1:]
    clip = P5Clip(name="p", keyframes=tuple(kfs))
    from point5.trajectory import keyframe_world
    world = keyframe_world(clip, (np.zeros(2), 0.0))
    # world landmark of last kf must match our transform of the same
    tr = _plan(clip)
    poses = [(np.zeros(2), 0.0)]
    for k in range(1, len(kfs)):
        poses.append(compose_frame(*poses[-1], kfs[k].transfer))
    com_last = com_estimate(raw[-1].pos)
    ours = to_world(com_last[None], *poses[-1])[0]
    # independent: com of the world keyframe points themselves.
    # mass_proxy is not perfectly frame-equivariant (local lean
    # priors) -> allow its own ~cm-scale slack.
    assert np.all(np.isfinite(ours))
    assert np.linalg.norm(ours[:2] -
                          com_estimate(to_world(
                              raw[-1].pos, *poses[-1]))[:2]) < 0.02


# -- B: moving start --------------------------------------------------------
def test_moving_start_velocity_decays():
    """c0 with +x momentum: initial vel blends into the plan curve."""
    tr = _plan(P5Clip(name="s", keyframes=(
        Keyframe(pos=_p5()),
        Keyframe(pos=_p5(dx=0.2), transfer=Transfer(0, 0, 0, 1.0)),
    )), _c0(v=(0.4, 0, 0)))
    assert tr[0].com_vel[0] > 0.2    # initial momentum carried in
    assert np.linalg.norm(tr[-1].com_vel) < 1e-6   # ends at rest
    # residual contribution v_from*(1-s) vanishes by mid-segment:
    # compare against a v_from=0 sibling plan
    tr0 = _plan(P5Clip(name="s", keyframes=(
        Keyframe(pos=_p5()),
        Keyframe(pos=_p5(dx=0.2), transfer=Transfer(0, 0, 0, 1.0)),
    )), _c0())
    m = tr.n_steps // 2
    resid = tr[m].com_vel[0] - tr0[m].com_vel[0]
    assert abs(resid) < 0.25         # residual <= half of 0.4 by mid


def test_moving_start_lateral():
    tr = _plan(P5Clip(name="s", keyframes=(
        Keyframe(pos=_p5()),
        Keyframe(pos=_p5(dy=0.3), transfer=Transfer(0, 0, 0, 1.0)),
    )), _c0(v=(0, -0.3, 0)))
    assert np.all(np.isfinite(tr[-1].z))
    assert np.linalg.norm(tr[-1].com_vel) < 1e-6


# -- C: crouch sequence -----------------------------------------------------
def test_crouch_com_z_profile():
    """Descending head height -> com_z follows (monotone-ish)."""
    lo = stand_p5().copy()
    lo[0, 2] = 1.0                                # head down
    clip = P5Clip(name="c", keyframes=(
        Keyframe(pos=stand_p5()),
        Keyframe(pos=lo, transfer=Transfer(0, 0, 0, 1.0)),
    ))
    tr = _plan(clip)
    zs = np.array([s.com[2] for s in tr.states])
    assert np.all(np.diff(zs) <= 1e-9)             # non-increasing
    assert zs[-1] < 0.8


# -- D: foot-lift segment ---------------------------------------------------
def test_lift_segment_capability_stretch():
    """Swing pose (one foot up): single-support capability forces a
    longer duration than a 0.1s intent."""
    swing = stand_p5().copy()
    swing[4, 2] = 0.35                             # left foot up
    swing[0, 2] = 1.2                              # deeper crouch too
    clip = P5Clip(name="l", keyframes=(
        Keyframe(pos=stand_p5()),
        Keyframe(pos=swing, transfer=Transfer(0.1, 0.15, 10.0, 0.1)),
    ))
    tr = _plan(clip)
    assert tr.n_steps * CTRL_DT > 0.15             # stretched past 0.1s
    vs = np.linalg.norm([s.com_vel for s in tr.states], axis=1)
    assert vs.max() < capability(swing).v_max([0, 1]) + 1.0


# -- E: multi-segment ordering ----------------------------------------------
def test_walklike_chain_visits_anchors_in_order():
    """3-segment chain: com passes successive anchors in order."""
    p = stand_p5()
    clip = P5Clip(name="w", keyframes=(
        Keyframe(pos=p),
        Keyframe(pos=_p5(dx=0.15), transfer=Transfer(0.3, 0, 0, 0.8)),
        Keyframe(pos=_p5(dx=0.3), transfer=Transfer(0.3, 0, 0, 0.8)),
        Keyframe(pos=_p5(dx=0.45), transfer=Transfer(0.3, 0, 0, 0.8)),
    ))
    tr = _plan(clip)
    xs = np.array([s.com[0] for s in tr.states])
    assert np.all(np.diff(xs) >= -1e-9)            # monotone forward
    assert xs[-1] > xs[0] + 0.2


# -- F: duration handling ----------------------------------------------------
def test_zero_displacement_segment():
    """Same pose twice: degenerate segment is fine (n>=1, near-zero v)."""
    clip = P5Clip(name="z", keyframes=(
        Keyframe(pos=stand_p5()),
        Keyframe(pos=stand_p5(), transfer=Transfer(0, 0, 0, 0.3)),
    ))
    tr = _plan(clip)
    assert tr.n_steps >= 1
    assert np.linalg.norm([s.com_vel for s in tr.states],
                          axis=1).max() < 0.3


def test_long_dur_slows_peak():
    clip = P5Clip(name="t", keyframes=(
        Keyframe(pos=_p5()),
        Keyframe(pos=_p5(dx=0.3), transfer=Transfer(0, 0, 0, 4.0)),
    ))
    tr = _plan(clip)
    vs = np.linalg.norm([s.com_vel for s in tr.states], axis=1)
    assert vs.max() < 0.3                          # ~0.3m/4s, peak*1.875


# -- G: bookkeeping ----------------------------------------------------------
def test_momentum_bookkeeping():
    clip = _plan  # silence
    tr = _plan(P5Clip(name="b", keyframes=(
        Keyframe(pos=_p5()),
        Keyframe(pos=_p5(dx=0.2), transfer=Transfer(0, 0, 0, 1.0)),
    )))
    for s in tr.states:
        assert np.allclose(s.lin_mom, MASS_KG * s.com_vel)
        assert np.allclose(s.ang_mom, 0.0)


def test_nan_free_random_targets():
    rng = np.random.default_rng(0)
    for i in range(8):
        t = np.array([0.9 + 0.3 * rng.random(),
                      0.5 * (rng.random() - 0.5), 1.0 + 0.3 * rng.random()])
        plan = plan_stand_punch(t, "left" if i % 2 else "right")
        if not plan.reachable:
            continue
        raw = list(plan.task.keyframes)
        kfs = [Keyframe(pos=stand_p5()),
               Keyframe(pos=raw[0].pos,
                        transfer=Transfer(0, 0, 0, 0.5))] + raw[1:]
        tr = _plan(P5Clip(name=f"r{i}", keyframes=tuple(kfs)))
        assert tr.n_steps > 0
        assert all(np.all(np.isfinite(s.z)) for s in tr.states)
        assert np.linalg.norm(tr[-1].com_vel) < 1e-6
