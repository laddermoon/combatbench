"""Tests for policy.humanoid21.gait_clock.GaitClockAdapterPolicy.

The adapter presents a standard 96-dim interface to the platform while
internally appending the 3-dim gait clock that step-lineage policies
(trained under GaitClockSimulator, obs_dim=99) expect.
"""
import numpy as np
import pytest

from policy.humanoid21.gait_clock.policy import GaitClockAdapterPolicy


class _EchoInner:
    """Stub inner policy that echoes its observation back as 'action'."""

    def __init__(self):
        self.seen_dims = []
        self.reset_count = 0

    def act(self, observation, want_extra=False):
        obs = np.asarray(observation)
        self.seen_dims.append(obs.shape[-1])
        return obs, {"extra": True} if want_extra else None

    def reset(self, seed=None):
        self.reset_count += 1


def _make(inner=None, **cfg):
    ad = GaitClockAdapterPolicy(
        inner={"version": 1,
               "cls": "policy.random.policy:RandomCombatPolicy",
               "config": {}},
        **cfg)
    if inner is not None:
        ad._inner = inner
    return ad


def test_clock_matches_gait_clock_simulator():
    """Adapter clock sequence == GaitClockSimulator.gait_clock sequence.

    The simulator's counter advances once per set_action call (2/frame in
    a 2-agent env); the adapter's ``counter_step=2`` must reproduce the
    identical sequence of [cmd_L, cmd_R, wprog] values frame by frame.
    """
    from baseline.humanoid21.end2end.gait_clock_simulator import (
        GaitClockSimulator)

    sim = GaitClockSimulator.__new__(GaitClockSimulator)
    sim.gait_period = 40
    sim._action_step = 0

    ad = _make(inner=_EchoInner(), gait_period=40, counter_step=2)
    obs = np.zeros(96, dtype=np.float32)
    for frame in range(60):  # 1.5 gait periods worth of frames
        env_clk = sim.gait_clock()
        out, _ = ad.act(obs)
        np.testing.assert_allclose(out[-3:], env_clk, atol=1e-7,
                                   err_msg=f"frame {frame}")
        # env side: two agents submit actions per frame
        sim._action_step += 2


def test_accepts_96_dim_and_appends_clock():
    inner = _EchoInner()
    ad = _make(inner=inner)
    out, _ = ad.act(np.zeros(96))
    assert inner.seen_dims[-1] == 99
    assert out.shape == (99,)
    np.testing.assert_allclose(out[-3:], [1.0, 0.0, 0.0])


def test_counter_advances_by_counter_step_per_act():
    inner = _EchoInner()
    ad = _make(inner=inner, gait_period=40, counter_step=2)
    obs = np.zeros(96)
    ad.act(obs)
    assert ad._step == 2
    out, _ = ad.act(obs)   # frame 1 → counter was 2 → pos 2 → wprog 0.1
    np.testing.assert_allclose(out[-3:], [1.0, 0.0, 0.1])
    assert ad._step == 4


def test_reset_restores_phase_and_inner():
    inner = _EchoInner()
    ad = _make(inner=inner)
    obs = np.zeros(96)
    for _ in range(15):
        ad.act(obs)
    ad.reset(seed=7)
    assert ad._step == 0
    assert inner.reset_count == 1
    out, _ = ad.act(obs)
    np.testing.assert_allclose(out[-3:], [1.0, 0.0, 0.0])


def test_passthrough_99_dim():
    inner = _EchoInner()
    ad = _make(inner=inner)
    obs99 = np.arange(99, dtype=np.float32)
    out, _ = ad.act(obs99)
    np.testing.assert_array_equal(out, obs99)
    assert ad._step == 2  # counter still advances


def test_rejects_wrong_dim():
    ad = _make(inner=_EchoInner())
    with pytest.raises(ValueError, match="96- or 99-dim"):
        ad.act(np.zeros(50))


def test_loads_via_policy_blueprint():
    """Full blueprint round-trip with a real nested inner policy."""
    from envs.framework.policy import PolicyBlueprint
    ad = PolicyBlueprint.from_dict({
        "version": 1,
        "cls": "policy.humanoid21.gait_clock.policy:GaitClockAdapterPolicy",
        "config": {
            "gait_period": 40,
            "counter_step": 2,
            "inner": {
                "version": 1,
                "cls": "policy.random.policy:RandomCombatPolicy",
                "config": {"scale": 0.0},
            },
        },
    }).build()
    ad.reset()
    a, _ = ad.act(np.zeros(96))
    assert np.asarray(a).shape == (21,)
    assert ad._step == 2
