"""Regression tests for strike-only two-stage episode slicing."""
from types import SimpleNamespace

import numpy as np
import pytest

from baseline.experiments_ppo.exp_strike_2phase import (
    SKILLS, Strike2Phase, _STRIKE_STEPS, _WARMUP_SKILLS)
from baseline.humanoid21.hierarchy.strike_track_observer import (
    strike_power_foot)


def _episode(gate=None, land=None, n=8):
    q0 = np.full(n, 0.4, np.float32)
    ql = np.zeros(n, np.float32)
    gated = np.zeros(n, np.float32)
    if gate is not None:
        q0[gate] = 0.08
        gated[gate + 1:] = 1.0
        ql[gate + 1:] = 0.4
    if land is not None:
        ql[land] = 0.08
    strike = {
        "in_strike": np.ones(n), "gated": gated,
        "gate_qmax": q0, "land_qmax": ql,
        "r_p1_mse": -0.3 * np.ones(n),
        "r_p1_var": -0.1 * np.ones(n),
        "r_p2_mse": -0.4 * np.ones(n),
        "r_p2_var": -0.2 * np.ones(n),
        "r_p2_foot": -0.05 * np.ones(n),
    }
    return SimpleNamespace(
        num_frames=n, agent_frame_boundary={"robot_a": n},
        observations={"robot_a": np.arange(n * 66).reshape(n, 66)},
        actions={"robot_a": np.zeros((n, 21))},
        final_observation={"robot_a": np.full(66, -1)},
        observer_outputs={"strike_track_a": strike,
                          "standing_balance_a": {"potential": np.ones(n)}},
        sampling_contexts={},
    )


@pytest.mark.parametrize("gate,land,lengths", [
    (None, None, [8]), (2, None, [3, 5]), (2, 5, [3, 3]),
    (7, None, [8]),
])
def test_trajectory_boundaries_and_terminal_rewards(gate, land, lengths):
    exp = Strike2Phase()
    trajs = exp._traj(_episode(gate, land), "robot_a",
                      "standing_balance_a", "strike_track_a")
    assert [len(t.obs) for t in trajs] == lengths
    assert all("r_potential" in t.channels for t in trajs)
    assert set(trajs[0].channels) == {
        "r_potential", "r_strike_p1_mse", "r_strike_p1_var"}
    assert trajs[0].channels["r_strike_p1_mse"].reward[-1] == pytest.approx(
        0.7 if gate is not None else -0.3)
    assert trajs[0].channels["r_strike_p1_mse"].is_terminated == (
        gate is not None)
    if gate is not None and gate < 7:
        assert set(trajs[1].channels) == {
            "r_potential", "r_strike_p2_mse", "r_strike_p2_var",
            "r_strike_p2_foot"}
        assert np.array_equal(trajs[0].last_obs, trajs[1].obs[0])
        assert trajs[1].channels["r_strike_p2_mse"].reward[-1] == \
            pytest.approx(0.6 if land is not None else -0.4)
        assert trajs[1].channels["r_strike_p2_mse"].is_terminated == (
            land is not None)
        assert trajs[1].channels["r_potential"].is_terminated == (
            land is not None)


def test_warmup_jobs_and_power_foot():
    exp = Strike2Phase()
    exp._env_pb = lambda: SimpleNamespace(materialize=lambda **kw: kw)
    bp = SimpleNamespace(cls="test_policy")
    jobs = exp.build_jobs(bp, base_seed=17, n_episodes=150,
                          update=0, stochastic=True)
    steps = set()
    for job in jobs:
        opts = job.episode_options
        assert job.env_bp["max_steps"] == _STRIKE_STEPS
        for rid in ("robot_a", "robot_b"):
            target = opts["skills"][rid]
            w = opts["warmup"][rid]
            assert w["skill"] in _WARMUP_SKILLS
            assert w["skill"] != target
            assert 0 <= w["steps"] <= 20
            assert opts["move_schedule"][rid][0][0] == 30
            steps.add(w["steps"])
    assert 0 in steps and 20 in steps
    assert strike_power_foot("cmu_144_20_m_seg01") == "foot_left"
    assert strike_power_foot("cmu_144_20_seg01") == "foot_right"
    assert strike_power_foot("hdm05_3146_88_2895_127") == "foot_right"
    assert strike_power_foot("hdm05_3146_88_2895_127_m") == "foot_left"
    assert set(SKILLS) == {"jab_l", "jab_r", "heavy_l", "heavy_r"}
