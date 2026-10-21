"""Regression tests for the single-skill from-scratch strike experiment."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baseline.experiments_ppo.exp_strike_kp_solo import (
    SKILLS, TARGET_SKILL, WARMUP_POLICY_CLS, StrikeKpSolo,
    _STRIKE_STEPS, _WARMUP_MAX_STEPS, _WARMUP_SKILLS)


def _episode(p1=None, p2=None, n=8, phi_val=1.0):
    """Synthetic episode: p1/p2 = frame indices where done flags fire."""
    p1d = np.zeros(n, np.float32)
    p2d = np.zeros(n, np.float32)
    phase = np.ones(n, np.float32)
    if p1 is not None:
        p1d[p1] = 1.0
        phase[p1 + 1:] = 2.0
    if p2 is not None:
        p2d[p2] = 1.0
    kp = {
        "reward": -0.3 * np.ones(n), "phase": phase,
        "p1_done": p1d, "p2_done": p2d,
        "kp_rms": 0.2 * np.ones(n), "kp_max_err_m": 0.4 * np.ones(n),
        "foot_disp_m": 0.03 * (phase > 1.5), "hand_err_m": 0.1 * np.ones(n),
        "in_strike": np.ones(n),
    }
    return SimpleNamespace(
        num_frames=n, agent_frame_boundary={"robot_a": n},
        observations={"robot_a": np.arange(n * 66).reshape(n, 66)},
        actions={"robot_a": np.zeros((n, 21))},
        final_observation={"robot_a": np.full(66, -1)},
        observer_outputs={
            "kp_track_a": kp,
            "standing_balance_a": {
                "potential": phi_val * np.ones(n)}},
        sampling_contexts={},
        episode_options={"skills": {"robot_a": TARGET_SKILL}},
    )


@pytest.mark.parametrize("p1,p2,lengths", [
    (None, None, [8]), (2, None, [3, 5]), (2, 5, [3, 3]),
    (7, None, [8]),
])
def test_trajectory_boundaries_and_terminal_rewards(p1, p2, lengths):
    exp = StrikeKpSolo()
    trajs = exp._traj(_episode(p1, p2), "robot_a",
                      "standing_balance_a", "kp_track_a")
    assert [len(t.obs) for t in trajs] == lengths
    assert all("r_potential" in t.channels for t in trajs)
    assert set(trajs[0].channels) == {"r_potential", "r_kp_p1"}
    assert trajs[0].channels["r_kp_p1"].reward[-1] == pytest.approx(
        0.7 if p1 is not None else -0.3)
    assert trajs[0].channels["r_kp_p1"].is_terminated
    assert trajs[0].channels["r_potential"].is_terminated == (
        p1 is None or p1 == 7)
    if p1 is not None and p1 < 7:
        assert set(trajs[1].channels) == {"r_potential", "r_kp_p2"}
        assert np.array_equal(trajs[0].last_obs, trajs[1].obs[0])
        assert trajs[1].channels["r_kp_p2"].reward[-1] == \
            pytest.approx(0.7 if p2 is not None else -0.3)
        assert trajs[1].channels["r_kp_p2"].is_terminated
        assert trajs[1].channels["r_potential"].is_terminated


def test_phi_squared_gate_on_actor_weights():
    """Strike channels' actor_weight = base_aw * phi**2, per frame."""
    exp = StrikeKpSolo()
    trajs = exp._traj(_episode(p1=2, p2=5, phi_val=0.5),
                      "robot_a", "standing_balance_a", "kp_track_a")
    t1, t2 = trajs
    aw1 = t1.channels["r_kp_p1"].actor_weight
    aw2 = t2.channels["r_kp_p2"].actor_weight
    assert np.shape(aw1) == (3,) and np.shape(aw2) == (3,)
    assert np.allclose(aw1, 4.0 * 0.5 ** 2)
    assert np.allclose(aw2, 4.0 * 0.5 ** 2)
    # r_potential stays scalar and ungated
    assert np.isscalar(t1.channels["r_potential"].actor_weight)
    assert t1.channels["r_potential"].actor_weight == 4.0
    # phi=0 frames -> zero strike weight
    trajs = exp._traj(_episode(phi_val=0.0), "robot_a",
                      "standing_balance_a", "kp_track_a")
    assert np.all(trajs[0].channels["r_kp_p1"].actor_weight == 0.0)


def test_warmup_jobs_single_skill_frozen_policy():
    exp = StrikeKpSolo()
    exp._env_pb = lambda: SimpleNamespace(materialize=lambda **kw: kw)
    bp = SimpleNamespace(cls="test_policy")
    jobs = exp.build_jobs(bp, base_seed=17, n_episodes=200,
                          update=0, stochastic=True)
    steps = set()
    wskills = set()
    for job in jobs:
        opts = job.episode_options
        assert job.env_bp["max_steps"] == _STRIKE_STEPS
        # warmup policy is the pinned SF export, never the learner's
        assert opts["warmup_policy_cls"] == WARMUP_POLICY_CLS
        assert opts["warmup_policy_cls"] != "test_policy"
        for rid in ("robot_a", "robot_b"):
            assert opts["skills"][rid] == TARGET_SKILL
            w = opts["warmup"][rid]
            wskills.add(w["skill"])
            assert w["skill"] in _WARMUP_SKILLS
            assert w["skill"] not in SKILLS       # no strike warmup
            assert 0 <= w["steps"] <= _WARMUP_MAX_STEPS
            assert opts["move_schedule"][rid][0][0] == _STRIKE_STEPS
            steps.add(w["steps"])
    assert 0 in steps and _WARMUP_MAX_STEPS in steps
    assert set(_WARMUP_SKILLS) == {
        "move_fwd", "move_back", "move_left", "move_right",
        "turn_left", "turn_right", "step"}
    assert TARGET_SKILL in SKILLS


def test_frozen_warmup_export_exists():
    """The pinned warmup policy export actually resolves on disk."""
    cls_str = WARMUP_POLICY_CLS
    assert cls_str.startswith("file:")
    path = Path(cls_str[5:].split(":")[0])
    assert path.is_file()
    assert (path.parent / "model.pt").is_file()
