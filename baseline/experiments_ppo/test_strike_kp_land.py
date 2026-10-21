"""Regression tests for the single-target landing-pose experiment."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baseline.experiments_ppo.exp_strike_kp_land import (
    SKILLS, TARGET_SKILL, WARMUP_POLICY_CLS, StrikeKpLand,
    _STRIKE_STEPS, _WARMUP_MAX_STEPS, _WARMUP_SKILLS)

CLIPS = Path(__file__).resolve().parent.parent / \
    "humanoid21" / "hierarchy" / "assets" / "clips" / "full"


def _episode(done=None, n=8, phi_val=1.0):
    p2d = np.zeros(n, np.float32)
    phase = np.full(n, 2.0, np.float32)   # land_only: phase=2 throughout
    if done is not None:
        p2d[done] = 1.0
    kp = {
        "reward": -0.3 * np.ones(n), "phase": phase,
        "p1_done": np.zeros(n), "p2_done": p2d,
        "kp_rms": 0.2 * np.ones(n), "kp_mean_err_m": 0.4 * np.ones(n),
        "kp_max_err_m": 0.4 * np.ones(n),
        "foot_disp_m": 0.03 * np.ones(n), "hand_err_m": 0.1 * np.ones(n),
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


@pytest.mark.parametrize("done,length", [(None, 8), (3, 4), (7, 8)])
def test_single_segment_and_terminal(done, length):
    exp = StrikeKpLand()
    trajs = exp._traj(_episode(done), "robot_a",
                      "standing_balance_a", "kp_track_a")
    assert len(trajs) == 1
    t = trajs[0]
    assert len(t.obs) == length
    assert set(t.channels) == {"r_potential", "r_kp_land"}
    ch = t.channels["r_kp_land"]
    assert ch.is_terminated
    assert t.channels["r_potential"].is_terminated
    assert ch.reward[-1] == pytest.approx(0.7 if done is not None
                                          else -0.3)


def test_phi_squared_gate():
    exp = StrikeKpLand()
    t = exp._traj(_episode(phi_val=0.5), "robot_a",
                  "standing_balance_a", "kp_track_a")[0]
    assert np.allclose(t.channels["r_kp_land"].actor_weight,
                       4.0 * 0.5 ** 2)
    assert t.channels["r_potential"].actor_weight == 4.0


def test_jobs_single_skill_frozen_warmup():
    exp = StrikeKpLand()
    exp._env_pb = lambda: SimpleNamespace(materialize=lambda **kw: kw)
    bp = SimpleNamespace(cls="test_policy")
    jobs = exp.build_jobs(bp, base_seed=17, n_episodes=200,
                          update=0, stochastic=True)
    steps = set()
    for job in jobs:
        opts = job.episode_options
        assert job.env_bp["max_steps"] == _STRIKE_STEPS
        assert opts["warmup_policy_cls"] == WARMUP_POLICY_CLS
        for rid in ("robot_a", "robot_b"):
            assert opts["skills"][rid] == TARGET_SKILL
            w = opts["warmup"][rid]
            assert w["skill"] in _WARMUP_SKILLS
            assert w["skill"] not in SKILLS
            assert 0 <= w["steps"] <= _WARMUP_MAX_STEPS
            assert opts["move_schedule"][rid][0][0] == _STRIKE_STEPS
            steps.add(w["steps"])
    assert 0 in steps and _WARMUP_MAX_STEPS in steps
    assert TARGET_SKILL in SKILLS


def test_land_only_observer_direct_t2():
    """land_only: no p1 phase — T2 built at anchor, rear foot pinned."""
    from baseline.humanoid21.hierarchy.keypoint_track_observer import (
        KeypointTrackObserver)
    from baseline.humanoid21.hierarchy import kp_kin
    import baseline.humanoid21.hierarchy.keypoint_track_observer as mod

    obs = KeypointTrackObserver(agent_id="robot_a", land_only=True,
                                clips_dir=str(CLIPS))
    opts = {"move_schedule": {"robot_a": [[60, 0, 0, 0, 0, 1, -1,
                                         0.438, -0.391, 0.498]]}}
    obs.on_pre_episode(SimpleNamespace(episode_options=opts))
    seg = obs._segments[0]
    assert seg["clip"] == "hdm05_3146_88_2895_127_m"  # jab_r mirrored

    # fake ctx: live state arbitrary; rear-foot keypoint must anchor
    live_pts = np.zeros((15, 3))
    live_pts[seg["rear_idx"]] = np.array([1.0, 2.0, 0.03])
    orig = mod.live_keypoints
    mod.live_keypoints = lambda c: live_pts.copy()
    ctx = SimpleNamespace(accessor=SimpleNamespace(
        get_core_state=lambda: {"robot_a": {
            "root_pos": np.array([0.5, 0.5, 1.0]),
            "root_rot": np.array([1.0, 0.0, 0.0, 0.0])}}))
    try:
        obs.on_post_action_step(ctx)
    finally:
        mod.live_keypoints = orig
    out = obs.get_output()
    assert out["phase"] == 2.0          # never went through phase 1
    assert out["p1_done"] == 0.0
    assert seg["T1"] is None
    assert np.allclose(seg["T2"][seg["rear_idx"]],
                       live_pts[seg["rear_idx"]])
    assert np.allclose(seg["rear_anchor"], live_pts[seg["rear_idx"]])
