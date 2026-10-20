"""Regression tests for the keypoint two-phase strike experiment."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baseline.experiments_ppo.exp_strike_kp_2phase import (
    SKILLS, StrikeKp2Phase, _STRIKE_STEPS, _WARMUP_SKILLS)
from baseline.humanoid21.hierarchy import kp_kin
from baseline.humanoid21.hierarchy.kp_kin import (
    KP_IDX, KP_NAMES, POWER_FOOT)

CLIPS = Path(__file__).resolve().parent.parent / \
    "humanoid21" / "hierarchy" / "assets" / "clips" / "full"


def _episode(p1=None, p2=None, n=8):
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
        observer_outputs={"kp_track_a": kp,
                          "standing_balance_a": {"potential": np.ones(n)}},
        sampling_contexts={},
        episode_options={"skills": {"robot_a": "jab_l"}},
    )


@pytest.mark.parametrize("p1,p2,lengths", [
    (None, None, [8]), (2, None, [3, 5]), (2, 5, [3, 3]),
    (7, None, [8]),
])
def test_trajectory_boundaries_and_terminal_rewards(p1, p2, lengths):
    exp = StrikeKp2Phase()
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


def test_warmup_jobs_and_power_foot():
    exp = StrikeKp2Phase()
    exp._env_pb = lambda: SimpleNamespace(materialize=lambda **kw: kw)
    bp = SimpleNamespace(cls="test_policy")
    jobs = exp.build_jobs(bp, base_seed=17, n_episodes=150,
                          update=0, stochastic=True)
    steps = set()
    for job in jobs:
        opts = job.episode_options
        assert job.env_bp["max_steps"] == _STRIKE_STEPS
        assert "launch_gate" not in opts
        for rid in ("robot_a", "robot_b"):
            target = opts["skills"][rid]
            w = opts["warmup"][rid]
            assert w["skill"] in _WARMUP_SKILLS
            assert w["skill"] != target
            assert 0 <= w["steps"] <= 20
            assert opts["move_schedule"][rid][0][0] == 30
            steps.add(w["steps"])
    assert 0 in steps and 20 in steps
    assert set(SKILLS) == {"jab_l", "jab_r", "heavy_l", "heavy_r"}
    # rule: jab -> opposite foot, heavy -> punching-side foot
    assert POWER_FOOT["hdm05_3146_88_2895_127"] == "foot_right"
    assert POWER_FOOT["hdm05_3146_88_2895_127_m"] == "foot_left"
    assert POWER_FOOT["cmu_144_20_seg01"] == "foot_right"
    assert POWER_FOOT["cmu_144_20_m_seg01"] == "foot_left"


def test_fk_keypoints_vs_mjforward():
    """Analytic 15-keypoint FK matches mj_forward (body xpos/xanchor)."""
    import mujoco
    from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, RobotFK
    from mocap.humanoid21.tracking.ref_data import RefClip
    fk = RobotFK()
    for cid in ("cmu_144_20_seg01", "hdm05_3146_88_2895_127"):
        ref = RefClip(cid, CLIPS)
        for k in (0, ref.n_ctrl - 1):
            q = {n: float(v) for n, v in
                 zip(CONTROLLED_JOINTS, ref.q_rad[k])}
            pts = kp_kin.fk_keypoints(q, ref.root_pos[k],
                                      ref.root_quat[k])
            pos, _ = fk.forward(fk.qvec(ref.root_pos[k],
                                        ref.root_quat[k], q))
            for i, name in enumerate(KP_NAMES):
                if name in kp_kin._KP_BODY:
                    mjp = pos[kp_kin._KP_BODY[name]]
                else:
                    jid = mujoco.mj_name2id(
                        fk.m, mujoco.mjtObj.mjOBJ_JOINT,
                        kp_kin._KP_JNT[name] + "_a")
                    mjp = fk.d.xanchor[jid]
                assert np.linalg.norm(pts[i] - mjp) < 1e-6


def test_ground_align_and_p2_rear_anchor():
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    p0 = kp_kin.ground_align(kp_kin.ref_keypoints(ref, 0))
    foot_z = p0[[KP_IDX["foot_left"], KP_IDX["foot_right"]], 2]
    assert foot_z.min() - kp_kin.FOOT_SOLE_R == pytest.approx(0.0, abs=1e-9)

    # T2: rear-foot target lands exactly on the actual position (3D).
    actual = np.array([1.23, -0.45, 0.067])
    rear_idx = KP_IDX["foot_right"]  # heavy_r rear foot
    t2 = kp_kin.build_p2_target(
        kp_kin.ref_keypoints(ref, ref.n_ctrl - 1),
        ref.root_pos[0][:2], kp_kin.yaw_of(ref.root_quat[0]),
        np.array([0.5, 0.5]), 0.3, rear_idx, actual)
    assert np.allclose(t2[rear_idx], actual)
    # rigid translation preserves shape
    base = kp_kin.anchored(
        kp_kin.ref_keypoints(ref, ref.n_ctrl - 1),
        ref.root_pos[0][:2], kp_kin.yaw_of(ref.root_quat[0]),
        np.array([0.5, 0.5]), 0.3)
    assert np.allclose(t2 - base, actual - base[rear_idx])


def test_pass_criterion_mean_and_max():
    """晋级判据: mean<0.05 AND max<0.10 (not per-point tolerances)."""
    ok = np.full(15, 0.04)
    assert kp_kin.kp_pass(ok)
    one_bad = ok.copy(); one_bad[3] = 0.09
    assert kp_kin.kp_pass(one_bad)          # max 0.09 < 0.10
    too_far = ok.copy(); too_far[3] = 0.11
    assert not kp_kin.kp_pass(too_far)      # max fails
    mean_bad = np.full(15, 0.055); mean_bad[0] = 0.0
    assert not kp_kin.kp_pass(mean_bad)     # mean fails


def test_observer_targets_and_success():
    """Observer end-to-end on a fake ctx: T1 anchored at first step,
    p1_done fires when live==target, T2 keeps the rear foot fixed."""
    from baseline.humanoid21.hierarchy.keypoint_track_observer import (
        KeypointTrackObserver)
    from mocap.humanoid21.tracking.ref_data import RefClip

    obs = KeypointTrackObserver(agent_id="robot_a", clips_dir=str(CLIPS))
    opts = {"move_schedule": {"robot_a": [[30, 0, 0, 0, 0, 1, 1,
                                         0.438, 0.391, 0.498]]}}
    obs.on_pre_episode(SimpleNamespace(episode_options=opts))
    seg = obs._segments[0]
    ref = seg["ref"]
    assert seg["clip"] == "hdm05_3146_88_2895_127"  # jab_l near landing

    # fake ctx whose live state IS the (grounded, anchored) target
    robot_xy = np.array([0.7, -0.4]); yaw = 0.25
    t1 = kp_kin.anchored(seg["pts0"], seg["ref_xy0"], seg["ref_yaw0"],
                         robot_xy, yaw)
    core = {"root_pos": np.array([robot_xy[0], robot_xy[1], 1.0]),
            "root_rot": np.array([np.cos(yaw / 2), 0, 0,
                                  np.sin(yaw / 2)])}
    # monkeypatch live_keypoints to return the target pose exactly
    import baseline.humanoid21.hierarchy.keypoint_track_observer as mod
    orig = mod.live_keypoints
    mod.live_keypoints = lambda c: t1.copy()
    ctx = SimpleNamespace(accessor=SimpleNamespace(
        get_core_state=lambda: {"robot_a": core}))
    try:
        obs.on_post_action_step(ctx)     # step 0: anchor + perfect match
    finally:
        mod.live_keypoints = orig
    out = obs.get_output()
    assert out["p1_done"] == 1.0
    assert out["phase"] == 1.0
    assert out["kp_max_err_m"] == pytest.approx(0.0, abs=1e-9)
    # T2 built; rear foot target == live rear foot on the success frame
    assert seg["phase"] == 2
    assert np.allclose(seg["T2"][seg["rear_idx"]],
                       t1[seg["rear_idx"]])
    assert np.allclose(seg["rear_anchor"], t1[seg["rear_idx"]])
    # frame-0 target anchored at robot xy (pelvis-relative check)
    assert np.allclose(seg["T1"], t1)
