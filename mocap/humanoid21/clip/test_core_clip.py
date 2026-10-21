"""CoreClip format / conversion / recorder / render tests."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as _R

from mocap.humanoid21.clip.core_clip import (
    CONTROL_DT, CORE_FIELDS, CoreClip)
from mocap.humanoid21.clip.convert import refclip_to_coreclip
from mocap.humanoid21.clip.recorder import CoreClipRecorder

CLIPS = Path(__file__).resolve().parents[3] / \
    "baseline" / "humanoid21" / "hierarchy" / "assets" / "clips" / "full"


def _rand_state(rng):
    return {
        "root_pos": rng.uniform(-1, 1, 3).astype(np.float32),
        "root_rot": _R.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
        .astype(np.float32),
        "root_vel_local": rng.uniform(-1, 1, 3).astype(np.float32),
        "root_angular_vel_local": rng.uniform(-1, 1, 3).astype(np.float32),
        "joint_pos_norm": rng.uniform(-1, 1, 21).astype(np.float32),
        "joint_vel_norm": rng.uniform(-1, 1, 21).astype(np.float32),
    }


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    states = [_rand_state(rng) for _ in range(12)]
    ra = rng.uniform(-1, 1, (12, 21)).astype(np.float32)
    clip = CoreClip.from_states(
        states, {"clip_id": "t1",
                 "ref_action_semantics": "pd_target_norm"}, ra)
    assert clip.T == 12 and clip.duration_s == pytest.approx(0.6)
    p = tmp_path / "t1.npz"
    clip.save(p)
    back = CoreClip.load(p)
    for f in CORE_FIELDS:
        assert np.allclose(back.states[f], clip.states[f])
    assert np.allclose(back.ref_action, ra)
    assert back.meta["control_dt"] == CONTROL_DT


def test_validation_catches_errors():
    rng = np.random.default_rng(1)
    st = _rand_state(rng)
    st["root_rot"] = np.array([2.0, 0, 0, 0], np.float32)  # non-unit
    with pytest.raises(ValueError, match="non-unit"):
        CoreClip.from_states([st], {"clip_id": "bad"})
    with pytest.raises(ValueError, match="clip_id"):
        CoreClip.from_states([_rand_state(rng)], {})


def test_state_at_slerp_yaw():
    """Midpoint yaw of a z-rotation clip is the half angle."""
    rng = np.random.default_rng(2)
    s0 = _rand_state(rng)
    s1 = _rand_state(rng)
    s0["root_rot"] = np.array([1.0, 0, 0, 0], np.float32)
    s1["root_rot"] = np.array([np.cos(np.pi / 4), 0, 0,
                               np.sin(np.pi / 4)], np.float32)  # yaw 90°
    clip = CoreClip.from_states([s0, s1], {"clip_id": "yaw"})
    mid = clip.state_at(0.5 * CONTROL_DT)
    yaw = _R.from_quat(np.roll(mid["root_rot"], -1)) \
        .as_euler("xyz")[2]
    assert yaw == pytest.approx(np.pi / 4, abs=1e-5)


def test_slice():
    rng = np.random.default_rng(3)
    clip = CoreClip.from_states(
        [_rand_state(rng) for _ in range(10)], {"clip_id": "s"})
    sub = clip.slice(3, 7)
    assert sub.T == 4
    assert np.allclose(sub.states["root_pos"],
                       clip.states["root_pos"][3:7])
    assert sub.meta["source"]["parent"] == "s"


def test_refclip_conversion():
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    cc = refclip_to_coreclip(ref, start=2, end=12)
    assert cc.T == 10
    # joints: norm vs sim convention is checked separately; here the
    # conversion is defined to reuse RefClip's mid/half which shares
    # the robot_spec source of truth.
    assert np.allclose(cc.states["joint_pos_norm"], ref.q_norm[2:12])
    assert np.allclose(cc.states["joint_vel_norm"], ref.qv_norm[2:12])
    # world -> local velocity: v_local = R^T v_world
    i = 3
    rot = _R.from_quat(np.roll(ref.root_quat[2 + i], -1))
    assert np.allclose(
        cc.states["root_vel_local"][i],
        rot.inv().apply(ref.root_linvel[2 + i]), atol=1e-6)
    assert np.allclose(
        cc.states["root_angular_vel_local"][i],
        rot.inv().apply(ref.root_angvel[2 + i]), atol=1e-6)
    assert cc.meta["ref_action_semantics"] == "pd_target_norm"
    assert np.allclose(cc.ref_action, ref.q_norm[2:12])


def test_norm_params_match_sim():
    """RefClip mid/half == simulator normalization basis."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from mocap.humanoid21.tracking.ref_data import RefClip
    sim = Humanoid21Simulator()
    np_a = sim._norm_params["robot_a"]
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    assert np.allclose(np_a["reference"], ref.mid, atol=1e-5)
    assert np.allclose(np_a["scale"], ref.half, atol=1e-5)
    sim.close()


def test_ground_check_and_fix():
    """Sink a converted clip by 5cm: checker must flag it, fixer must cure it."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from mocap.humanoid21.clip.check import ground_report
    from mocap.humanoid21.clip.fix import fix_ground_penetration
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    clip = refclip_to_coreclip(ref)
    clip.states["root_pos"][:, 2] -= 0.05   # artificial sink
    sim = Humanoid21Simulator()
    rep = ground_report(clip, sim=sim)
    assert rep.max_penetration > 0.04
    assert len(rep.bad_frames) == clip.T
    fixed = fix_ground_penetration(clip, sim=sim)
    rep2 = ground_report(fixed, sim=sim)
    assert rep2.max_penetration <= 0.0025
    # only root z changed; joints/vel untouched
    for f in ("joint_pos_norm", "joint_vel_norm",
              "root_vel_local", "root_angular_vel_local"):
        assert np.allclose(fixed.states[f], clip.states[f])
    assert fixed.meta["source"]["type"] == "fix_ground_penetration"
    sim.close()


def test_recorder_replay_consistency():
    """Record states from a sim, write them back, verify round-trip."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    sim = Humanoid21Simulator()
    rec = CoreClipRecorder("robot_a", clip_id="rt")
    for i in range(5):
        sim.set_action({"robot_a": np.zeros(21),
                        "robot_b": np.zeros(21)})
        for _ in range(25):
            sim.physical_step()
        rec.record_sim(sim, np.zeros(21))
    clip = rec.finish()
    assert clip.T == 5
    # kinematic replay: state -> set_core_state -> get_core_state
    for k in range(clip.T):
        sim.set_core_state({"robot_a": clip.state(k)})
        got = sim.get_core_state()["robot_a"]
        for f in CORE_FIELDS:
            assert np.allclose(got[f], clip.states[f][k], atol=1e-6)
    sim.close()


def test_check_velocity_detects_seam():
    """A clip with a hand-edited pose jump must be flagged by the
    velocity section (sign disagreement and/or accel seam)."""
    from mocap.humanoid21.clip.checks import check_velocity
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    clip = refclip_to_coreclip(ref)
    T = clip.T
    # introduce a per-frame independent edit: jump joint 5 at k=5 only
    clip.states["joint_pos_norm"][5, 5] = np.clip(
        clip.states["joint_pos_norm"][5, 5] + 0.6, -1, 1)
    r = check_velocity(clip)
    assert r.status in ("WARN", "FAIL")
    seams = r.metrics["accel_seam_frames"]
    assert any(abs(k - 5) <= 1 for k in seams) or \
        r.metrics["vel_sign_disagreements"] > 0 or \
        r.metrics["joint_vel_res_max"] > 0.8


def test_check_actuator_screen_fields():
    """Actuator screen returns finite utilization metrics."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from mocap.humanoid21.clip.checks import check_actuator
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    clip = refclip_to_coreclip(ref, start=2, end=8)
    sim = Humanoid21Simulator()
    r = check_actuator(clip, "robot_a", sim)
    sim.close()
    m = r.metrics
    assert "joint_util_max_pct" in m
    assert all(np.isfinite(v) for v in m["joint_util_max_pct"].values())
    assert m["root_wrench_force_max"] >= 0


def test_check_ref_action_replay():
    """ref_action replay produces tracking metrics; a garbage action
    sequence must fail."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from mocap.humanoid21.clip.checks import check_ref_action
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    clip = refclip_to_coreclip(ref, start=2, end=8)
    sim = Humanoid21Simulator()
    r = check_ref_action(clip, "robot_a", sim)
    assert r.status in ("PASS", "WARN", "FAIL")
    assert np.isfinite(r.metrics["joint_err_mean_max"])
    # garbage actions -> should diverge or fall
    clip2 = refclip_to_coreclip(ref, start=2, end=8)
    clip2.ref_action[:] = 1.0
    r2 = check_ref_action(clip2, "robot_a", sim)
    assert r2.status == "FAIL"
    sim.close()


def test_rederive_vel_and_lineage():
    """rederive-vel makes interior velocities exactly consistent with
    poses, keeps endpoints, and records lineage."""
    from mocap.humanoid21.clip.checks import check_velocity
    from mocap.humanoid21.clip.rederive import rederive_velocities
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    clip = refclip_to_coreclip(ref)
    new = rederive_velocities(clip)
    assert new.meta["source"]["type"] == "rederive_vel"
    assert new.meta["lineage"][-1]["op"] == "rederive_vel"
    assert new.meta["lineage"][-1]["parent"] == clip.clip_id
    # parent chain: convert hop present too
    assert any(h["op"] == "convert" for h in new.meta["lineage"])
    # endpoints unchanged
    assert np.allclose(new.states["joint_vel_norm"][0],
                       clip.states["joint_vel_norm"][0])
    # interior residual ~0
    r = check_velocity(new)
    assert r.metrics["joint_vel_res_max"] < 1e-4
    assert r.metrics["vel_sign_disagreements"] == 0
    # poses untouched
    assert np.allclose(new.states["joint_pos_norm"],
                       clip.states["joint_pos_norm"])

    new0 = rederive_velocities(clip, zero_endpoints=True)
    assert np.allclose(new0.states["joint_vel_norm"][0], 0)
    assert np.allclose(new0.states["root_vel_local"][-1], 0)


def test_frame_distance_and_rederive_action():
    """frame_distance components; rederive_actions produces valid
    actions + solver meta; small CEM budget for speed."""
    from mocap.humanoid21.clip.rederive_action import (
        frame_distance, rederive_actions)
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    clip = refclip_to_coreclip(ref, start=2, end=5)
    s = clip.state(0)
    d0 = frame_distance(s, s)
    assert d0["total"] == pytest.approx(0.0)
    s2 = dict(s); s2["joint_pos_norm"] = s["joint_pos_norm"] + 0.5
    d1 = frame_distance(s2, s)
    assert d1["components"]["jp"] == pytest.approx(0.5)
    assert d1["total"] == pytest.approx(0.5 * 1.0)

    new = rederive_actions(clip, n_samples=16, n_iters=2, seed=0)
    assert new.ref_action.shape == (new.T, 21)
    assert np.abs(new.ref_action).max() <= 1.0
    sv = new.meta["ref_action_solver"]
    assert len(sv["per_frame_D"]) == new.T
    assert all(np.isfinite(d) for d in sv["per_frame_D"])
    assert new.meta["ref_action_semantics"] == "best_effort_pd_target"
    assert new.meta["lineage"][-1]["op"] == "rederive_action"
    # states untouched
    for f in CORE_FIELDS:
        assert np.allclose(new.states[f], clip.states[f])
