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
