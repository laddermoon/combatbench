"""Path-based reclip sweep tests."""
from pathlib import Path

import numpy as np
import pytest

from mocap.humanoid21.clip.convert import refclip_to_coreclip
from mocap.humanoid21.clip.core_clip import CoreClip
from mocap.humanoid21.clip.objective_evaluator import (
    ReclipObjectiveConfig)
from mocap.humanoid21.clip.path_reclip import (
    pd_init_actions, reclip_path_sweep)
from mocap.humanoid21.clip.target_path import TargetPath
from mocap.humanoid21.tracking.ref_data import RefClip

CLIPS = Path(__file__).resolve().parents[3] / \
    "baseline" / "humanoid21" / "hierarchy" / "assets" / "clips" / "full"


def test_path_sweep_executable_result(tmp_path):
    """A small path-sweep run produces simulator landings, a final
    replay, metrics and provenance."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    src = refclip_to_coreclip(ref, start=2, end=6)
    path = TargetPath.from_clip(src, "endpoints", s_ext=0.1)
    objective = ReclipObjectiveConfig(
        kind="pass", time_budget_frames=4, ds_max=0.3)
    sim = Humanoid21Simulator()
    res = reclip_path_sweep(
        src.state(0), path, objective, budget_frames=4,
        sweeps=1, n_samples=8, n_iters=1, seed=0, sim=sim,
        clip_id="path_test")
    sim.close()

    out = res.clip
    assert out.T >= 1
    assert out.ref_action.shape == (out.T, 21)
    assert out.meta["ref_action_semantics"] == "executed_pd_target"
    assert out.meta["source"]["parent"] == src.clip_id
    assert out.meta["lineage"][-1]["op"] == "reclip"
    rc = out.meta["reclip"]
    assert rc["objective"] == "pass"
    assert rc["path"]["mode"] == "endpoints"
    assert rc["solver"]["method"] == "path_sweep"
    assert np.isfinite(rc["solver_score"])
    assert np.isfinite(rc["metrics"]["s_final"])
    assert 0.0 <= rc["metrics"]["s_final"] <= path.s_max
    for key in ("joint_pos_err_mean_max", "joint_vel_err_mean_max",
                "root_pos_err_max", "root_rot_err_max",
                "root_vel_err_max", "root_angvel_err_max"):
        assert rc["replay_error"][key] < 1e-5
    if res.evaluation.status == "completed":
        assert out.T == res.evaluation.event_frame + 1


def test_pd_init_actions_follow_path():
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    src = refclip_to_coreclip(ref, start=2, end=6)
    path = TargetPath.from_clip(src, "endpoints")
    a = pd_init_actions(path, 5)
    assert a.shape == (5, 21)
    np.testing.assert_allclose(a[-1], path.sample(1.0).joint_pos_norm,
                               atol=1e-6)
    assert np.abs(a).max() <= 1.0


def test_cli_via_maps_to_track_all_path_sweep(tmp_path):
    """Legacy --via becomes track+all under the new default method."""
    from mocap.humanoid21.clip.cli import main
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    src = refclip_to_coreclip(ref, start=2, end=6)
    src_p = tmp_path / "via.npz"
    out_p = tmp_path / "out.npz"
    src.save(src_p)
    rc = main([
        "reclip", "--start", str(src_p), "--via", str(src_p),
        "--out", str(out_p), "--n-samples", "4", "--n-iters", "1",
        "--sweeps", "1", "--seed", "0"])
    assert rc == 0
    out = CoreClip.load(out_p)
    assert out.T == src.T
    meta = out.meta["reclip"]
    assert meta["objective"] == "track"
    assert meta["path"]["mode"] == "all"
    assert meta["solver"]["method"] == "path_sweep"
    assert meta["replay_error"]["joint_pos_err_mean_max"] < 1e-5


def test_cli_via_legacy_sweep_still_available(tmp_path):
    """Explicit --method sweep keeps the old frame-guide behavior."""
    from mocap.humanoid21.clip.cli import main
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    src = refclip_to_coreclip(ref, start=2, end=6)
    src_p = tmp_path / "via.npz"
    out_p = tmp_path / "legacy.npz"
    src.save(src_p)
    rc = main([
        "reclip", "--start", str(src_p), "--via", str(src_p),
        "--out", str(out_p), "--method", "sweep",
        "--n-samples", "4", "--n-iters", "1", "--sweeps", "1"])
    assert rc == 0
    out = CoreClip.load(out_p)
    assert out.T == src.T
    assert out.meta["reclip"]["method"] == "sweep"


def test_cli_clip_pass_all_and_budget_search(tmp_path):
    """--clip --path all supports pass; budget search reports attempts."""
    from mocap.humanoid21.clip.cli import main
    ref = RefClip("cmu_144_20_seg01", CLIPS)
    src = refclip_to_coreclip(ref, start=2, end=6)
    src_p = tmp_path / "src.npz"
    out_p = tmp_path / "pass_all.npz"
    src.save(src_p)
    rc = main([
        "reclip", "--clip", str(src_p), "--objective", "pass",
        "--path", "all", "--out", str(out_p),
        "--budget-search", "2,4", "--ds-max", "1.0",
        "--n-samples", "4", "--n-iters", "1", "--sweeps", "1",
        "--seeds", "0"])
    assert rc == 0
    out = CoreClip.load(out_p)
    meta = out.meta["reclip"]
    assert meta["objective"] == "pass"
    assert meta["path"]["mode"] == "all"
    assert meta["solver"]["method"] == "path_sweep"
    assert [a["budget_frames"] for a in meta["time_search"]] == [2, 4]
    assert meta["replay_error"]["root_rot_err_max"] < 1e-5
