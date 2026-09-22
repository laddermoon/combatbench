"""Tests for dumpkit.dump_analysis — the shared analysis layer behind
the viewer API and debug.py CLI.

Fixtures are hand-written npz/json files in a tmpdir (same approach as
test_viewer._write_gradsig_npz) — no training run required.

Conventions follow test_dump.py: print PASS per test, __main__ runs all.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.dumpkit.frame_access import DumpDataset
from baseline.framework.ppo.dumpkit import dump_analysis as da


# ---------------------------------------------------------------------------
# Fixture: 2 episodes × 2 agents, traj_lengths [3,3,2,2] → 10 buffer frames
# ---------------------------------------------------------------------------

def _obj_dict(d: dict) -> np.ndarray:
    arr = np.empty(1, dtype=object)
    arr[0] = d
    return arr


def _make_dump(root: Path, with_gradsig: bool = True,
               gradsig_partial: bool = False,
               early_stop: int = -1) -> Path:
    d = root / "u00010"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "update": 10, "experiment_name": "test_exp",
        "dump_source": "sentinel", "hypothesis": "h",
        "n_episodes": 2, "n_trajectories": 4, "total_frames": 10,
    }))
    (d / "traj_map.json").write_text(json.dumps([
        {"list_pos": 0, "seed": 1, "num_frames": 3, "trajectories": [
            {"traj_idx": 0, "agent_id": "a", "t_start": 0, "length": 3},
            {"traj_idx": 1, "agent_id": "b", "t_start": 0, "length": 3}]},
        {"list_pos": 1, "seed": 2, "num_frames": 2, "trajectories": [
            {"traj_idx": 2, "agent_id": "a", "t_start": 0, "length": 2},
            {"traj_idx": 3, "agent_id": "b", "t_start": 0, "length": 2}]},
    ]))
    fids = np.array([
        "ep0000:a:0", "ep0000:a:1", "ep0000:a:2",
        "ep0000:b:0", "ep0000:b:1", "ep0000:b:2",
        "ep0001:a:0", "ep0001:a:1",
        "ep0001:b:0", "ep0001:b:1",
    ], dtype=object)
    n = len(fids)
    np.savez_compressed(
        d / "buffer.npz",
        frame_id=fids,
        traj_lengths=np.array([3, 3, 2, 2], dtype=np.int64),
        log_probs=np.linspace(-2, -1, n).astype(np.float32),
        sample_weights=np.ones(n, dtype=np.float32),
        explore_factor=np.zeros(n, dtype=np.float32),
        floor_weight=np.ones(n, dtype=np.float32),
        uncertainty=np.linspace(0, 1, n).astype(np.float32),
    )
    np.savez_compressed(
        d / "gae.npz",
        advs_all=_obj_dict({"c0": np.linspace(-1, 1, n)}),
        rets_all=_obj_dict({"c0": np.linspace(0, 2, n)}),
        values_all=_obj_dict({"c0": np.linspace(0, 1, n)}),
        key_frame_mask=_obj_dict({"c0": np.ones(n, dtype=bool)}),
    )
    np.savez_compressed(
        d / "combine.npz",
        combined_adv=np.linspace(-2, 2, n).astype(np.float32),
        combined_adv_raw=np.linspace(-3, 3, n).astype(np.float32),
        adv_winsorize_clip_frac=np.float32(0.2),
        normed_advs=_obj_dict({"c0": np.linspace(-2, 2, n)}),
        key_actor_weight_frame=_obj_dict({"c0": np.ones(n)}),
        key_frame_mask=_obj_dict({"c0": np.ones(n, dtype=bool)}),
        confidences=_obj_dict({"c0": np.float32(0.9)}),
        aw_l1_sum=np.ones(n, dtype=np.float32),
        explained_variances=_obj_dict({"c0": np.float32(0.5)}),
    )
    np.savez_compressed(
        d / "update.npz",
        kl_mean=np.array(0.03), kl_max=np.array(0.06),
        early_stop_kl_mean=np.array(0.05),
        policy_loss_mean=np.array(0.4),
        grad_norm_actor_mean=np.array(3.0),
        ratio_max=np.array(2.5),
        epochs_done=np.array(2), actor_epochs_done=np.array(1),
    )
    n_steps = 4
    np.savez_compressed(
        d / "timeline.npz",
        n_epochs=np.array(2), n_batches=np.array(2),
        n_steps=np.array(n_steps),
        epoch_idx=np.array([0, 0, 1, 1], dtype=np.int64),
        mb_idx=np.array([0, 1, 0, 1], dtype=np.int64),
        actor_active=np.array([True, True, early_stop < 0, False]),
        kl=np.array([0.01, 0.02, 0.05, np.nan], dtype=np.float32),
        window_mean_kl=np.array([0.01, 0.03, 0.06, np.nan],
                                dtype=np.float32),
        ratio_max=np.array([1.1, 1.2, 3.0, np.nan], dtype=np.float32),
        ratio_mean=np.ones(n_steps, dtype=np.float32),
        ratio_min=np.ones(n_steps, dtype=np.float32) * 0.9,
        clip_frac=np.zeros(n_steps, dtype=np.float32),
        clip_frac_hi=np.zeros(n_steps, dtype=np.float32),
        clip_frac_lo=np.zeros(n_steps, dtype=np.float32),
        policy_loss=np.ones(n_steps, dtype=np.float32) * 0.4,
        actor_grad=np.array([1.0, 9.0, 2.0, np.nan], dtype=np.float32),
        dtheta_norm=np.array([0.01, 0.05, 0.02, np.nan], dtype=np.float32),
        dtheta_cos_descent=np.array([0.9, 0.3, 0.8, np.nan],
                                    dtype=np.float32),
        adv_mean=np.ones(n_steps, dtype=np.float32) * 0.1,
        adv_std=np.ones(n_steps, dtype=np.float32),
        adv_min=np.ones(n_steps, dtype=np.float32) * -1,
        adv_max=np.ones(n_steps, dtype=np.float32),
        argmax_ratio_bufidx=np.array([0, 1, 4, np.nan], dtype=np.float32),
        argmax_ratio_adv=np.ones(n_steps, dtype=np.float32),
        argmax_ratio_logr=np.ones(n_steps, dtype=np.float32) * 0.5,
        argmin_ratio_bufidx=np.array([2, 3, 9, np.nan], dtype=np.float32),
        argmin_ratio_adv=np.ones(n_steps, dtype=np.float32) * -1,
        argmin_ratio_logr=np.ones(n_steps, dtype=np.float32) * -0.5,
        n_ratio_gt2=np.zeros(n_steps, dtype=np.float32),
        n_ratio_lt05=np.zeros(n_steps, dtype=np.float32),
        floor_loss=np.zeros(n_steps, dtype=np.float32),
        mb_size=np.ones(n_steps, dtype=np.float32) * 5,
        dual_clip_frac=np.zeros(n_steps, dtype=np.float32),
        critic_loss=_obj_dict({"c0": np.ones(n_steps) * 0.2}),
        critic_grad=_obj_dict({"c0": np.ones(n_steps) * 0.1}),
        early_stop_step=np.array(early_stop, dtype=np.int64),
        target_kl=np.array(0.05, dtype=np.float32),
        clip_eps=np.array(0.2, dtype=np.float32),
    )
    np.savez_compressed(
        d / "epoch_frames.npz",
        n_epochs=np.array(2),
        **{f"ratio.{e}": np.linspace(0.8, 1.2, n).astype(np.float32)
           for e in range(2)},
        **{f"new_log_prob.{e}": np.linspace(-2, -1, n).astype(np.float32)
           for e in range(2)},
        **{f"clip_mask.{e}": np.zeros(n, dtype=bool) for e in range(2)},
        **{f"new_value.{e}.c0": np.linspace(0, 1, n).astype(np.float32)
           for e in range(2)},
        actor_stopped_epoch=np.array(-1, dtype=np.int64),
    )
    if with_gradsig:
        payload = {
            "sampled_idx": np.array([1, 4, 7], dtype=np.int64),
            "valid": np.array([True, True, False]),
            "grad_norm": np.array([2.0, 5.0, np.nan], dtype=np.float32),
            "cos": np.array([0.5, -0.8, np.nan], dtype=np.float32),
            "proj": np.array([1.0, -4.0, np.nan], dtype=np.float32),
            "w_adv": np.array([0.3, -1.2, 0.5], dtype=np.float32),
            "floor_pen": np.zeros(3, dtype=np.float32),
            "n_params": np.array(64, dtype=np.int64),
        }
        if not gradsig_partial:
            payload.update({
                "hist": np.zeros((4, 8), dtype=np.int64),
                "cos_edges": np.linspace(-1, 1, 9),
                "norm_edges": np.geomspace(0.1, 10, 5),
                "n_sampled": np.array(3), "n_valid": np.array(2),
                "n_excluded": np.array(1), "n_nonfinite": np.array(0),
                "n_frames_in_hist": np.array(2),
                "gnorm": np.array(0.5), "coherence": np.array(0.1),
                "dir_cos": np.array(0.9),
                "proj_mean": np.array(-1.5), "proj_std": np.array(2.5),
                "frac_neg": np.array(0.5),
                "norm_quantiles": np.ones(5),
                "norm_edges_derived": np.array(False),
                "hist_under": np.zeros(8, dtype=np.int64),
                "hist_over": np.zeros(8, dtype=np.int64),
                "n_under": np.array(0), "n_over": np.array(0),
            })
        np.savez_compressed(d / "gradsig.npz", **payload)
    return d


def _dd(path: Path) -> DumpDataset:
    return DumpDataset(path)


# ---------------------------------------------------------------------------

def test_inspect_overview():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td), early_stop=2))
        out = da.inspect_dump(dd)
        assert out["meta"]["update"] == 10
        assert out["capabilities"]["gradsig"] == "full"
        assert out["capabilities"]["timeline"] is True
        assert out["sampling"]["n_episodes"] == 2
        assert out["adv"]["winsorized"] is True
        assert abs(out["adv"]["winsorize_clip_frac"] - 0.2) < 1e-6
        assert out["gradsig"]["n_sampled"] == 3
        assert out["timeline"]["early_stop_step"] == 2
        assert "flags" not in out
        json.dumps(out, allow_nan=False)
        print("test_inspect_overview: PASS")


def test_inspect_degraded_no_gradsig():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td), with_gradsig=False))
        out = da.inspect_dump(dd)
        assert out["capabilities"]["gradsig"] is False
        assert "gradsig" not in out
        json.dumps(out, allow_nan=False)
        print("test_inspect_degraded_no_gradsig: PASS")


def test_timeline_overview_extended_fields():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td), early_stop=2))
        ov = da.timeline_overview(dd)
        assert ov["available"] is True
        # Previously-captured-but-unexposed fields now present
        for k in ("dtheta_norm", "dtheta_cos_descent", "adv_mean",
                  "adv_std", "adv_min", "adv_max", "argmax_ratio_bufidx",
                  "argmin_ratio_bufidx", "n_ratio_gt2", "n_ratio_lt05",
                  "floor_loss", "mb_size", "dual_clip_frac"):
            assert k in ov, k
        assert abs(ov["dtheta_norm"][1] - 0.05) < 1e-6
        # NaN serializes as None
        assert ov["dtheta_norm"][3] is None
        assert abs(ov["argmax_ratio_bufidx"][2] - 4.0) < 1e-6
        ks = {s["kind"]: s["step"] for s in ov["key_steps"]}
        assert ks["early_stop"] == 2
        assert ks["max_actor_grad"] == 1
        assert ks["max_dtheta"] == 1
        assert ks["max_ratio"] == 2
        json.dumps(ov, allow_nan=False)
        print("test_timeline_overview_extended_fields: PASS")


def test_timeline_step():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td)))
        st, s = da.timeline_step(dd, 1)
        assert st == 200
        assert s["actor_grad"] == 9.0
        assert s["argmax_ratio_bufidx"] == 1.0
        assert s["critic_loss_c0"] == 0.2
        st, _ = da.timeline_step(dd, 99)
        assert st == 404
        print("test_timeline_step: PASS")


def test_samples_sort_filter():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td)))
        st, out = da.gradsig_samples(dd, sort="abs_proj", limit=10)
        assert st == 200
        rows = out["rows"]
        # invalid frame excluded by default → 2 valid rows
        assert len(rows) == 2
        assert abs(rows[0]["proj"]) >= abs(rows[1]["proj"])
        assert rows[0]["buffer_idx"] == 4  # proj=-4.0 first
        # provenance joined: buf 4 = traj 1 frame 1 = ep0000:b:1
        assert rows[0]["episode"] == 0 and rows[0]["agent"] == "b"
        assert rows[0]["traj_idx"] == 1 and rows[0]["traj_frame"] == 1
        # meta marks sampled scope
        assert out["meta"]["scope"] == "sampled"
        # sign filter
        st, out = da.gradsig_samples(dd, sign="neg")
        assert [r["proj"] for r in out["rows"]] == [-4.0]
        # bad params → 400
        st, _ = da.gradsig_samples(dd, sort="bogus")
        assert st == 400
        st, _ = da.gradsig_samples(dd, sign="bogus")
        assert st == 400
        json.dumps(out, allow_nan=False)
        print("test_samples_sort_filter: PASS")


def test_samples_group_by_episode():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td)))
        st, out = da.gradsig_samples(dd, group_by="episode")
        assert st == 200
        eps = {e["episode"]: e for e in out["episodes"]}
        # ep0 has sampled frames buf 1 (proj+1) and buf 4 (proj-4)
        assert eps[0]["n_sampled"] == 2
        assert abs(eps[0]["sum_pos_proj"] - 1.0) < 1e-6
        assert abs(eps[0]["sum_neg_proj"] + 4.0) < 1e-6
        assert eps[0]["max_abs_proj_bufidx"] == 4
        # ep1's only sampled frame (buf 7) is invalid → filtered out
        assert 1 not in eps
        json.dumps(out, allow_nan=False)
        print("test_samples_group_by_episode: PASS")


def test_samples_partial_and_missing():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td), gradsig_partial=True))
        st, out = da.gradsig_samples(dd, sort="grad_norm")
        assert st == 200  # per-frame arrays still usable
        dd2 = _dd(_make_dump(Path(td) / "x", with_gradsig=False))
        st, out = da.gradsig_samples(dd2)
        assert st == 404 and out["available"] is False
        print("test_samples_partial_and_missing: PASS")


def test_trace_full_join():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td)))
        st, tr = da.trace_frame(dd, 4)
        assert st == 200
        assert tr["location"]["episode"] == 0
        assert tr["location"]["agent"] == "b"
        assert tr["location"]["env_frame"] == 1
        assert tr["location"]["traj_idx"] == 1
        assert tr["location"]["traj_frame"] == 1
        assert "log_probs" in tr["buffer"]
        assert "adv" in tr["gae"]["c0"]
        assert tr["combine"]["combined_adv"] is not None
        assert tr["combine"]["combined_adv_raw"] is not None
        assert tr["gradsig"]["sampled"] is True
        assert tr["gradsig"]["proj"] == -4.0
        assert tr["epoch_frames"]["0"]["ratio"] is not None
        assert tr["epoch_frames"]["0"]["new_value"]["c0"] is not None
        # buf 4 is argmax_ratio of step 2
        refs = tr["timeline_refs"]
        assert any(r["step"] == 2 and r["role"] == "argmax_ratio"
                   for r in refs)
        assert tr["links"]["trajectory"] == "trajectory/1"
        json.dumps(tr, allow_nan=False)
        print("test_trace_full_join: PASS")


def test_trace_unsampled_and_out_of_range():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td)))
        st, tr = da.trace_frame(dd, 0)  # not in sampled_idx
        assert st == 200
        assert tr["gradsig"]["sampled"] is False
        assert "not a zero" in tr["gradsig"]["reason"]
        st, tr = da.trace_frame(dd, 999)
        assert st == 404
        print("test_trace_unsampled_and_out_of_range: PASS")


def test_adv_histograms():
    with tempfile.TemporaryDirectory() as td:
        dd = _dd(_make_dump(Path(td)))
        r = da.adv_histograms(dd, bins=16)
        assert r["available"]
        labels = [s["label"] for s in r["stages"]]
        # chain order: raw gae → normed per-channel → pre-winsorize → final
        assert labels[0] == "raw adv:c0"
        assert "normed:c0" in labels
        assert labels[-2] == "combined (pre-winsorize)"
        assert labels[-1] == "combined (final)"
        fin = r["stages"][-1]
        assert len(fin["counts"]) == 16 and len(fin["edges"]) == 17
        assert sum(fin["counts"]) == fin["n_valid"]
        assert fin["min"] == -2.0 and fin["max"] == 2.0
        json.dumps(r, allow_nan=False)
        print("test_adv_histograms: PASS")


if __name__ == "__main__":
    test_inspect_overview()
    test_inspect_degraded_no_gradsig()
    test_timeline_overview_extended_fields()
    test_timeline_step()
    test_samples_sort_filter()
    test_samples_group_by_episode()
    test_samples_partial_and_missing()
    test_trace_full_join()
    test_trace_unsampled_and_out_of_range()
    test_adv_histograms()
    print("\nAll test_dump_analysis tests passed.")
