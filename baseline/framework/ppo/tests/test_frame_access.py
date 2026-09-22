"""Tests for dumpkit.frame_access — the unified dump read layer.

Fixtures are hand-written npz/json files in a tmpdir (same approach as
test_dump_analysis) — no training run required.

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


# ---------------------------------------------------------------------------
# Fixture: 2 episodes × 2 agents, traj_lengths [3,3,2,2] → 10 buffer frames
# ---------------------------------------------------------------------------

def _obj_dict(d: dict) -> np.ndarray:
    arr = np.empty(1, dtype=object)
    arr[0] = d
    return arr


def _make_dump(root: Path, with_episodes: bool = True) -> Path:
    d = root / "u00010"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "update": 10, "experiment_name": "test_exp",
        "n_episodes": 2, "n_trajectories": 4, "total_frames": 10,
    }))
    (d / "traj_map.json").write_text(json.dumps([
        {"list_pos": 0, "seed": 1, "num_frames": 3, "trajectories": [
            {"traj_idx": 0, "agent_id": "robot_a", "t_start": 0,
             "length": 3},
            {"traj_idx": 1, "agent_id": "robot_b", "t_start": 0,
             "length": 3}]},
        {"list_pos": 1, "seed": 2, "num_frames": 2, "trajectories": [
            {"traj_idx": 2, "agent_id": "robot_a", "t_start": 0,
             "length": 2},
            {"traj_idx": 3, "agent_id": "robot_b", "t_start": 0,
             "length": 2}]},
    ]))
    fids = np.array([
        "ep0000:robot_a:0", "ep0000:robot_a:1", "ep0000:robot_a:2",
        "ep0000:robot_b:0", "ep0000:robot_b:1", "ep0000:robot_b:2",
        "ep0001:robot_a:0", "ep0001:robot_a:1",
        "ep0001:robot_b:0", "ep0001:robot_b:1",
    ], dtype=object)
    n = len(fids)
    np.savez_compressed(
        d / "trajectories.npz",
        n_trajectories=np.array(4),
        traj_lengths=np.array([3, 3, 2, 2], dtype=np.int64),
        channel_names=np.array(["c0", "c1"], dtype=object),
        **{
            "reward.c0": np.linspace(0, 1, n).astype(np.float32),
            "reward.c1": np.linspace(1, 2, n).astype(np.float32),
            "actor_weight.c0": np.ones(n, dtype=np.float32),
            "actor_weight.c1": np.ones(n, dtype=np.float32) * 2,
            "is_terminated.c0": np.zeros(4, dtype=bool),
            "is_terminated.c1": np.array([0, 0, 1, 0], dtype=bool),
            "floor_weight": np.ones(n, dtype=np.float32),
            "explore_factor": np.zeros(n, dtype=np.float32),
            "importance": np.ones(4, dtype=np.float32),
            "frame_id": fids,
        },
    )
    np.savez_compressed(
        d / "buffer.npz",
        frame_id=fids,
        traj_lengths=np.array([3, 3, 2, 2], dtype=np.int64),
        log_probs=np.linspace(-2, -1, n).astype(np.float32),
        sample_weights=np.ones(n, dtype=np.float32),
        explore_factor=np.zeros(n, dtype=np.float32),
        floor_weight=np.ones(n, dtype=np.float32),
        uncertainty=np.linspace(0, 1, n).astype(np.float32),
        obs=np.zeros((n, 4), dtype=np.float32),
        actions=np.zeros((n, 2), dtype=np.float32),
    )
    np.savez_compressed(
        d / "gae.npz",
        advs_all=_obj_dict({
            "c0": np.linspace(-1, 1, n), "c1": np.linspace(-2, 2, n)}),
        rets_all=_obj_dict({"c0": np.linspace(0, 2, n)}),
        values_all=_obj_dict({"c0": np.linspace(0, 1, n)}),
        key_frame_mask=_obj_dict({"c0": np.ones(n, dtype=bool)}),
        bootstrap_values=_obj_dict({"c0": np.array([0.1, 0.2, 0.3, 0.4])}),
        bootstrap_indices=np.array([3]),
    )
    np.savez_compressed(
        d / "combine.npz",
        combined_adv=np.linspace(-2, 2, n).astype(np.float32),
        normed_advs=_obj_dict({
            "c0": np.linspace(-2, 2, n), "c1": np.ones(n)}),
        aw_normed=_obj_dict({
            "c0": np.full(n, 0.25), "c1": np.full(n, 0.5)}),
        confidences=_obj_dict({"c0": np.float32(0.8), "c1": np.float32(0.5)}),
        aw_l1_sum=np.ones(n, dtype=np.float32),
        key_actor_weight_frame=_obj_dict({"c0": np.ones(n)}),
    )
    np.savez_compressed(
        d / "epoch_frames.npz",
        n_epochs=np.array(2),
        **{f"ratio.{e}": np.linspace(0.8, 1.2, n).astype(np.float32)
           for e in range(2)},
        **{f"new_value.{e}.c0": np.linspace(0, 1, n).astype(np.float32)
           for e in range(2)},
    )
    if with_episodes:
        # ep space: ep0 has 3 frames, ep1 has 2 → flat offsets [0,3,5]
        m = 5
        np.savez_compressed(
            d / "episodes.npz",
            n_episodes=np.array(2),
            episode_indices=np.array([0, 1]),
            num_frames=np.array([3, 2]),
            base_seeds=np.array([1, 2]),
            episode_frame_offsets=np.array([0, 3, 5], dtype=np.int64),
            **{
                "obs.robot_a": np.zeros((m, 4), dtype=np.float32),
                "obs.robot_b": np.ones((m, 4), dtype=np.float32),
                "actions.robot_a": np.zeros((m, 2), dtype=np.float32),
                "actions.robot_b": np.ones((m, 2), dtype=np.float32),
                "explore_factors.robot_a": np.zeros(m, dtype=np.float32),
                # observer flat layout: per-ep-frame arrays
                "observer_outputs.height_a.h":
                    np.arange(m, dtype=np.float64) * 0.01,
                "observer_outputs.height_b.h":
                    np.arange(m, dtype=np.float64) * 0.02,
            },
        )
    return d


# ---------------------------------------------------------------------------

def test_lazy_column_loading():
    """columns load on demand: untouched members stay unloaded."""
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        # only traj_lengths + keys touched so far
        f = ds.frames
        assert f.n_rows == 10
        a = f["adv.c0"]
        assert a.shape == (10,)
        src = ds.npz("gae")
        assert "advs_all" in src._cols
        assert "values_all" not in src._cols  # untouched → not loaded
        print("test_lazy_column_loading: PASS")


def test_verbatim_and_template_columns():
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        f = ds.frames
        # verbatim member passthrough
        assert np.allclose(f["reward.c0"], np.linspace(0, 1, 10))
        assert np.allclose(f["actor_weight.c1"], 2.0)
        assert str(f["frame_id"][0]) == "ep0000:robot_a:0"
        assert np.allclose(f["combined_adv"], np.linspace(-2, 2, 10))
        assert np.allclose(f["log_probs"], np.linspace(-2, -1, 10))
        assert np.allclose(f["epoch.0.ratio"], np.linspace(0.8, 1.2, 10))
        assert np.allclose(f["epoch.1.value.c0"], np.linspace(0, 1, 10))
        # normalized dict templates
        assert np.allclose(f["adv.c1"], np.linspace(-2, 2, 10))
        assert np.allclose(f["value.c0"], np.linspace(0, 1, 10))
        assert np.allclose(f["key_frame_mask.c0"], 1)
        assert np.allclose(f["normed_adv.c0"], np.linspace(-2, 2, 10))
        assert np.allclose(f["aw_normed.c1"], 0.5)
        # missing → None via col(), KeyError via []
        assert f.col("adv.nonexistent") is None
        assert f.col("bogus") is None
        print("test_verbatim_and_template_columns: PASS")


def test_contrib_derived():
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        f = ds.frames
        # contrib.c0 = aw_normed(0.25) × conf(0.8) × normed_adv
        exp = 0.25 * 0.8 * np.linspace(-2, 2, 10)
        assert np.allclose(f["contrib.c0"], exp)
        assert np.allclose(f["contrib.c1"], 0.5 * 0.5 * np.ones(10))
        print("test_contrib_derived: PASS")


def test_observer_join():
    """observer.<base>.<field> resolves per-traj agent suffix + gathers."""
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        f = ds.frames
        # traj 0 (robot_a, ep0 t0..2) → height_a.h[0:3] = 0,0.01,0.02
        # traj 1 (robot_b, ep0 t0..2) → height_b.h[0:3] = 0,0.02,0.04
        # traj 2 (robot_a, ep1 t0..1) → height_a.h[3:5] = 0.03,0.04
        # traj 3 (robot_b, ep1 t0..1) → height_b.h[3:5] = 0.06,0.08
        h = f["observer.height.h"]
        exp = np.array([0, .01, .02, 0, .02, .04, .03, .04, .06, .08])
        assert np.allclose(h, exp), h
        # suffixed name = agent filter: only that agent's trajs get data
        ha = f["observer.height_a.h"]
        assert np.isnan(ha[[3, 4, 5, 8, 9]]).all()
        assert np.allclose(ha[[0, 1, 2, 6, 7]], [0, .01, .02, .03, .04])
        # columns advertise canonical (unsuffixed) names only
        assert "observer.height.h" in f.columns
        assert "observer.height_a.h" not in f.columns
        print("test_observer_join: PASS")


def test_traj_slice_and_view():
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        f = ds.frames
        assert f.traj_slice(1) == (3, 6)
        t = f.traj(2)
        assert len(t) == 2
        assert np.allclose(t["adv.c0"], np.linspace(-1, 1, 10)[6:8])
        # mask view composes
        m = f["adv.c0"] > 0
        sub = f[m]
        assert len(sub) == int(m.sum())
        assert np.all(sub["adv.c0"] > 0)
        print("test_traj_slice_and_view: PASS")


def test_provenance():
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        assert ds.frame_traj(4) == (1, 1)
        loc = ds.frame_provenance(4)
        assert loc["mapped"] and loc["episode"] == 0
        assert loc["agent"] == "robot_b" and loc["env_frame"] == 1
        assert loc["traj_idx"] == 1 and loc["traj_frame"] == 1
        p = ds.provenance(2)
        assert p["ep_pos"] == 1 and p["agent_id"] == "robot_a"
        tr = ds.trajs
        assert tr["ep_pos"].tolist() == [0, 0, 1, 1]
        assert tr["agent_id"].tolist() == ["robot_a", "robot_b",
                                           "robot_a", "robot_b"]
        assert tr["is_terminated.c1"].tolist() == [0, 0, 1, 0]
        assert np.allclose(tr["bootstrap_value.c0"], [0.1, 0.2, 0.3, 0.4])
        print("test_provenance: PASS")


def test_episode_view():
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        ev = ds.episodes[0]
        assert ev.n_frames == 3 and ev.seed == 1
        assert ev.col("obs.robot_b").shape == (3, 4)
        assert np.allclose(ev.col("obs.robot_b"), 1)
        assert np.allclose(
            ev.col("observer.height_a.h"), [0, 0.01, 0.02])
        assert np.allclose(
            ev.col("observer.height_b.h", frame=2), 0.04)
        assert len(ev.trajectories) == 2
        ep_tab = ds.episodes
        assert ep_tab.n_rows == 2
        assert ep_tab["num_frames"].tolist() == [3, 2]
        assert ep_tab["seed"].tolist() == [1, 2]
        print("test_episode_view: PASS")


def test_window_reduce():
    """forward-looking reduce respects traj boundaries."""
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        f = ds.frames
        h = f["observer.height.h"]  # values per fixture
        w = f.window_reduce("observer.height.h", k=2, how="max")
        # traj0 seg [0:3]: max of [0,.01]=.01, [.01,.02]=.02, [.02]=.02
        assert np.allclose(w[:3], [0.01, 0.02, 0.02])
        # boundary: traj1 last frame must NOT see traj2's data
        assert w[5] == h[5]  # last frame of traj1 → only itself
        assert np.allclose(w[6:8], [0.04, 0.04])  # traj2
        print("test_window_reduce: PASS")


def test_registered_column():
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td)))
        f = ds.frames
        f.register("standing",
                   lambda t: t["observer.height.h"] >= 0.02)
        s = f["standing"]
        assert s.dtype != object and len(s) == 10
        assert s.sum() == 7  # h>=0.02: idx 2(a),5? → count via exp
        exp = np.array([0, .01, .02, 0, .02, .04, .03, .04, .06, .08])
        assert s.sum() == int((exp >= 0.02).sum())
        # works under mask
        m = f["reward.c0"] > 0.5
        assert len(f[m]["standing"]) == int(m.sum())
        print("test_registered_column: PASS")


def test_missing_files():
    """Dump without episodes.npz: frame space still works, observer → None."""
    with tempfile.TemporaryDirectory() as td:
        ds = DumpDataset(_make_dump(Path(td), with_episodes=False))
        f = ds.frames
        assert f.n_rows == 10
        assert f.col("observer.height.h") is None
        assert np.allclose(f["adv.c0"], np.linspace(-1, 1, 10))
        # episodes table falls back to traj_map
        assert ds.episodes.n_rows == 2
        assert ds.episodes["num_frames"].tolist() == [3, 2]
        print("test_missing_files: PASS")


def test_timeline_and_gradsig_tables():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = _make_dump(root)
        np.savez_compressed(
            d / "timeline.npz",
            n_steps=np.array(3),
            kl_mean=np.array([0.01, 0.02, 0.03], dtype=np.float32),
            critic_loss=_obj_dict({"c0": np.ones(3) * 0.2}),
        )
        np.savez_compressed(
            d / "gradsig.npz",
            sampled_idx=np.array([1, 4, 9], dtype=np.int64),
            valid=np.array([True, True, False]),
            proj=np.array([1.0, -4.0, np.nan], dtype=np.float32),
        )
        ds = DumpDataset(d)
        tl = ds.timeline
        assert tl.n_rows == 3
        assert np.allclose(tl["kl_mean"], [0.01, 0.02, 0.03])
        assert np.allclose(tl["critic_loss.c0"], 0.2)
        g = ds.gradsig
        assert g.n_rows == 3
        assert g["traj_idx"].tolist() == [0, 1, 3]
        assert g["episode"].tolist() == [0, 0, 1]
        assert g["agent"].tolist() == ["robot_a", "robot_b", "robot_b"]
        assert g["frame_id"].tolist()[0] == "ep0000:robot_a:1"
        print("test_timeline_and_gradsig_tables: PASS")


def test_update_scalars_and_capabilities():
    with tempfile.TemporaryDirectory() as td:
        d = _make_dump(Path(td))
        np.savez_compressed(
            d / "update.npz", kl_mean=np.array(0.03),
            **{"ev.c0": np.array(0.9)})
        ds = DumpDataset(d)
        assert ds.update.scalar("kl_mean") == 0.03
        assert ds.update.scalar("ev.c0") == 0.9
        caps = ds.capabilities()
        assert caps["episodes"] and caps["trajectories"]
        assert caps["gradsig"] is False  # no gradsig.npz in fixture
        assert ds.channel_names == ["c0", "c1"]
        assert ds.agent_ids == ["robot_a", "robot_b"]
        assert "height_a" in ds.observer_keys
        print("test_update_scalars_and_capabilities: PASS")


def test_jsonable():
    v = DumpDataset.to_jsonable(np.array([1.0, np.nan, np.inf]))
    assert v == [1.0, None, None]
    assert DumpDataset.to_jsonable(np.float32(0.5)) == 0.5
    assert DumpDataset.to_jsonable(np.int64(3)) == 3
    assert DumpDataset.to_jsonable(np.bool_(True)) is True
    assert DumpDataset.to_jsonable(
        np.array(["a", "b"], dtype=object)) == ["a", "b"]
    print("test_jsonable: PASS")


if __name__ == "__main__":
    test_lazy_column_loading()
    test_verbatim_and_template_columns()
    test_contrib_derived()
    test_observer_join()
    test_traj_slice_and_view()
    test_provenance()
    test_episode_view()
    test_window_reduce()
    test_registered_column()
    test_missing_files()
    test_timeline_and_gradsig_tables()
    test_update_scalars_and_capabilities()
    test_jsonable()
    print("\nAll test_frame_access tests passed.")
