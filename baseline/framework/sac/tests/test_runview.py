"""R3-W1: run-level viewer machinery (runview.py)."""
from __future__ import annotations

import json
import os
import time

import pytest

from baseline.framework.sac import runview
from baseline.framework.sac.tests.test_env_integration import (
    _FakeExperiment,
    _FakeRollouter,
)
from baseline.framework.sac.loop import train_sac


def _train(tmp_path, name="run"):
    run_dir = tmp_path / name
    train_sac(_FakeExperiment(), run_dir=run_dir, rollouter=_FakeRollouter())
    return run_dir


def test_run_events_table_rows_and_tick_aggregation(tmp_path):
    run_dir = _train(tmp_path)
    # Tick events emit every TICK_METRIC_INTERVAL=64 critic ticks — the
    # tiny fake run never reaches one, so append a synthetic tick event
    # to exercise the per-round aggregation path.
    ev_path = run_dir / runview.EVENTS_NAME
    with ev_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "event_type": "tick",
            "clocks": {"collection_round": 1, "critic_tick": 64,
                       "env_step": 4},
            "metrics": {"critic.loss": 2.0, "critic.q_mean": 5.0,
                        "timing.update_s": 0.5,
                        "critic.q1_loss.r": 1.0},
        }) + "\n")
        fh.write(json.dumps({
            "event_type": "tick",
            "clocks": {"collection_round": 1, "critic_tick": 128,
                       "env_step": 4},
            "metrics": {"critic.loss": 4.0, "critic.q_mean": 7.0,
                        "timing.update_s": 0.5,
                        "critic.q1_loss.r": 3.0},
        }) + "\n")
    rows = runview.run_events_table(run_dir)
    assert rows, "expected at least one round row"
    row = rows[-1]
    for k in ("round", "env_step", "critic_tick",
              "collection.episodes", "collection.env_steps",
              "collection.agent_transitions", "replay.size",
              "behavior.explore_factor"):
        assert k in row, k
    # tick metrics aggregated under tck.* (mean; timing.update_s sums)
    assert row["tck.critic.loss"] == pytest.approx(3.0)
    assert row["tck.critic.q_mean"] == pytest.approx(6.0)
    assert row["tck.timing.update_s"] == pytest.approx(1.0)
    assert row["tck.critic.q1_loss.r"] == pytest.approx(2.0)
    # round context preserved under _ctx
    assert isinstance(row.get("_ctx"), dict)


def test_run_events_table_eval_merged_sparsely(tmp_path):
    run_dir = _train(tmp_path)
    # The fake run's eval_interval never fires — synthesize an eval event.
    ev_path = run_dir / runview.EVENTS_NAME
    with ev_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "event_type": "eval",
            "clocks": {"collection_round": 1, "critic_tick": 4,
                       "env_step": 4},
            "metrics": {"eval.success": 0.5, "eval.max_pot": 0.9},
        }) + "\n")
    rows = runview.run_events_table(run_dir)
    eval_rows = [r for r in rows if "eval.success" in r]
    assert eval_rows, "expected at least one eval-merged row"
    assert eval_rows[0]["eval.success"] == pytest.approx(0.5)
    # eval stays sparse — earlier/later rounds must not inherit it
    sparse = [r for r in rows if "eval.success" not in r]
    assert len(sparse) == len(rows) - len(eval_rows)


def test_read_log_window_backward_paging(tmp_path):
    log = tmp_path / "train.log"
    lines = [f"line {i}" for i in range(500)]
    log.write_text("\n".join(lines) + "\n")
    w1 = runview.read_log_window(log, 100, None)
    assert len(w1["lines"]) == 100
    assert w1["lines"][-1] == "line 499"
    assert w1["has_more"]
    w2 = runview.read_log_window(log, 100, w1["start"])
    assert w2["lines"][-1] == "line 399"
    assert w1["lines"][0] == "line 400"
    # no gap, no overlap
    assert w2["lines"] + w1["lines"] == lines[-200:]


def test_run_status_finished(tmp_path):
    run_dir = _train(tmp_path)
    status, err = runview.run_status(run_dir)
    assert status == "finished", (status, err)


def test_run_status_unknown_for_empty(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    status, _ = runview.run_status(d)
    assert status == "unknown"


def test_dumps_list_reads_manifest_and_clocks(tmp_path):
    run_dir = _train(tmp_path)
    dump = run_dir / "debug_dumps" / "critic_tick_00000042"
    dump.mkdir(parents=True)
    (dump / "manifest.json").write_text(json.dumps({
        "schema_version": "sac_dump_v3",
        "kind": "critic_tick",
        "critic_tick": 42,
        "evidence_level": "recompute",
        "files": {"batch.pt": {"bytes": 1}},
    }))
    (dump / "clocks.json").write_text(json.dumps({
        "collection_round": 3, "env_step": 38400, "critic_tick": 42}))
    out = runview.dumps_list(run_dir)
    assert len(out) == 1
    d = out[0]
    assert d["name"] == "critic_tick_00000042"
    assert d["critic_tick"] == 42
    assert d["collection_round"] == 3
    assert d["env_step"] == 38400
    assert d["evidence_level"] == "recompute"


def test_videos_list_parses_env_step(tmp_path):
    run_dir = _train(tmp_path)
    vid = run_dir / "videos"
    vid.mkdir(exist_ok=True)
    (vid / "s00128000.mp4").write_bytes(b"x")
    (vid / "s00006400.mp4").write_bytes(b"x")
    out = runview.videos_list(run_dir)
    assert [v["env_step"] for v in out] == [128000, 6400]
    assert all(v["size"] == 1 for v in out)


def test_list_runs_and_query_index(tmp_path):
    _train(tmp_path, name="run_a")
    runs = runview.list_runs(tmp_path)
    a = next(r for r in runs if r["name"] == "run_a")
    assert a["experiment"] == "fake_sac"
    assert a["algo"] == "sac"
    assert a["status"] in ("finished", "stopped")
    assert a["env_step"] is not None
    assert "params" in a
    page = runview.query_runs_index(runs, q="run_a", size=5)
    assert page["total"] == 1
    empty = runview.query_runs_index(runs, q="zzz")
    assert empty["total"] == 0


def test_run_info_terminal_and_config(tmp_path):
    run_dir = _train(tmp_path)
    info = runview.run_info(run_dir)
    assert info["run_name"] == "run"
    assert info["experiment_name"] == "fake_sac"
    assert info["status"] in ("finished", "stopped")
    assert info["config"].get("experiment")
    assert info["dump_pending"] is False


def test_scan_experiments_finds_registry():
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[3] / "experiments_sac"
    exps = runview.scan_experiments(pkg)
    names = {e["name"] for e in exps}
    assert {"sac_standup", "sac_balance"} <= names
    for e in exps:
        assert e["tunables"], e["name"]
