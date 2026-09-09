"""S6 ``intervene-check`` tool tests.

Tests cover:
- build_context from synthetic run_dir.
- find_latest_snapshot auto-discovery.
- intervene_check with mock log data.
- render table format.
- --knob filter.
- Missing log/snapshot handling.

Conventions follow test_s3_where.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.intervene import (
    intervene_check, build_context, find_latest_snapshot,
)
from baseline.framework.ppo.debug.knobs import KnobContext, KnobResult, render_knob_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(run_dir, lines):
    """Write a train.log with the given lines."""
    (run_dir / "train.log").write_text("\n".join(lines) + "\n")


def _write_config(run_dir, config=None):
    """Write a config.json."""
    config = config or {"experiment": {"name": "test"}}
    (run_dir / "config.json").write_text(json.dumps(config))


def _make_run_dir(tmp_path, with_log=True, with_config=True):
    """Create a minimal run directory with log + config."""
    if with_config:
        _write_config(tmp_path)
    if with_log:
        _write_log(tmp_path, [
            "[update 1]",
            '__RAW_STATS__ {"update": 1, "stats": {"std_mean": 0.36, "eff_std_mean": 0.18}}',
            "  [GradDiag] floor_active=0.5",
            "[update 2]",
            '__RAW_STATS__ {"update": 2, "stats": {"std_mean": 0.36, "eff_std_mean": 0.18}, '
            '"buffer_stats": {"per_channel": {"r_a": {"actor_weight_min": 0.0, "actor_weight_max": 1.0}}}}',
            "  [GradDiag] floor_active=0.0",
        ])


# ---------------------------------------------------------------------------
# find_latest_snapshot tests
# ---------------------------------------------------------------------------

def test_find_latest_snapshot_none(tmp_path):
    """find_latest_snapshot returns None when no snapshots."""
    assert find_latest_snapshot(tmp_path) is None
    print("test_find_latest_snapshot_none: PASS")


def test_find_latest_snapshot_found(tmp_path):
    """find_latest_snapshot finds the latest snapshot."""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "u00001").mkdir()
    (debug_dir / "u00050").mkdir()
    (debug_dir / "u00010").mkdir()
    result = find_latest_snapshot(tmp_path)
    assert result is not None
    assert result.name == "u00050"
    print("test_find_latest_snapshot_found: PASS")


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------

def test_build_context_basic(tmp_path):
    """build_context loads config + log entry."""
    _make_run_dir(tmp_path)
    ctx = build_context(tmp_path, load_experiment=False)
    assert ctx.run_dir == tmp_path.resolve()
    assert ctx.config.get("experiment", {}).get("name") == "test"
    assert ctx.log_entry is not None
    assert ctx.log_entry["update"] == 2
    assert ctx.log_stats is not None
    assert "std_mean" in ctx.log_stats
    print("test_build_context_basic: PASS")


def test_build_context_target_update(tmp_path):
    """build_context with target update finds the right entry."""
    _make_run_dir(tmp_path)
    ctx = build_context(tmp_path, update=1, load_experiment=False)
    assert ctx.log_entry["update"] == 1
    print("test_build_context_target_update: PASS")


def test_build_context_no_log(tmp_path):
    """build_context with no log returns None for log_entry."""
    _write_config(tmp_path)
    ctx = build_context(tmp_path, load_experiment=False)
    assert ctx.log_entry is None
    assert ctx.log_stats is None
    print("test_build_context_no_log: PASS")


def test_build_context_with_snapshot(tmp_path):
    """build_context with explicit snapshot_dir."""
    _make_run_dir(tmp_path)
    snap = tmp_path / "debug" / "u00001"
    snap.mkdir(parents=True)
    ctx = build_context(tmp_path, snapshot_dir=snap, load_experiment=False)
    assert ctx.snapshot_dir == snap.resolve()
    print("test_build_context_with_snapshot: PASS")


# ---------------------------------------------------------------------------
# intervene_check tests
# ---------------------------------------------------------------------------

def test_intervene_check_basic(tmp_path):
    """intervene_check returns a rendered table."""
    _make_run_dir(tmp_path)
    result = intervene_check(tmp_path, load_experiment=False) if "load_experiment" in intervene_check.__code__.co_varnames else None
    # intervene_check doesn't take load_experiment; it always tries to load
    # but gracefully handles failure
    result = intervene_check(tmp_path)
    assert "干预验证" in result
    assert "explore_factor" in result
    assert "uncertainty_floor" in result
    print("test_intervene_check_basic: PASS")


def test_intervene_check_with_update(tmp_path):
    """intervene_check with --at targets specific update."""
    _make_run_dir(tmp_path)
    result = intervene_check(tmp_path, update=2)
    assert "update 2" in result
    print("test_intervene_check_with_update: PASS")


def test_intervene_check_knob_filter(tmp_path):
    """intervene_check with --knob filters to one knob."""
    _make_run_dir(tmp_path)
    result = intervene_check(tmp_path, knob_name="explore_factor")
    assert "explore_factor" in result
    assert "uncertainty_floor" not in result
    print("test_intervene_check_knob_filter: PASS")


def test_intervene_check_unknown_knob(tmp_path):
    """intervene_check with unknown knob name returns error message."""
    _make_run_dir(tmp_path)
    result = intervene_check(tmp_path, knob_name="nonexistent")
    assert "not found" in result
    print("test_intervene_check_unknown_knob: PASS")


def test_intervene_check_missing_run(tmp_path):
    """intervene_check raises FileNotFoundError for missing run_dir."""
    with pytest.raises(FileNotFoundError):
        intervene_check(tmp_path / "nonexistent")
    print("test_intervene_check_missing_run: PASS")


def test_intervene_check_no_log(tmp_path):
    """intervene_check handles missing log gracefully."""
    _write_config(tmp_path)
    result = intervene_check(tmp_path)
    # Should still render, with knobs showing "—" for missing data
    assert "干预验证" in result
    print("test_intervene_check_no_log: PASS")


def test_intervene_check_snapshot_knob_skipped(tmp_path):
    """intervene_check skips observer knob when no snapshot."""
    _make_run_dir(tmp_path)
    result = intervene_check(tmp_path)
    # observer knob should show "需快照"
    assert "observer" in result
    assert "需快照" in result
    print("test_intervene_check_snapshot_knob_skipped: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_find_latest_snapshot_none(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_find_latest_snapshot_found(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_context_basic(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_context_target_update(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_context_no_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_context_with_snapshot(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_basic(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_with_update(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_knob_filter(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_unknown_knob(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_missing_run(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_no_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_intervene_check_snapshot_knob_skipped(Path(td))
    print("\nAll S6 intervene tests passed.")
