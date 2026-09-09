"""S6 ``timeline`` tool tests.

Tests cover:
- parse_events from synthetic log (resume, new_best, floor, early_stop, seed).
- parse_metric_series extraction.
- render_timeline with events and sparklines.
- timeline() public API.
- Empty/missing log handling.

Conventions follow test_s3_where.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.timeline import (
    parse_events, parse_metric_series, render_timeline, timeline,
    TimelineEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(run_dir, lines):
    """Write a train.log with the given lines."""
    (run_dir / "train.log").write_text("\n".join(lines) + "\n")


def _make_raw_stats(update, stats=None, eval_info=None):
    """Build a __RAW_STATS__ line."""
    entry = {"update": update}
    if stats:
        entry["stats"] = stats
    if eval_info:
        entry["eval_info"] = eval_info
    return f'__RAW_STATS__ {json.dumps(entry)}'


# ---------------------------------------------------------------------------
# parse_events tests
# ---------------------------------------------------------------------------

def test_parse_events_resume(tmp_path):
    """parse_events finds resume event."""
    _write_log(tmp_path, [
        "[checkpoint] resuming from update 150",
        "[update 151] something",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 1
    assert events[0].kind == "resume"
    assert events[0].update == 150
    print("test_parse_events_resume: PASS")


def test_parse_events_reset_update(tmp_path):
    """parse_events finds reset_update event."""
    _write_log(tmp_path, [
        "[checkpoint] update counter reset to 0 (was 150)",
        "[update 1] something",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 1
    assert events[0].kind == "reset_update"
    assert events[0].update == 0
    print("test_parse_events_reset_update: PASS")


def test_parse_events_new_best(tmp_path):
    """parse_events finds new_best event."""
    _write_log(tmp_path, [
        "[eval  100] max_pot=0.95  [new_best]",
        "[update 100] something",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 1
    assert events[0].kind == "best"
    assert events[0].update == 100
    print("test_parse_events_new_best: PASS")


def test_parse_events_floor(tmp_path):
    """parse_events finds floor transition event."""
    _write_log(tmp_path, [
        "[update 110] something",
        "  [floor] stepping success for 3 consecutive evals; disabling floor loss at update 110",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 1
    assert events[0].kind == "floor"
    assert events[0].update == 110
    print("test_parse_events_floor: PASS")


def test_parse_events_early_stop(tmp_path):
    """parse_events finds early_stop event."""
    _write_log(tmp_path, [
        "[update 200] something",
        "[early_stop] no improvement for 10 evals, stopping at update 200",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 1
    assert events[0].kind == "early_stop"
    assert events[0].update == 200
    print("test_parse_events_early_stop: PASS")


def test_parse_events_seed(tmp_path):
    """parse_events finds seed override event."""
    _write_log(tmp_path, [
        "[seed] overridden to 12345",
        "[update 1] something",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 1
    assert events[0].kind == "seed"
    assert "12345" in events[0].description
    print("test_parse_events_seed: PASS")


def test_parse_events_multiple(tmp_path):
    """parse_events finds multiple events, sorted by update."""
    _write_log(tmp_path, [
        "[seed] overridden to 42",
        "[checkpoint] resuming from update 150",
        "[update 151] something",
        "[eval  200] max_pot=0.95  [new_best]",
        "[update 200] something",
        "  [floor] disabling floor loss at update 250",
    ])
    events = parse_events(tmp_path)
    assert len(events) == 4
    # Sorted by update
    updates = [e.update for e in events]
    assert updates == sorted(updates)
    print("test_parse_events_multiple: PASS")


def test_parse_events_no_log(tmp_path):
    """parse_events returns [] when no log."""
    assert parse_events(tmp_path) == []
    print("test_parse_events_no_log: PASS")


def test_parse_events_empty_log(tmp_path):
    """parse_events returns [] for empty log."""
    _write_log(tmp_path, ["some line", "another line"])
    assert parse_events(tmp_path) == []
    print("test_parse_events_empty_log: PASS")


# ---------------------------------------------------------------------------
# parse_metric_series tests
# ---------------------------------------------------------------------------

def test_parse_metric_series(tmp_path):
    """parse_metric_series extracts metric time series."""
    _write_log(tmp_path, [
        _make_raw_stats(1, stats={"uncertainty": 0.2}, eval_info={"max_pot": 0.9}),
        _make_raw_stats(2, stats={"uncertainty": 0.3}, eval_info={"max_pot": 0.95}),
    ])
    updates, series = parse_metric_series(tmp_path)
    assert updates == [1, 2]
    assert series["uncertainty"] == [0.2, 0.3]
    assert series["max_pot"] == [0.9, 0.95]
    print("test_parse_metric_series: PASS")


def test_parse_metric_series_specific(tmp_path):
    """parse_metric_series with specific metric names."""
    _write_log(tmp_path, [
        _make_raw_stats(1, stats={"uncertainty": 0.2, "std_mean": 0.36}),
    ])
    updates, series = parse_metric_series(tmp_path, metric_names=["uncertainty"])
    assert "uncertainty" in series
    assert "std_mean" not in series
    print("test_parse_metric_series_specific: PASS")


def test_parse_metric_series_no_log(tmp_path):
    """parse_metric_series returns empty when no log."""
    updates, series = parse_metric_series(tmp_path)
    assert updates == []
    print("test_parse_metric_series_no_log: PASS")


def test_parse_metric_series_missing_metric(tmp_path):
    """parse_metric_series fills missing metrics with 0.0."""
    _write_log(tmp_path, [
        _make_raw_stats(1, stats={"uncertainty": 0.2}),
    ])
    updates, series = parse_metric_series(tmp_path, metric_names=["uncertainty", "missing"])
    assert series["missing"] == [0.0]
    print("test_parse_metric_series_missing_metric: PASS")


# ---------------------------------------------------------------------------
# render_timeline tests
# ---------------------------------------------------------------------------

def test_render_timeline_events_only():
    """render_timeline with only events (no metrics)."""
    events = [
        TimelineEvent(update=0, description="resume from checkpoint", kind="resume"),
        TimelineEvent(update=100, description="best checkpoint", kind="best"),
    ]
    text = render_timeline(events, [], {})
    assert "resume" in text
    assert "best" in text
    assert "u0000" in text
    assert "u0100" in text
    print("test_render_timeline_events_only: PASS")


def test_render_timeline_with_sparklines():
    """render_timeline with metric sparklines."""
    events = [TimelineEvent(update=100, description="best", kind="best")]
    updates = [1, 2, 3, 4, 5]
    series = {"uncertainty": [0.1, 0.2, 0.3, 0.4, 0.5]}
    text = render_timeline(events, updates, series, overlay=["uncertainty"])
    assert "uncertainty" in text
    # Should contain sparkline characters
    assert any(c in text for c in "▁▂▃▄▅▆▇█")
    print("test_render_timeline_with_sparklines: PASS")


def test_render_timeline_no_events():
    """render_timeline with no events but has updates."""
    updates = [1, 2, 3]
    series = {"x": [0.1, 0.2, 0.3]}
    text = render_timeline([], updates, series)
    assert "当前" in text
    print("test_render_timeline_no_events: PASS")


def test_render_timeline_empty():
    """render_timeline with no events and no updates."""
    text = render_timeline([], [], {})
    # Should not crash
    assert isinstance(text, str)
    print("test_render_timeline_empty: PASS")


# ---------------------------------------------------------------------------
# timeline() public API tests
# ---------------------------------------------------------------------------

def test_timeline_basic(tmp_path):
    """timeline() returns rendered timeline."""
    _write_log(tmp_path, [
        "[eval  100] max_pot=0.95  [new_best]",
        _make_raw_stats(100, stats={"uncertainty": 0.2}, eval_info={"max_pot": 0.95}),
    ])
    result = timeline(tmp_path)
    assert "事件时间线" in result or "当前" in result
    print("test_timeline_basic: PASS")


def test_timeline_with_overlay(tmp_path):
    """timeline() with --overlay shows specific metrics."""
    _write_log(tmp_path, [
        _make_raw_stats(1, stats={"uncertainty": 0.2, "std_mean": 0.36}),
        _make_raw_stats(2, stats={"uncertainty": 0.3, "std_mean": 0.37}),
    ])
    result = timeline(tmp_path, overlay=["uncertainty"])
    assert "uncertainty" in result
    # std_mean should NOT appear (not in overlay)
    assert "std_mean" not in result
    print("test_timeline_with_overlay: PASS")


def test_timeline_missing_run(tmp_path):
    """timeline() raises FileNotFoundError for missing run_dir."""
    with pytest.raises(FileNotFoundError):
        timeline(tmp_path / "nonexistent")
    print("test_timeline_missing_run: PASS")


def test_timeline_no_log(tmp_path):
    """timeline() handles missing log gracefully."""
    result = timeline(tmp_path)
    assert isinstance(result, str)
    print("test_timeline_no_log: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_resume(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_reset_update(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_new_best(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_floor(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_early_stop(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_seed(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_multiple(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_no_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_events_empty_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_metric_series(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_metric_series_specific(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_metric_series_no_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_metric_series_missing_metric(Path(td))
    test_render_timeline_events_only()
    test_render_timeline_with_sparklines()
    test_render_timeline_no_events()
    test_render_timeline_empty()
    with tempfile.TemporaryDirectory() as td:
        test_timeline_basic(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_timeline_with_overlay(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_timeline_missing_run(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_timeline_no_log(Path(td))
    print("\nAll S6 timeline tests passed.")
