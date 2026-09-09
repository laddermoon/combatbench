"""S3 ``attribute`` tool tests.

Tests cover:
- parse_log_stats finds the right update.
- parse_log_stats_window returns last N entries.
- build_report extracts influence_share, aw_normed, dead_frame_ratio.
- build_report with window averages across entries.
- render_report produces bar chart + dead frame ratio + grad dims.
- render_report with by_action_dim shows top/bottom 3.
- Missing log file raises FileNotFoundError.

Conventions follow test_s2_sink.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.attribute import (
    AttributeReport, build_report, render_report, attribute,
    parse_log_stats, parse_log_stats_window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(run_dir, entries):
    """Write a train.log with __RAW_STATS__ entries."""
    lines = []
    for e in entries:
        stats = e.get("stats", {})
        line = f"[update {e['update']}] __RAW_STATS__ {json.dumps({'update': e['update'], 'stats': stats})}\n"
        lines.append(line)
    (run_dir / "train.log").write_text("".join(lines))


def _make_stats(update, *, influence=None, aw=None, dead=0.0, grads=None):
    """Build a stats dict with S0 aggregates."""
    stats = {
        "dead_frame_ratio": dead,
    }
    if influence:
        for k, v in influence.items():
            stats[f"influence_share_{k}"] = v
    if aw:
        for k, v in aw.items():
            stats[f"aw_normed_{k}"] = v
    if grads:
        for i, g in enumerate(grads):
            stats[f"grad_dim_{i:02d}"] = g
    return {"update": update, "stats": stats}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_log_stats_finds_update(tmp_path):
    """parse_log_stats finds the entry for a specific update."""
    _write_log(tmp_path, [
        _make_stats(1, influence={"r_a": 0.5}),
        _make_stats(2, influence={"r_a": 0.3}),
    ])
    stats = parse_log_stats(tmp_path, 2)
    assert stats is not None
    assert stats["influence_share_r_a"] == 0.3
    print("test_parse_log_stats_finds_update: PASS")


def test_parse_log_stats_latest(tmp_path):
    """parse_log_stats returns latest when update is None."""
    _write_log(tmp_path, [
        _make_stats(1, influence={"r_a": 0.5}),
        _make_stats(2, influence={"r_a": 0.3}),
    ])
    stats = parse_log_stats(tmp_path)
    assert stats is not None
    assert stats["influence_share_r_a"] == 0.3
    print("test_parse_log_stats_latest: PASS")


def test_parse_log_stats_no_log(tmp_path):
    """parse_log_stats returns None when no log exists."""
    assert parse_log_stats(tmp_path) is None
    print("test_parse_log_stats_no_log: PASS")


def test_parse_log_stats_window(tmp_path):
    """parse_log_stats_window returns last N entries."""
    _write_log(tmp_path, [
        _make_stats(i, influence={"r_a": i * 0.1}) for i in range(5)
    ])
    entries = parse_log_stats_window(tmp_path, 3)
    assert len(entries) == 3
    assert entries[0]["influence_share_r_a"] == 0.2
    assert entries[2]["influence_share_r_a"] == 0.4
    print("test_parse_log_stats_window: PASS")


def test_build_report_single_update(tmp_path):
    """build_report extracts all fields from one update."""
    _write_log(tmp_path, [
        _make_stats(1, influence={"r_a": 0.6, "r_b": 0.4},
                    aw={"r_a": 0.5, "r_b": 0.5}, dead=0.12,
                    grads=[0.1, 0.05, 0.001]),
    ])
    report = build_report(tmp_path, update=1)
    assert report.update == 1
    assert report.influence_shares["r_a"] == 0.6
    assert report.influence_shares["r_b"] == 0.4
    assert report.aw_normed["r_a"] == 0.5
    assert report.dead_frame_ratio == 0.12
    assert report.action_dim_grad_norms is not None
    assert len(report.action_dim_grad_norms) == 3
    print("test_build_report_single_update: PASS")


def test_build_report_window_average(tmp_path):
    """build_report averages across window."""
    _write_log(tmp_path, [
        _make_stats(1, influence={"r_a": 0.6}, dead=0.10),
        _make_stats(2, influence={"r_a": 0.4}, dead=0.20),
    ])
    report = build_report(tmp_path, window=2)
    assert report.influence_shares["r_a"] == pytest.approx(0.5)  # average
    assert report.dead_frame_ratio == pytest.approx(0.15)  # average
    print("test_build_report_window_average: PASS")


def test_build_report_missing_log(tmp_path):
    """build_report raises FileNotFoundError when no log."""
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path, update=1)
    print("test_build_report_missing_log: PASS")


def test_render_report_bar_chart(tmp_path):
    """render_report produces a bar chart with percentages."""
    report = AttributeReport(
        update=1,
        influence_shares={"r_a": 0.6, "r_b": 0.4},
        aw_normed={"r_a": 0.5, "r_b": 0.5},
        dead_frame_ratio=0.12,
        action_dim_grad_norms=None,
        n_channels=2,
    )
    text = render_report(report)
    assert "60.0%" in text or "60.0" in text
    assert "40.0%" in text or "40.0" in text
    assert "12.0%" in text or "12.0" in text  # dead_frame_ratio
    print("test_render_report_bar_chart: PASS")


def test_render_report_by_action_dim(tmp_path):
    """render_report with by_action_dim shows top/bottom 3."""
    report = AttributeReport(
        update=1,
        influence_shares={"r_a": 1.0},
        aw_normed={"r_a": 1.0},
        dead_frame_ratio=0.0,
        action_dim_grad_norms=np.array([0.1, 0.05, 0.001, 0.08, 0.02, 0.0005]),
        n_channels=1,
    )
    text = render_report(report, by_action_dim=True)
    assert "top" in text.lower() or "最大" in text
    assert "bottom" in text.lower() or "最小" in text
    print("test_render_report_by_action_dim: PASS")


def test_attribute_full(tmp_path):
    """Full attribute() call returns rendered string."""
    _write_log(tmp_path, [
        _make_stats(1, influence={"r_a": 0.6, "r_b": 0.4},
                    aw={"r_a": 0.5, "r_b": 0.5}, dead=0.12,
                    grads=[0.1, 0.05, 0.001]),
    ])
    text = attribute(tmp_path, update=1, by_action_dim=True)
    assert "update 1" in text
    assert "r_a" in text
    print("test_attribute_full: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_parse_log_stats_finds_update(td)
    with tempfile.TemporaryDirectory() as td:
        test_parse_log_stats_latest(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_log_stats_no_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_log_stats_window(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_report_single_update(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_report_window_average(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_build_report_missing_log(Path(td))
    test_render_report_bar_chart(Path("/tmp"))
    test_render_report_by_action_dim(Path("/tmp"))
    with tempfile.TemporaryDirectory() as td:
        test_attribute_full(Path(td))
    print("\nAll S3 attribute tests passed.")
