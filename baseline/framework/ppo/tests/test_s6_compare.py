"""S6 ``compare`` tool tests.

Tests cover:
- parse_all_entries from synthetic log.
- extract_metrics with window averaging.
- parse_run_metrics full pipeline.
- compare_runs with two runs.
- NoiseBand load/save.
- render_compare_table format.
- compare() public API.
- Significance testing with noise band.

Conventions follow test_s3_where.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.compare import (
    parse_all_entries, extract_metrics, parse_run_metrics,
    compare_runs, render_compare_table, compare, load_noise_band,
    NoiseBand, MetricComparison, CompareReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(run_dir, entries):
    """Write a train.log with __RAW_STATS__ entries."""
    lines = []
    for e in entries:
        lines.append(f'__RAW_STATS__ {json.dumps(e)}')
    (run_dir / "train.log").write_text("\n".join(lines) + "\n")


def _make_entry(update, stats=None, eval_info=None):
    """Build a __RAW_STATS__ entry."""
    entry = {"update": update}
    if stats:
        entry["stats"] = stats
    if eval_info:
        entry["eval_info"] = eval_info
    return entry


# ---------------------------------------------------------------------------
# parse_all_entries tests
# ---------------------------------------------------------------------------

def test_parse_all_entries(tmp_path):
    """parse_all_entries reads all entries."""
    _write_log(tmp_path, [
        _make_entry(1, stats={"x": 0.5}),
        _make_entry(2, stats={"x": 0.6}),
    ])
    entries = parse_all_entries(tmp_path)
    assert len(entries) == 2
    assert entries[0]["update"] == 1
    assert entries[1]["update"] == 2
    print("test_parse_all_entries: PASS")


def test_parse_all_entries_no_log(tmp_path):
    """parse_all_entries returns [] when no log."""
    assert parse_all_entries(tmp_path) == []
    print("test_parse_all_entries_no_log: PASS")


# ---------------------------------------------------------------------------
# extract_metrics tests
# ---------------------------------------------------------------------------

def test_extract_metrics_latest():
    """extract_metrics returns latest values by default."""
    entries = [
        _make_entry(1, stats={"x": 0.5}, eval_info={"max_pot": 0.9}),
        _make_entry(2, stats={"x": 0.6}, eval_info={"max_pot": 0.95}),
    ]
    metrics = extract_metrics(entries, window=1)
    assert metrics["x"] == 0.6
    assert metrics["max_pot"] == 0.95
    print("test_extract_metrics_latest: PASS")


def test_extract_metrics_window():
    """extract_metrics averages over window."""
    entries = [
        _make_entry(1, stats={"x": 0.4}),
        _make_entry(2, stats={"x": 0.6}),
    ]
    metrics = extract_metrics(entries, window=2)
    assert abs(metrics["x"] - 0.5) < 1e-6
    print("test_extract_metrics_window: PASS")


def test_extract_metrics_empty():
    """extract_metrics returns {} for empty entries."""
    assert extract_metrics([]) == {}
    print("test_extract_metrics_empty: PASS")


def test_extract_metrics_mixed_sources():
    """extract_metrics pulls from both stats and eval_info."""
    entries = [
        _make_entry(1, stats={"uncertainty": 0.2}, eval_info={"max_pot": 0.9}),
    ]
    metrics = extract_metrics(entries)
    assert "uncertainty" in metrics
    assert "max_pot" in metrics
    print("test_extract_metrics_mixed_sources: PASS")


# ---------------------------------------------------------------------------
# parse_run_metrics tests
# ---------------------------------------------------------------------------

def test_parse_run_metrics(tmp_path):
    """parse_run_metrics reads from a run directory."""
    _write_log(tmp_path, [
        _make_entry(1, stats={"x": 0.5}),
        _make_entry(2, stats={"x": 0.6}),
    ])
    metrics = parse_run_metrics(tmp_path)
    assert metrics["x"] == 0.6
    print("test_parse_run_metrics: PASS")


def test_parse_run_metrics_window(tmp_path):
    """parse_run_metrics with window averages."""
    _write_log(tmp_path, [
        _make_entry(1, stats={"x": 0.4}),
        _make_entry(2, stats={"x": 0.6}),
    ])
    metrics = parse_run_metrics(tmp_path, window=2)
    assert abs(metrics["x"] - 0.5) < 1e-6
    print("test_parse_run_metrics_window: PASS")


# ---------------------------------------------------------------------------
# compare_runs tests
# ---------------------------------------------------------------------------

def test_compare_runs_basic(tmp_path):
    """compare_runs compares two runs."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_log(run_a, [_make_entry(1, stats={"x": 0.5})])
    _write_log(run_b, [_make_entry(1, stats={"x": 0.7})])

    report = compare_runs(run_a, run_b)
    assert report.metrics_a["x"] == 0.5
    assert report.metrics_b["x"] == 0.7
    assert len(report.comparisons) == 1
    c = report.comparisons[0]
    assert c.name == "x"
    assert abs(c.diff - 0.2) < 1e-6
    assert c.significant is None  # no noise band
    print("test_compare_runs_basic: PASS")


def test_compare_runs_with_noise_band(tmp_path):
    """compare_runs with noise band tests significance."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_log(run_a, [_make_entry(1, stats={"x": 0.5})])
    _write_log(run_b, [_make_entry(1, stats={"x": 0.7})])

    band = NoiseBand(metrics={"x": {"mean": 0.6, "std": 0.01}})
    report = compare_runs(run_a, run_b, noise_band=band)
    c = report.comparisons[0]
    # diff=0.2, 2*std=0.02 → significant
    assert c.significant is True
    print("test_compare_runs_with_noise_band: PASS")


def test_compare_runs_noise_within(tmp_path):
    """compare_runs: diff within noise band → not significant."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_log(run_a, [_make_entry(1, stats={"x": 0.5})])
    _write_log(run_b, [_make_entry(1, stats={"x": 0.51})])

    band = NoiseBand(metrics={"x": {"mean": 0.5, "std": 0.1}})
    report = compare_runs(run_a, run_b, noise_band=band)
    c = report.comparisons[0]
    # diff=0.01, 2*std=0.2 → within noise
    assert c.significant is False
    print("test_compare_runs_noise_within: PASS")


# ---------------------------------------------------------------------------
# NoiseBand tests
# ---------------------------------------------------------------------------

def test_noise_band_to_dict():
    """NoiseBand serializes to dict."""
    band = NoiseBand(metrics={"x": {"mean": 0.5, "std": 0.1}}, n_seeds=4, n_updates=50)
    d = band.to_dict()
    assert d["metrics"]["x"]["mean"] == 0.5
    assert d["n_seeds"] == 4
    print("test_noise_band_to_dict: PASS")


def test_noise_band_from_dict():
    """NoiseBand deserializes from dict."""
    d = {"metrics": {"x": {"mean": 0.5, "std": 0.1}}, "n_seeds": 4, "n_updates": 50}
    band = NoiseBand.from_dict(d)
    assert band.metrics["x"]["mean"] == 0.5
    assert band.n_seeds == 4
    print("test_noise_band_from_dict: PASS")


def test_load_noise_band(tmp_path):
    """load_noise_band reads from JSON file."""
    path = tmp_path / "noise.json"
    d = {"metrics": {"x": {"mean": 0.5, "std": 0.1}}, "n_seeds": 4, "n_updates": 50}
    path.write_text(json.dumps(d))
    band = load_noise_band(path)
    assert band.metrics["x"]["std"] == 0.1
    print("test_load_noise_band: PASS")


# ---------------------------------------------------------------------------
# render_compare_table tests
# ---------------------------------------------------------------------------

def test_render_compare_table_basic():
    """render_compare_table produces a table."""
    report = CompareReport(
        run_a=Path("/tmp/run_a"),
        run_b=Path("/tmp/run_b"),
        window=1,
        metrics_a={"x": 0.5},
        metrics_b={"x": 0.7},
        comparisons=[
            MetricComparison(name="x", value_a=0.5, value_b=0.7, diff=0.2,
                             noise_std=None, significant=None),
        ],
    )
    text = render_compare_table(report)
    assert "run_a" in text
    assert "run_b" in text
    assert "x" in text
    assert "无噪声带" in text
    print("test_render_compare_table_basic: PASS")


def test_render_compare_table_with_noise():
    """render_compare_table with noise band shows significance."""
    report = CompareReport(
        run_a=Path("/tmp/run_a"),
        run_b=Path("/tmp/run_b"),
        window=1,
        metrics_a={"x": 0.5},
        metrics_b={"x": 0.7},
        comparisons=[
            MetricComparison(name="x", value_a=0.5, value_b=0.7, diff=0.2,
                             noise_std=0.01, significant=True),
        ],
        noise_band=NoiseBand(metrics={"x": {"mean": 0.6, "std": 0.01}}, n_seeds=4, n_updates=50),
    )
    text = render_compare_table(report)
    assert "噪声基线" in text
    assert "✓ 显著" in text
    print("test_render_compare_table_with_noise: PASS")


# ---------------------------------------------------------------------------
# compare() public API tests
# ---------------------------------------------------------------------------

def test_compare_basic(tmp_path):
    """compare() returns rendered report."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_log(run_a, [_make_entry(1, stats={"x": 0.5})])
    _write_log(run_b, [_make_entry(1, stats={"x": 0.7})])

    result = compare(run_a, run_b)
    assert "run_a" in result
    assert "x" in result
    print("test_compare_basic: PASS")


def test_compare_with_noise_band(tmp_path):
    """compare() with noise band file."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_log(run_a, [_make_entry(1, stats={"x": 0.5})])
    _write_log(run_b, [_make_entry(1, stats={"x": 0.7})])

    noise_path = tmp_path / "noise.json"
    noise_path.write_text(json.dumps(
        {"metrics": {"x": {"mean": 0.6, "std": 0.01}}, "n_seeds": 4, "n_updates": 50}
    ))

    result = compare(run_a, run_b, noise_band_path=noise_path)
    assert "✓ 显著" in result
    print("test_compare_with_noise_band: PASS")


def test_compare_missing_run(tmp_path):
    """compare() raises FileNotFoundError for missing run."""
    with pytest.raises(FileNotFoundError):
        compare(tmp_path / "nonexistent", tmp_path)
    print("test_compare_missing_run: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_parse_all_entries(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_all_entries_no_log(Path(td))
    test_extract_metrics_latest()
    test_extract_metrics_window()
    test_extract_metrics_empty()
    test_extract_metrics_mixed_sources()
    with tempfile.TemporaryDirectory() as td:
        test_parse_run_metrics(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_run_metrics_window(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_compare_runs_basic(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_compare_runs_with_noise_band(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_compare_runs_noise_within(Path(td))
    test_noise_band_to_dict()
    test_noise_band_from_dict()
    with tempfile.TemporaryDirectory() as td:
        test_load_noise_band(Path(td))
    test_render_compare_table_basic()
    test_render_compare_table_with_noise()
    with tempfile.TemporaryDirectory() as td:
        test_compare_basic(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_compare_with_noise_band(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_compare_missing_run(Path(td))
    print("\nAll S6 compare tests passed.")
