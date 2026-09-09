"""S6 ``noise`` tool tests.

Tests cover:
- compute_noise_band from synthetic run dirs.
- save_noise_band / load round-trip.
- render_noise_summary format.
- NoiseBand integration with compare.
- _pid_alive helper.
- launch_training_run is mocked (not actually launched).

Conventions follow test_s3_where.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.noise import (
    compute_noise_band, save_noise_band, render_noise_summary,
    _pid_alive, NoiseBand,
)
from baseline.framework.ppo.debug.compare import parse_run_metrics


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
    entry = {"update": update}
    if stats:
        entry["stats"] = stats
    if eval_info:
        entry["eval_info"] = eval_info
    return entry


def _make_run_dir(tmp_path, name, stats_values):
    """Create a run dir with entries having the given stats values."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    entries = [_make_entry(i, stats={"x": v}) for i, v in enumerate(stats_values, 1)]
    _write_log(run_dir, entries)
    return run_dir


# ---------------------------------------------------------------------------
# compute_noise_band tests
# ---------------------------------------------------------------------------

def test_compute_noise_band_basic(tmp_path):
    """compute_noise_band computes mean ± std across runs."""
    rd1 = _make_run_dir(tmp_path, "run1", [0.5, 0.6, 0.55])
    rd2 = _make_run_dir(tmp_path, "run2", [0.7, 0.8, 0.75])
    rd3 = _make_run_dir(tmp_path, "run3", [0.6, 0.7, 0.65])

    band = compute_noise_band([rd1, rd2, rd3], window=3)
    assert band.n_seeds == 3
    assert "x" in band.metrics
    # mean of (0.55, 0.75, 0.65) = 0.65
    assert abs(band.metrics["x"]["mean"] - 0.65) < 1e-6
    # std should be > 0
    assert band.metrics["x"]["std"] > 0
    print("test_compute_noise_band_basic: PASS")


def test_compute_noise_band_single_run(tmp_path):
    """compute_noise_band with one run returns empty band (need >= 2)."""
    rd1 = _make_run_dir(tmp_path, "run1", [0.5, 0.6])
    band = compute_noise_band([rd1], window=2)
    # With only 1 run, std can't be computed
    assert band.n_seeds == 1
    # x might not be in band because len(vals) < 2
    assert "x" not in band.metrics
    print("test_compute_noise_band_single_run: PASS")


def test_compute_noise_band_empty():
    """compute_noise_band with no runs returns empty band."""
    band = compute_noise_band([])
    assert band.n_seeds == 0
    assert band.metrics == {}
    print("test_compute_noise_band_empty: PASS")


def test_compute_noise_band_window(tmp_path):
    """compute_noise_band respects window parameter."""
    rd1 = _make_run_dir(tmp_path, "run1", [0.1, 0.2, 0.5, 0.6])
    rd2 = _make_run_dir(tmp_path, "run2", [0.1, 0.2, 0.7, 0.8])

    # window=2 → average last 2 entries
    band = compute_noise_band([rd1, rd2], window=2)
    # run1 avg of (0.5, 0.6) = 0.55
    # run2 avg of (0.7, 0.8) = 0.75
    # mean = 0.65
    assert abs(band.metrics["x"]["mean"] - 0.65) < 1e-6
    print("test_compute_noise_band_window: PASS")


# ---------------------------------------------------------------------------
# save/load round-trip tests
# ---------------------------------------------------------------------------

def test_save_noise_band(tmp_path):
    """save_noise_band writes JSON."""
    band = NoiseBand(metrics={"x": {"mean": 0.5, "std": 0.1}}, n_seeds=4, n_updates=50)
    path = tmp_path / "noise.json"
    save_noise_band(band, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["metrics"]["x"]["mean"] == 0.5
    assert data["n_seeds"] == 4
    print("test_save_noise_band: PASS")


def test_save_load_roundtrip(tmp_path):
    """save → load round-trip preserves data."""
    band = NoiseBand(
        metrics={"x": {"mean": 0.5, "std": 0.1}, "y": {"mean": 0.9, "std": 0.05}},
        n_seeds=4, n_updates=50,
    )
    path = tmp_path / "noise.json"
    save_noise_band(band, path)

    from baseline.framework.ppo.debug.compare import load_noise_band
    loaded = load_noise_band(path)
    assert loaded.metrics["x"]["mean"] == 0.5
    assert loaded.metrics["y"]["std"] == 0.05
    assert loaded.n_seeds == 4
    print("test_save_load_roundtrip: PASS")


# ---------------------------------------------------------------------------
# render_noise_summary tests
# ---------------------------------------------------------------------------

def test_render_noise_summary():
    """render_noise_summary produces a summary."""
    band = NoiseBand(
        metrics={"x": {"mean": 0.5, "std": 0.1}, "y": {"mean": 0.9, "std": 0.05}},
        n_seeds=4, n_updates=50,
    )
    run_dirs = [Path("/tmp/run1"), Path("/tmp/run2")]
    text = render_noise_summary(band, run_dirs)
    assert "4 seeds" in text
    assert "x" in text
    assert "y" in text
    assert "0.5000" in text or "0.5000" in text
    print("test_render_noise_summary: PASS")


# ---------------------------------------------------------------------------
# _pid_alive tests
# ---------------------------------------------------------------------------

def test_pid_alive_self():
    """_pid_alive returns True for current process."""
    import os
    assert _pid_alive(os.getpid())
    print("test_pid_alive_self: PASS")


def test_pid_alive_dead():
    """_pid_alive returns False for non-existent PID."""
    # PID 999999 is very likely non-existent
    assert not _pid_alive(999999)
    print("test_pid_alive_dead: PASS")


# ---------------------------------------------------------------------------
# Integration: noise band → compare
# ---------------------------------------------------------------------------

def test_noise_band_with_compare(tmp_path):
    """Noise band from noise() can be used with compare()."""
    from baseline.framework.ppo.debug.compare import compare

    # Create 3 noise runs
    rd1 = _make_run_dir(tmp_path, "noise1", [0.5, 0.55])
    rd2 = _make_run_dir(tmp_path, "noise2", [0.6, 0.65])
    rd3 = _make_run_dir(tmp_path, "noise3", [0.7, 0.75])

    band = compute_noise_band([rd1, rd2, rd3], window=2)
    noise_path = tmp_path / "noise.json"
    save_noise_band(band, noise_path)

    # Create two experiment runs
    run_a = _make_run_dir(tmp_path, "run_a", [0.5])
    run_b = _make_run_dir(tmp_path, "run_b", [0.8])

    result = compare(run_a, run_b, noise_band_path=noise_path)
    assert "噪声基线" in result
    print("test_noise_band_with_compare: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_compute_noise_band_basic(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_compute_noise_band_single_run(Path(td))
    test_compute_noise_band_empty()
    with tempfile.TemporaryDirectory() as td:
        test_compute_noise_band_window(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_save_noise_band(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_save_load_roundtrip(Path(td))
    test_render_noise_summary()
    test_pid_alive_self()
    test_pid_alive_dead()
    with tempfile.TemporaryDirectory() as td:
        test_noise_band_with_compare(Path(td))
    print("\nAll S6 noise tests passed.")
