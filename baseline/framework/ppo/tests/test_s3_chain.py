"""S3 ``chain`` tool tests.

Tests cover:
- Per-ring assessment functions (physics, observation, reward, etc.).
- Diagnosis text generation for main breakpoint.
- Diagnosis for no-breakpoint case.
- Full chain() with synthetic log + snapshot (no probes).
- chain() degrades gracefully without snapshot.
- render_report produces the nine-ring table.

Conventions follow test_s2_sink.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.chain import (
    ChainReport, RingResult, chain, render_report,
    _assess_physics, _assess_observation, _assess_reward,
    _assess_prediction, _assess_advantage, _assess_gating,
    _assess_combination, _assess_gradient, _assess_behavior,
    _generate_diagnosis,
    _OK, _WARN, _FAIL, _NA,
)


# ---------------------------------------------------------------------------
# Per-ring assessment tests
# ---------------------------------------------------------------------------

def test_assess_physics_ok():
    r = _assess_physics(0.5)
    assert r.judgment == _OK
    print("test_assess_physics_ok: PASS")


def test_assess_physics_fail():
    r = _assess_physics(0.01)
    assert r.judgment == _FAIL
    print("test_assess_physics_fail: PASS")


def test_assess_physics_warn():
    r = _assess_physics(0.1)
    assert r.judgment == _WARN
    print("test_assess_physics_warn: PASS")


def test_assess_physics_na():
    r = _assess_physics(None)
    assert r.judgment == _NA
    print("test_assess_physics_na: PASS")


def test_assess_prediction_ok():
    r = _assess_prediction(0.8)
    assert r.judgment == _OK
    print("test_assess_prediction_ok: PASS")


def test_assess_prediction_fail():
    r = _assess_prediction(0.1)
    assert r.judgment == _FAIL
    print("test_assess_prediction_fail: PASS")


def test_assess_advantage_fail():
    """Low SNR → ✗."""
    arrays = {"gae.advantages.r_a": np.zeros(100, dtype=np.float32)}
    r = _assess_advantage(arrays, "r_a")
    assert r.judgment == _FAIL
    print("test_assess_advantage_fail: PASS")


def test_assess_advantage_ok():
    """High SNR → ✓."""
    arrays = {"gae.advantages.r_a": np.linspace(-1, 1, 100, dtype=np.float32)}
    r = _assess_advantage(arrays, "r_a")
    assert r.judgment == _OK
    print("test_assess_advantage_ok: PASS")


def test_assess_combination_fail():
    """Low influence share → ✗."""
    r = _assess_combination(0.02)
    assert r.judgment == _FAIL
    print("test_assess_combination_fail: PASS")


def test_assess_combination_ok():
    r = _assess_combination(0.5)
    assert r.judgment == _OK
    print("test_assess_combination_ok: PASS")


def test_assess_gating_warn_negative():
    """Negative mean aw → ⚠ (net suppression)."""
    arr = np.full(100, -0.1, dtype=np.float32)
    arrays = {"combine.aw_frame.r_a": arr}
    r = _assess_gating(arrays, "r_a")
    assert r.judgment == _WARN
    print("test_assess_gating_warn_negative: PASS")


def test_assess_gating_ok():
    """Positive mean aw, high non-zero ratio → ✓."""
    arr = np.full(100, 0.1, dtype=np.float32)
    arrays = {"combine.aw_frame.r_a": arr}
    r = _assess_gating(arrays, "r_a")
    assert r.judgment == _OK
    print("test_assess_gating_ok: PASS")


def test_assess_gradient_fail():
    """Min gradient near zero → ✗."""
    grads = np.array([0.1, 0.05, 0.0001], dtype=np.float32)
    r = _assess_gradient(grads)
    assert r.judgment == _FAIL
    print("test_assess_gradient_fail: PASS")


def test_assess_gradient_ok():
    """All gradients comparable → ✓."""
    grads = np.array([0.1, 0.08, 0.09], dtype=np.float32)
    r = _assess_gradient(grads)
    assert r.judgment == _OK
    print("test_assess_gradient_ok: PASS")


def test_assess_behavior_fail():
    r = _assess_behavior(0.0)
    assert r.judgment == _FAIL
    print("test_assess_behavior_fail: PASS")


def test_assess_behavior_ok():
    r = _assess_behavior(0.8)
    assert r.judgment == _OK
    print("test_assess_behavior_ok: PASS")


# ---------------------------------------------------------------------------
# Diagnosis tests
# ---------------------------------------------------------------------------

def test_diagnosis_main_breakpoint():
    """Main breakpoint = first ✗ ring."""
    rings = [
        RingResult(1, "物理", "rate 50%", _OK),
        RingResult(2, "观测", "ok", _OK),
        RingResult(3, "奖励", "ok", _OK),
        RingResult(4, "预测", "EV 0.1", _FAIL),
        RingResult(5, "优势", "ok", _OK),
        RingResult(6, "门控", "ok", _OK),
        RingResult(7, "合成", "share 2%", _FAIL),
        RingResult(8, "梯度", "ok", _OK),
        RingResult(9, "行为", "ok", _OK),
    ]
    main_bp, diag = _generate_diagnosis(rings)
    assert main_bp == 4
    assert "4" in diag
    print("test_diagnosis_main_breakpoint: PASS")


def test_diagnosis_no_breakpoint():
    """No ✗ → no main breakpoint."""
    rings = [RingResult(i+1, "x", "ok", _OK) for i in range(9)]
    main_bp, diag = _generate_diagnosis(rings)
    assert main_bp is None
    assert "正常" in diag
    print("test_diagnosis_no_breakpoint: PASS")


def test_diagnosis_with_warnings():
    """No ✗ but warnings → diagnosis mentions warnings."""
    rings = [RingResult(i+1, "x", "ok", _OK) for i in range(9)]
    rings[2] = RingResult(3, "奖励", "P50≈0", _WARN)
    main_bp, diag = _generate_diagnosis(rings)
    assert main_bp is None
    assert "警告" in diag
    print("test_diagnosis_with_warnings: PASS")


# ---------------------------------------------------------------------------
# Full chain() tests
# ---------------------------------------------------------------------------

def _write_log(run_dir, entries):
    lines = []
    for e in entries:
        stats = e.get("stats", {})
        line = f"[update {e['update']}] __RAW_STATS__ {json.dumps({'update': e['update'], 'stats': stats})}\n"
        lines.append(line)
    (run_dir / "train.log").write_text("".join(lines))


def test_chain_log_only(tmp_path):
    """chain() with only log data (no snapshot, no probes)."""
    _write_log(tmp_path, [{
        "update": 1,
        "stats": {
            "explained_variance_r_a": 0.7,
            "influence_share_r_a": 0.5,
            "dead_frame_ratio": 0.1,
            "grad_dim_00": 0.1, "grad_dim_01": 0.05, "grad_dim_02": 0.08,
        },
    }])

    report = chain(tmp_path, channel="r_a", update=1, run_probes=False)
    assert report.update == 1
    assert report.channel == "r_a"
    assert len(report.rings) == 9
    # ④预测 should be OK (EV=0.7).
    assert report.rings[3].judgment == _OK
    # ⑦合成 should be OK (influence=0.5).
    assert report.rings[6].judgment == _OK
    print("test_chain_log_only: PASS")


def test_chain_with_snapshot(tmp_path):
    """chain() with snapshot replay arrays."""
    _write_log(tmp_path, [{
        "update": 1,
        "stats": {
            "explained_variance_r_a": 0.7,
            "influence_share_r_a": 0.02,  # low → ✗
            "dead_frame_ratio": 0.1,
            "grad_dim_00": 0.1, "grad_dim_01": 0.05, "grad_dim_02": 0.08,
        },
    }])

    # Create a fake snapshot with replay arrays.
    snapshot_dir = tmp_path / "debug" / "u00001"
    replay_dir = snapshot_dir / "replay"
    replay_dir.mkdir(parents=True)
    np.savez(replay_dir / "gae.npz",
             **{"advantages.r_a": np.linspace(-1, 1, 50, dtype=np.float32),
                "returns.r_a": np.ones(50, dtype=np.float32)})
    np.savez(replay_dir / "combine.npz",
             **{"aw_frame.r_a": np.full(50, 0.1, dtype=np.float32),
                "aw_normed.r_a": np.full(50, 0.5, dtype=np.float32)})
    np.savez(replay_dir / "debug_arrays.npz",
             **{"h_left": np.ones(50, dtype=np.float32) * 0.1,
                "h_right": np.ones(50, dtype=np.float32) * 0.1})

    report = chain(tmp_path, channel="r_a", update=1, run_probes=False)
    assert report.rings[1].judgment == _OK  # ②观测 has debug arrays
    assert report.rings[4].judgment == _OK  # ⑤优势 has good SNR
    assert report.rings[5].judgment == _OK  # ⑥门控 positive mean
    assert report.rings[6].judgment == _FAIL  # ⑦合成 low influence
    # Main breakpoint should be ⑦.
    assert report.main_breakpoint == 7
    print("test_chain_with_snapshot: PASS")


def test_chain_no_log_raises(tmp_path):
    """chain() raises when no log exists and update is None."""
    with pytest.raises(FileNotFoundError):
        chain(tmp_path, channel="r_a", update=None, run_probes=False)
    print("test_chain_no_log_raises: PASS")


def test_render_report(tmp_path):
    """render_report produces the nine-ring table."""
    report = ChainReport(
        channel="r_a", update=1,
        rings=[RingResult(i+1, f"ring{i+1}", "ok", _OK) for i in range(9)],
        main_breakpoint=None,
        diagnosis="所有环正常。",
    )
    text = render_report(report)
    assert "r_a" in text
    assert "update 1" in text
    assert "①" in text or "1" in text
    print("test_render_report: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_assess_physics_ok()
    test_assess_physics_fail()
    test_assess_physics_warn()
    test_assess_physics_na()
    test_assess_prediction_ok()
    test_assess_prediction_fail()
    test_assess_advantage_fail()
    test_assess_advantage_ok()
    test_assess_combination_fail()
    test_assess_combination_ok()
    test_assess_gating_warn_negative()
    test_assess_gating_ok()
    test_assess_gradient_fail()
    test_assess_gradient_ok()
    test_assess_behavior_fail()
    test_assess_behavior_ok()
    test_diagnosis_main_breakpoint()
    test_diagnosis_no_breakpoint()
    test_diagnosis_with_warnings()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_chain_log_only(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_chain_with_snapshot(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_chain_no_log_raises(Path(td))
    test_render_report(Path("/tmp"))
    print("\nAll S3 chain tests passed.")
