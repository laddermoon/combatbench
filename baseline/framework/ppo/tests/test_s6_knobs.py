"""S6 ``knobs`` module tests.

Tests cover:
- KnobCheck/KnobContext/KnobResult construction.
- Built-in knob configured/observed/agree functions with synthetic data.
- run_knob_checks with passing/failing/skipped knobs.
- render_knob_table format.
- parse_log_entry, parse_floor_active, load_config helpers.

Conventions follow test_s3_where.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.knobs import (
    KnobCheck, KnobContext, KnobResult,
    builtin_knob_checks, run_knob_checks, render_knob_table,
    parse_log_entry, parse_floor_active, load_config,
    _cfg_explore_factor, _obs_explore_factor, _agree_explore_factor,
    _cfg_uncertainty_floor, _obs_uncertainty_floor, _agree_uncertainty_floor,
    _cfg_floor_weight, _obs_floor_weight, _agree_floor_weight,
    _cfg_resume, _obs_resume, _agree_resume,
    _cfg_observer, _obs_observer, _agree_observer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(run_dir, lines):
    """Write a train.log with the given lines."""
    (run_dir / "train.log").write_text("\n".join(lines) + "\n")


def _write_config(run_dir, config):
    """Write a config.json."""
    (run_dir / "config.json").write_text(json.dumps(config))


def _make_ctx(run_dir, log_entry=None, log_stats=None, snapshot_dir=None, experiment=None, config=None):
    """Build a KnobContext for testing."""
    return KnobContext(
        run_dir=Path(run_dir),
        config=config or {},
        log_entry=log_entry,
        log_stats=log_stats,
        snapshot_dir=Path(snapshot_dir) if snapshot_dir else None,
        experiment=experiment,
    )


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

def test_knob_check_construction():
    """KnobCheck can be constructed with callables."""
    check = KnobCheck(
        name="test",
        configured=lambda ctx: 1,
        observed=lambda ctx: 1,
        agree=lambda cfg, obs: cfg == obs,
    )
    assert check.name == "test"
    assert not check.requires_snapshot
    print("test_knob_check_construction: PASS")


def test_knob_check_requires_snapshot():
    """KnobCheck with requires_snapshot=True."""
    check = KnobCheck(
        name="snap_knob",
        configured=lambda ctx: "x",
        observed=lambda ctx: "y",
        agree=lambda cfg, obs: True,
        requires_snapshot=True,
    )
    assert check.requires_snapshot
    print("test_knob_check_requires_snapshot: PASS")


def test_knob_result_construction():
    """KnobResult can be constructed."""
    r = KnobResult(name="test", configured_value=1, observed_value=1, passed=True)
    assert r.passed
    assert not r.skipped
    print("test_knob_result_construction: PASS")


def test_builtin_knob_checks_count():
    """builtin_knob_checks returns 5 knobs."""
    checks = builtin_knob_checks()
    assert len(checks) == 5
    names = [c.name for c in checks]
    assert "explore_factor" in names
    assert "uncertainty_floor" in names
    assert "floor_weight" in names
    assert "resume" in names
    assert "observer" in names
    print("test_builtin_knob_checks_count: PASS")


def test_observer_knob_requires_snapshot():
    """The observer knob requires a snapshot."""
    checks = builtin_knob_checks()
    obs = [c for c in checks if c.name == "observer"][0]
    assert obs.requires_snapshot
    print("test_observer_knob_requires_snapshot: PASS")


# ---------------------------------------------------------------------------
# explore_factor knob tests
# ---------------------------------------------------------------------------

def test_obs_explore_factor_from_stats():
    """_obs_explore_factor reads eff_std_mean / std_mean."""
    stats = {"eff_std_mean": 0.18, "std_mean": 0.36}
    ctx = _make_ctx("/tmp", log_stats=stats)
    ratio = _obs_explore_factor(ctx)
    assert abs(ratio - 0.5) < 1e-6
    print("test_obs_explore_factor_from_stats: PASS")


def test_obs_explore_factor_missing():
    """_obs_explore_factor returns None when stats missing."""
    ctx = _make_ctx("/tmp", log_stats=None)
    assert _obs_explore_factor(ctx) is None
    print("test_obs_explore_factor_missing: PASS")


def test_agree_explore_factor_phase_dependent():
    """Phase-dependent ef: ratio 1.5 → agree."""
    assert _agree_explore_factor("相位相关", 1.5)
    assert _agree_explore_factor("相位相关", 1.3)
    assert _agree_explore_factor("相位相关", 1.8)
    assert not _agree_explore_factor("相位相关", 1.0)
    assert not _agree_explore_factor("相位相关", 2.0)
    print("test_agree_explore_factor_phase_dependent: PASS")


def test_agree_explore_factor_fixed():
    """Fixed ef: ratio ~1.0 → agree."""
    assert _agree_explore_factor(1.0, 1.0)
    assert _agree_explore_factor(0.5, 0.95)
    assert not _agree_explore_factor(0.5, 1.5)
    print("test_agree_explore_factor_fixed: PASS")


# ---------------------------------------------------------------------------
# uncertainty_floor knob tests
# ---------------------------------------------------------------------------

def test_obs_uncertainty_floor_from_log(tmp_path):
    """_obs_uncertainty_floor reads floor_active from GradDiag lines."""
    _write_log(tmp_path, [
        "[update 1] something",
        "  [GradDiag] log_std grad: pol_abs=0.1 floor_abs=0.0 | floor_active=0.5",
        "[update 2] something",
        "  [GradDiag] log_std grad: pol_abs=0.1 floor_abs=0.0 | floor_active=0.0",
    ])
    ctx = _make_ctx(tmp_path)
    # Latest should be 0.0
    assert _obs_uncertainty_floor(ctx) == 0.0

    # Target update 1
    ctx1 = _make_ctx(tmp_path)
    from baseline.framework.ppo.debug.knobs import parse_floor_active
    assert parse_floor_active(tmp_path, target_update=1) == 0.5
    print("test_obs_uncertainty_floor_from_log: PASS")


def test_agree_uncertainty_floor_active():
    """Floor configured and active → agree when floor_active > 0."""
    assert _agree_uncertainty_floor(0.35, 0.5)
    assert not _agree_uncertainty_floor(0.35, 0.0)
    print("test_agree_uncertainty_floor_active: PASS")


def test_agree_uncertainty_floor_disabled():
    """Floor disabled → agree when floor_active ≈ 0."""
    assert _agree_uncertainty_floor("0.35 (已关闭@u110)", 0.0)
    assert not _agree_uncertainty_floor("0.35 (已关闭@u110)", 0.5)
    print("test_agree_uncertainty_floor_disabled: PASS")


# ---------------------------------------------------------------------------
# floor_weight knob tests
# ---------------------------------------------------------------------------

def test_obs_floor_weight_from_buffer_stats():
    """_obs_floor_weight reads unique aw values from buffer_stats."""
    entry = {
        "buffer_stats": {
            "per_channel": {
                "r_a": {"actor_weight_min": 0.0, "actor_weight_max": 1.0},
                "r_b": {"actor_weight_min": 0.0, "actor_weight_max": 0.0},
            }
        }
    }
    ctx = _make_ctx("/tmp", log_entry=entry)
    vals = _obs_floor_weight(ctx)
    assert set(vals) == {0.0, 1.0}
    print("test_obs_floor_weight_from_buffer_stats: PASS")


def test_agree_floor_weight_mask():
    """Mask-based weights (0/1) → agree."""
    assert _agree_floor_weight("balance_mask", [0.0, 1.0])
    assert _agree_floor_weight("balance_mask", [0.0, 0.0, 1.0])
    assert not _agree_floor_weight("balance_mask", [0.5, 1.5])
    print("test_agree_floor_weight_mask: PASS")


# ---------------------------------------------------------------------------
# resume knob tests
# ---------------------------------------------------------------------------

def test_cfg_resume_none(tmp_path):
    """_cfg_resume returns '无' when no resume in log."""
    _write_log(tmp_path, ["[update 1] something"])
    ctx = _make_ctx(tmp_path)
    assert _cfg_resume(ctx) == "无"
    print("test_cfg_resume_none: PASS")


def test_cfg_resume_found(tmp_path):
    """_cfg_resume finds resume event."""
    _write_log(tmp_path, [
        "[checkpoint] resuming from update 150",
        "[update 151] something",
    ])
    ctx = _make_ctx(tmp_path)
    assert "resume@u150" in _cfg_resume(ctx)
    print("test_cfg_resume_found: PASS")


def test_obs_resume_no_checkpoint(tmp_path):
    """_obs_resume returns '无checkpoint' when no checkpoints."""
    ctx = _make_ctx(tmp_path)
    assert _obs_resume(ctx) == "无checkpoint"
    print("test_obs_resume_no_checkpoint: PASS")


def test_obs_resume_with_checkpoint(tmp_path):
    """_obs_resume finds checkpoint."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint_u00250.pt").write_text("dummy")
    ctx = _make_ctx(tmp_path)
    assert "u00250" in _obs_resume(ctx)
    print("test_obs_resume_with_checkpoint: PASS")


def test_agree_resume():
    """_agree_resume logic."""
    assert _agree_resume("无", "无checkpoint")
    assert _agree_resume("resume@u150", "ckpt@u250")
    assert not _agree_resume("resume@u150", "无checkpoint")
    print("test_agree_resume: PASS")


# ---------------------------------------------------------------------------
# observer knob tests
# ---------------------------------------------------------------------------

def test_obs_observer_no_snapshot():
    """_obs_observer returns None when no snapshot."""
    ctx = _make_ctx("/tmp", snapshot_dir=None)
    assert _obs_observer(ctx) is None
    print("test_obs_observer_no_snapshot: PASS")


# ---------------------------------------------------------------------------
# run_knob_checks tests
# ---------------------------------------------------------------------------

def test_run_knob_checks_passing():
    """run_knob_checks with all-passing knobs."""
    ctx = _make_ctx("/tmp")
    checks = [
        KnobCheck(name="a", configured=lambda c: 1, observed=lambda c: 1,
                  agree=lambda cfg, obs: cfg == obs),
    ]
    results = run_knob_checks(ctx, checks)
    assert len(results) == 1
    assert results[0].passed
    print("test_run_knob_checks_passing: PASS")


def test_run_knob_checks_failing():
    """run_knob_checks with failing knob."""
    ctx = _make_ctx("/tmp")
    checks = [
        KnobCheck(name="a", configured=lambda c: 1, observed=lambda c: 2,
                  agree=lambda cfg, obs: cfg == obs),
    ]
    results = run_knob_checks(ctx, checks)
    assert not results[0].passed
    print("test_run_knob_checks_failing: PASS")


def test_run_knob_checks_skip_snapshot():
    """run_knob_checks skips snapshot-dependent knobs without snapshot."""
    ctx = _make_ctx("/tmp", snapshot_dir=None)
    checks = [
        KnobCheck(name="snap", configured=lambda c: "x", observed=lambda c: "y",
                  agree=lambda cfg, obs: True, requires_snapshot=True),
    ]
    results = run_knob_checks(ctx, checks)
    assert results[0].skipped
    assert "需快照" in results[0].note
    print("test_run_knob_checks_skip_snapshot: PASS")


def test_run_knob_checks_configured_exception():
    """run_knob_checks handles configured() exceptions."""
    ctx = _make_ctx("/tmp")
    checks = [
        KnobCheck(name="bad", configured=lambda c: 1/0, observed=lambda c: 1,
                  agree=lambda cfg, obs: True),
    ]
    results = run_knob_checks(ctx, checks)
    assert "错误" in str(results[0].configured_value)
    print("test_run_knob_checks_configured_exception: PASS")


# ---------------------------------------------------------------------------
# render_knob_table tests
# ---------------------------------------------------------------------------

def test_render_knob_table_basic():
    """render_knob_table produces a table."""
    results = [
        KnobResult(name="test", configured_value=1.0, observed_value=1.0, passed=True),
        KnobResult(name="bad", configured_value=1.0, observed_value=2.0, passed=False),
    ]
    text = render_knob_table(results, update=250)
    assert "update 250" in text
    assert "test" in text
    assert "✓" in text
    assert "✗" in text
    print("test_render_knob_table_basic: PASS")


def test_render_knob_table_skipped():
    """render_knob_table shows skipped knobs."""
    results = [
        KnobResult(name="obs", configured_value="—", observed_value=None,
                   passed=False, note="需快照", skipped=True),
    ]
    text = render_knob_table(results)
    assert "需快照" in text
    assert "—" in text
    print("test_render_knob_table_skipped: PASS")


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------

def test_parse_log_entry(tmp_path):
    """parse_log_entry finds the right entry."""
    _write_log(tmp_path, [
        "some line",
        '__RAW_STATS__ {"update": 1, "stats": {"x": 0.5}}',
        '__RAW_STATS__ {"update": 2, "stats": {"x": 0.6}}',
    ])
    # Latest
    entry = parse_log_entry(tmp_path)
    assert entry["update"] == 2
    # Target
    entry = parse_log_entry(tmp_path, target_update=1)
    assert entry["update"] == 1
    print("test_parse_log_entry: PASS")


def test_parse_log_entry_no_log(tmp_path):
    """parse_log_entry returns None when no log."""
    assert parse_log_entry(tmp_path) is None
    print("test_parse_log_entry_no_log: PASS")


def test_parse_floor_active(tmp_path):
    """parse_floor_active extracts floor_active."""
    _write_log(tmp_path, [
        "[update 1]",
        "  [GradDiag] floor_active=0.5",
        "[update 2]",
        "  [GradDiag] floor_active=0.0",
    ])
    assert parse_floor_active(tmp_path) == 0.0
    assert parse_floor_active(tmp_path, target_update=1) == 0.5
    print("test_parse_floor_active: PASS")


def test_load_config(tmp_path):
    """load_config reads config.json."""
    _write_config(tmp_path, {"experiment": {"name": "test"}})
    cfg = load_config(tmp_path)
    assert cfg["experiment"]["name"] == "test"
    print("test_load_config: PASS")


def test_load_config_missing(tmp_path):
    """load_config returns {} when no config.json."""
    assert load_config(tmp_path) == {}
    print("test_load_config_missing: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_knob_check_construction()
    test_knob_check_requires_snapshot()
    test_knob_result_construction()
    test_builtin_knob_checks_count()
    test_observer_knob_requires_snapshot()
    test_obs_explore_factor_from_stats()
    test_obs_explore_factor_missing()
    test_agree_explore_factor_phase_dependent()
    test_agree_explore_factor_fixed()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_obs_uncertainty_floor_from_log(Path(td))
    test_agree_uncertainty_floor_active()
    test_agree_uncertainty_floor_disabled()
    test_obs_floor_weight_from_buffer_stats()
    test_agree_floor_weight_mask()
    with tempfile.TemporaryDirectory() as td:
        test_cfg_resume_none(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_cfg_resume_found(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_obs_resume_no_checkpoint(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_obs_resume_with_checkpoint(Path(td))
    test_agree_resume()
    test_obs_observer_no_snapshot()
    test_run_knob_checks_passing()
    test_run_knob_checks_failing()
    test_run_knob_checks_skip_snapshot()
    test_run_knob_checks_configured_exception()
    test_render_knob_table_basic()
    test_render_knob_table_skipped()
    with tempfile.TemporaryDirectory() as td:
        test_parse_log_entry(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_log_entry_no_log(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_parse_floor_active(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_load_config(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_load_config_missing(Path(td))
    print("\nAll S6 knobs tests passed.")
