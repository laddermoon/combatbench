"""S3 ``--where`` expression parser tests.

Tests cover:
- Single condition parse + evaluate.
- Multiple conditions (&&) parse + evaluate.
- All operators (<, >, <=, >=, ==, !=).
- Empty expression → match all.
- Syntax errors raise ValueError.
- Unknown field raises KeyError.
- flatten_npz + build_frame_arrays on synthetic data.

Conventions follow test_s2_sink.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.where import (
    Condition, WhereClause, parse, flatten_npz, build_frame_arrays,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_single_condition():
    """Parse a single field-comparison condition."""
    clause = parse("combine.aw_normed.r_left_foot < 0")
    assert len(clause.conditions) == 1
    c = clause.conditions[0]
    assert c.field_path == "combine.aw_normed.r_left_foot"
    assert c.op == "<"
    assert c.value == 0.0
    print("test_parse_single_condition: PASS")


def test_parse_multiple_conditions():
    """Parse multiple conditions joined by &&."""
    clause = parse("buffer.explore_factor > 0 && combine.aw_normed.r_potential < 0")
    assert len(clause.conditions) == 2
    assert clause.conditions[0].field_path == "buffer.explore_factor"
    assert clause.conditions[0].op == ">"
    assert clause.conditions[0].value == 0.0
    assert clause.conditions[1].field_path == "combine.aw_normed.r_potential"
    assert clause.conditions[1].op == "<"
    assert clause.conditions[1].value == 0.0
    print("test_parse_multiple_conditions: PASS")


def test_parse_all_operators():
    """All six operators parse correctly."""
    for op in ["<", ">", "<=", ">=", "==", "!="]:
        clause = parse(f"combine.aw_frame.r_a {op} 0.5")
        assert len(clause.conditions) == 1
        assert clause.conditions[0].op == op
    print("test_parse_all_operators: PASS")


def test_parse_negative_values():
    """Negative values parse correctly."""
    clause = parse("combine.aw_normed.r_a < -0.5")
    assert clause.conditions[0].value == -0.5
    print("test_parse_negative_values: PASS")


def test_parse_empty_matches_all():
    """Empty expression matches all frames."""
    clause = parse("")
    assert len(clause.conditions) == 0
    arrays = {"x": np.arange(10)}
    mask = clause.match(arrays)
    assert mask.all()
    print("test_parse_empty_matches_all: PASS")


def test_match_single_condition():
    """Single condition returns correct boolean mask."""
    clause = parse("x > 5")
    arrays = {"x": np.arange(10).astype(np.float32)}
    mask = clause.match(arrays)
    assert mask.sum() == 4  # 6, 7, 8, 9
    print("test_match_single_condition: PASS")


def test_match_multiple_conditions_and():
    """Multiple conditions are ANDed."""
    clause = parse("x > 3 && x < 7")
    arrays = {"x": np.arange(10).astype(np.float32)}
    mask = clause.match(arrays)
    assert mask.sum() == 3  # 4, 5, 6
    print("test_match_multiple_conditions_and: PASS")


def test_match_no_conditions():
    """No conditions → match all (when arrays exist)."""
    clause = WhereClause(conditions=[])
    arrays = {"x": np.arange(5)}
    mask = clause.match(arrays)
    assert mask.sum() == 5
    print("test_match_no_conditions: PASS")


def test_syntax_error_raises():
    """Malformed expressions raise ValueError."""
    with pytest.raises(ValueError):
        parse("not a valid expression")
    with pytest.raises(ValueError):
        parse("x")
    with pytest.raises(ValueError):
        parse("x >> 5")  # invalid operator
    print("test_syntax_error_raises: PASS")


def test_unknown_field_raises():
    """Unknown field raises KeyError during evaluation."""
    clause = parse("nonexistent.field < 5")
    arrays = {"x": np.arange(10)}
    with pytest.raises(KeyError):
        clause.match(arrays)
    print("test_unknown_field_raises: PASS")


def test_flatten_npz():
    """flatten_npz prefixes keys with stage name."""
    data = {"aw_normed.r_left_foot": np.arange(5), "combined_adv": np.arange(5)}
    out = flatten_npz(data, "combine")
    assert "combine.aw_normed.r_left_foot" in out
    assert "combine.combined_adv" in out
    print("test_flatten_npz: PASS")


def test_build_frame_arrays(tmp_path):
    """build_frame_arrays loads all 4 stage .npz + debug_arrays.npz."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for stage in ("buffer", "gae", "combine", "update"):
        np.savez(replay_dir / f"{stage}.npz",
                 **{f"test_key_{stage}": np.arange(3)})
    np.savez(replay_dir / "debug_arrays.npz",
             **{"h_left": np.arange(3), "h_right": np.arange(3)})

    arrays = build_frame_arrays(replay_dir)
    assert "buffer.test_key_buffer" in arrays
    assert "gae.test_key_gae" in arrays
    assert "combine.test_key_combine" in arrays
    assert "update.test_key_update" in arrays
    assert "debug.h_left" in arrays
    assert "debug.h_right" in arrays
    print("test_build_frame_arrays: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_parse_single_condition()
    test_parse_multiple_conditions()
    test_parse_all_operators()
    test_parse_negative_values()
    test_parse_empty_matches_all()
    test_match_single_condition()
    test_match_multiple_conditions_and()
    test_match_no_conditions()
    test_syntax_error_raises()
    test_unknown_field_raises()
    test_flatten_npz()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_build_frame_arrays(Path(td))
    print("\nAll S3 where tests passed.")
