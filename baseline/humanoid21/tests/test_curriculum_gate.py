"""Pin the eval-driven curriculum gate's stage classification.

The gate (rewritten 2026-05-09) is a stateless single-shot classifier:

  - on every eval cycle, ``assign_from_eval(eval_summary)`` looks at
    ``mean_length`` and ``in_range_ratio`` from the deterministic eval
    batch and picks the next training stage (1, 2, or 3);
  - no hysteresis, no dwell, no rolling window;
  - any stage can transition to any other stage in a single call;
  - lower-stage rewards remain active in higher stages via the weight
    schedule (1,0,0) / (1,1,0) / (1,1,1).

Tests cover:
  * weight schedule;
  * single-eval classification of all three regimes;
  * arbitrary jumps (1->3, 3->1, 2->2 stay);
  * regression to stage 1 in a single eval;
  * ``current_state()`` snapshot behavior between evals;
  * input validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from baseline.humanoid21.common import CurriculumStageGate


def _make_gate(**overrides):
    defaults = dict(
        max_steps=200,
        pass_len_ratio=0.95,
        pass_in_range=0.80,
    )
    defaults.update(overrides)
    return CurriculumStageGate(**defaults)


def _eval(*, length, in_range=0.0, max_steps=200):
    return {"mean_length": length, "in_range_ratio": in_range, "max_steps": max_steps}


# ---------------------------------------------------------------------------
# Weight schedule.
# ---------------------------------------------------------------------------
class TestWeightSchedule:
    def test_initial_stage1_weights(self):
        gate = _make_gate()
        assert gate.stage == 1
        assert gate.weights == (1.0, 0.0, 0.0)

    def test_stage_weights_table(self):
        assert CurriculumStageGate.STAGE_WEIGHTS[1] == (1.0, 0.0, 0.0)
        assert CurriculumStageGate.STAGE_WEIGHTS[2] == (1.0, 1.0, 0.0)
        assert CurriculumStageGate.STAGE_WEIGHTS[3] == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Single-shot classification: all three regimes.
# ---------------------------------------------------------------------------
class TestSingleShotClassification:
    def test_below_pass_len_ratio_picks_stage1(self):
        gate = _make_gate(initial_stage=3)  # start at 3 to prove we can drop
        info = gate.assign_from_eval(_eval(length=0.5 * 200, in_range=0.99))
        assert info["stage"] == 1
        assert info["weights"] == (1.0, 0.0, 0.0)
        assert info["prev_stage"] == 3
        assert "stage 1" in info["reason"]

    def test_balance_ok_but_low_in_range_picks_stage2(self):
        gate = _make_gate()
        info = gate.assign_from_eval(_eval(length=0.99 * 200, in_range=0.30))
        assert info["stage"] == 2
        assert info["weights"] == (1.0, 1.0, 0.0)
        assert "stage 2" in info["reason"]

    def test_balance_and_in_range_both_ok_picks_stage3(self):
        gate = _make_gate()
        info = gate.assign_from_eval(_eval(length=1.0 * 200, in_range=0.85))
        assert info["stage"] == 3
        assert info["weights"] == (1.0, 1.0, 1.0)
        assert "stage 3" in info["reason"]


# ---------------------------------------------------------------------------
# Arbitrary jumps — no fixed transition graph.
# ---------------------------------------------------------------------------
class TestArbitraryJumps:
    def test_stage1_to_stage3_in_one_eval(self):
        gate = _make_gate()
        info = gate.assign_from_eval(_eval(length=0.99 * 200, in_range=0.90))
        assert info["prev_stage"] == 1
        assert info["stage"] == 3

    def test_stage3_to_stage1_in_one_eval(self):
        gate = _make_gate(initial_stage=3)
        info = gate.assign_from_eval(_eval(length=0.50 * 200, in_range=0.99))
        assert info["prev_stage"] == 3
        assert info["stage"] == 1

    def test_repeated_eval_stays_same_when_metrics_stable(self):
        gate = _make_gate()
        for _ in range(5):
            info = gate.assign_from_eval(_eval(length=0.96 * 200, in_range=0.30))
        assert info["stage"] == 2

    def test_eval_can_oscillate_freely(self):
        gate = _make_gate()
        # The gate has zero memory — perfectly fine for the classifier
        # to flip 1->2->1->3 if eval metrics flip. This is by design.
        seq = [
            (0.99, 0.30, 2),
            (0.50, 0.99, 1),
            (0.99, 0.99, 3),
            (0.50, 0.50, 1),
        ]
        for length_ratio, in_range, expected in seq:
            info = gate.assign_from_eval(
                _eval(length=length_ratio * 200, in_range=in_range)
            )
            assert info["stage"] == expected, (
                f"len_ratio={length_ratio} in_range={in_range} "
                f"expected stage {expected}, got {info['stage']}"
            )


# ---------------------------------------------------------------------------
# current_state() — snapshot for non-eval updates.
# ---------------------------------------------------------------------------
class TestCurrentState:
    def test_current_state_before_any_eval(self):
        gate = _make_gate()
        snap = gate.current_state()
        assert snap["stage"] == 1
        assert snap["weights"] == (1.0, 0.0, 0.0)
        assert snap["eval_len_ratio"] is None
        assert snap["eval_in_range_ratio"] is None
        assert snap["reason"] == "init"

    def test_current_state_after_eval_carries_last_decision(self):
        gate = _make_gate()
        gate.assign_from_eval(_eval(length=200, in_range=0.85))
        snap = gate.current_state()
        assert snap["stage"] == 3
        assert snap["weights"] == (1.0, 1.0, 1.0)
        assert snap["eval_len_ratio"] == pytest.approx(1.0)
        assert snap["eval_in_range_ratio"] == pytest.approx(0.85)
        assert "stage 3" in snap["reason"]


# ---------------------------------------------------------------------------
# Boundary conditions.
# ---------------------------------------------------------------------------
class TestBoundaries:
    def test_exact_pass_len_ratio_qualifies_for_stage2_or_3(self):
        # len_ratio == pass_len_ratio is treated as PASS (>= comparison).
        gate = _make_gate(pass_len_ratio=0.95, pass_in_range=0.80)
        info = gate.assign_from_eval(_eval(length=0.95 * 200, in_range=0.20))
        assert info["stage"] == 2

    def test_exact_pass_in_range_qualifies_for_stage3(self):
        gate = _make_gate(pass_len_ratio=0.95, pass_in_range=0.80)
        info = gate.assign_from_eval(_eval(length=0.99 * 200, in_range=0.80))
        assert info["stage"] == 3

    def test_just_below_pass_len_ratio_drops_to_stage1(self):
        gate = _make_gate(pass_len_ratio=0.95)
        info = gate.assign_from_eval(_eval(length=0.949 * 200, in_range=0.99))
        assert info["stage"] == 1


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
class TestValidation:
    def test_invalid_initial_stage(self):
        with pytest.raises(ValueError):
            CurriculumStageGate(max_steps=10, initial_stage=4)

    def test_zero_max_steps(self):
        with pytest.raises(ValueError):
            CurriculumStageGate(max_steps=0)
