"""Pin the data-driven curriculum gate's stage transitions.

Covers:

  * initial state stays at stage 1 during the dwell window even when
    metrics already pass — the gate is intentionally hesitant early on
    to avoid premature promotion;
  * once dwell elapses, passing thresholds promote stage 1 -> 2;
  * crossing the FAIL hysteresis band on stage-1 metrics demotes back
    to stage 1 from any higher stage;
  * mid-band metrics (between PASS and FAIL) do NOT cause oscillation;
  * stage 3 promotion requires the in-range condition;
  * weight schedule is exactly (1,0,0) / (1,1,0) / (1,1,1).
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
        pass_term_rate=0.05,
        fail_term_rate=0.15,
        pass_len_ratio=0.95,
        fail_len_ratio=0.85,
        pass_in_range=0.80,
        fail_in_range=0.60,
        window=2,
        min_dwell=3,
    )
    defaults.update(overrides)
    return CurriculumStageGate(**defaults)


def _passing_summary(max_steps=200):
    return {"term_rate": 0.0, "mean_length": 1.0 * max_steps, "in_range_ratio": 0.0}


def _failing_summary(max_steps=200):
    return {"term_rate": 0.5, "mean_length": 0.5 * max_steps, "in_range_ratio": 0.0}


def _approach_passing_summary(max_steps=200):
    return {"term_rate": 0.0, "mean_length": 1.0 * max_steps, "in_range_ratio": 0.95}


# ---------------------------------------------------------------------------
# Initial weights & weight schedule.
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
# Dwell time.
# ---------------------------------------------------------------------------
class TestMinDwell:
    def test_no_promotion_within_dwell_even_when_passing(self):
        gate = _make_gate(min_dwell=3)
        for _ in range(2):  # 2 < min_dwell=3
            info = gate.update(_passing_summary())
            assert info["stage"] == 1
            assert info["weights"] == (1.0, 0.0, 0.0)
            assert "dwell" in info["reason"]

    def test_promotion_after_dwell_elapses(self):
        gate = _make_gate(min_dwell=3, window=2)
        for _ in range(2):
            gate.update(_passing_summary())
        # 3rd call: dwell satisfied, average of last 2 passing summaries
        # is also passing, so promote 1 -> 2.
        info = gate.update(_passing_summary())
        assert info["stage"] == 2
        assert info["prev_stage"] == 1
        assert info["weights"] == (1.0, 1.0, 0.0)
        # dwell counter resets on transition.
        assert info["dwell"] == 0


# ---------------------------------------------------------------------------
# Hysteresis: PASS -> FAIL band, no flapping in the mid zone.
# ---------------------------------------------------------------------------
class TestHysteresis:
    def test_demotion_from_stage2_when_stage1_violated(self):
        gate = _make_gate(min_dwell=2, window=1)
        # Drive into stage 2.
        gate.update(_passing_summary())
        info = gate.update(_passing_summary())
        assert info["stage"] == 2
        # Now imbalance regresses past FAIL.
        for _ in range(2):
            info = gate.update(_failing_summary())
        assert info["stage"] == 1
        assert "regressed" in info["reason"]

    def test_mid_band_does_not_oscillate(self):
        gate = _make_gate(min_dwell=1, window=1)
        # Push into stage 2.
        gate.update(_passing_summary())
        info = gate.update(_passing_summary())
        assert info["stage"] == 2
        # Mid-band metric: term_rate=0.10 is between 0.05 (pass) and
        # 0.15 (fail); len_ratio=0.90 is between 0.85 (fail) and
        # 0.95 (pass). Neither triggers stage1_violated, neither
        # triggers stage1_ok, so we stay at 2.
        mid_band = {
            "term_rate": 0.10,
            "mean_length": 0.90 * 200,
            "in_range_ratio": 0.0,
        }
        for _ in range(5):
            info = gate.update(mid_band)
        assert info["stage"] == 2

    def test_no_promotion_to_stage2_when_only_one_metric_passes(self):
        gate = _make_gate(min_dwell=1, window=1)
        # term_rate passes (0.0) but len_ratio (0.5 < 0.95) does not.
        partial = {"term_rate": 0.0, "mean_length": 0.5 * 200,
                   "in_range_ratio": 0.0}
        for _ in range(5):
            info = gate.update(partial)
        assert info["stage"] == 1


# ---------------------------------------------------------------------------
# Stage 3.
# ---------------------------------------------------------------------------
class TestStage3:
    def test_stage2_to_stage3_requires_in_range(self):
        gate = _make_gate(min_dwell=1, window=1)
        # Reach stage 2.
        gate.update(_passing_summary())
        info = gate.update(_passing_summary())
        assert info["stage"] == 2
        # stage 1 still passing but in_range high enough -> promote.
        info = gate.update(_approach_passing_summary())
        assert info["stage"] == 3
        assert info["weights"] == (1.0, 1.0, 1.0)

    def test_stage3_to_stage1_when_balance_regresses(self):
        gate = _make_gate(min_dwell=1, window=1)
        gate.update(_passing_summary())
        gate.update(_passing_summary())
        gate.update(_approach_passing_summary())
        assert gate.stage == 3
        # Imbalance regression jumps straight back to stage 1.
        info = gate.update(_failing_summary())
        assert info["stage"] == 1
        assert "regressed" in info["reason"]


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
