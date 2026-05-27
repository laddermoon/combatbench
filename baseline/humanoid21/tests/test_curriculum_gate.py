"""Pin the eval-driven curriculum gate's stage classification.

The gate (rewritten 2026-05-10) is a stateless single-shot classifier
keyed on two scalars from the deterministic eval batch:

  * ``mean_length`` — average ``num_steps`` over the batch.
  * ``final_in_zone_ratio`` — fraction of episodes whose LAST step has
    both distance in [dist_min, dist_max] AND heading angle within the
    OpponentRelationRewarder's ``heading_max_angle_deg``.

Decision rule::

    if mean_length / max_steps < pass_len_ratio:
        stage = 1
    elif final_in_zone_ratio < pass_final_in_zone:
        stage = 2
    else:
        stage = 3

Tests cover: weight schedule, classification of all three regimes,
arbitrary jumps, stale-eval freedom, ``current_state()`` snapshot, and
input validation.
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
    # Tests pin the *gate logic*, not the package-level pass thresholds
    # (which may drift to absorb empirical findings). Use values that
    # make the assertions in this file unambiguous.
    defaults = dict(
        max_steps=200,
        pass_len_ratio=0.98,
        pass_final_in_zone=0.5,
        stage3_sticky_len_ratio=0.70,
    )
    defaults.update(overrides)
    return CurriculumStageGate(**defaults)


def _eval(*, length, final_in_zone=0.0, max_steps=200):
    return {
        "mean_length": length,
        "final_in_zone_ratio": final_in_zone,
        "max_steps": max_steps,
    }


# ---------------------------------------------------------------------------
# Weight schedule.
# ---------------------------------------------------------------------------
class TestWeightSchedule:
    def test_initial_stage1_weights(self):
        gate = _make_gate()
        assert gate.stage == 1
        # 4 components (r_fall, r1, r2, r3); stage 1 splits across the
        # two active components (r_fall, r1) summing to 1.
        assert gate.weights == pytest.approx((0.5, 0.5, 0.0, 0.0))

    def test_stage_weights_table(self):
        # Active flags before normalization. The ``weights`` property
        # then renormalizes the active components to sum to 1.
        assert CurriculumStageGate.STAGE_WEIGHTS[1] == (1.0, 1.0, 0.0, 0.0)
        assert CurriculumStageGate.STAGE_WEIGHTS[2] == (1.0, 1.0, 1.0, 0.0)
        assert CurriculumStageGate.STAGE_WEIGHTS[3] == (1.0, 1.0, 1.0, 1.0)

    def test_normalized_weights_sum_to_one(self):
        for stage in (1, 2, 3):
            gate = _make_gate(initial_stage=stage)
            assert sum(gate.weights) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Single-shot classification: all three regimes.
# ---------------------------------------------------------------------------
class TestSingleShotClassification:
    def test_below_pass_len_ratio_picks_stage1(self):
        gate = _make_gate(initial_stage=3)
        info = gate.assign_from_eval(_eval(length=0.5 * 200, final_in_zone=0.99))
        assert info["stage"] == 1
        assert info["weights"] == pytest.approx((0.5, 0.5, 0.0, 0.0))
        assert info["prev_stage"] == 3
        assert "stage 1" in info["reason"]

    def test_balance_ok_but_low_final_in_zone_picks_stage2(self):
        gate = _make_gate()
        info = gate.assign_from_eval(_eval(length=0.99 * 200, final_in_zone=0.30))
        assert info["stage"] == 2
        third = 1.0 / 3.0
        assert info["weights"] == pytest.approx((third, third, third, 0.0))
        assert "stage 2" in info["reason"]

    def test_balance_and_final_in_zone_both_ok_picks_stage3(self):
        gate = _make_gate()
        info = gate.assign_from_eval(_eval(length=0.99 * 200, final_in_zone=0.85))
        assert info["stage"] == 3
        assert info["weights"] == pytest.approx((0.25, 0.25, 0.25, 0.25))
        assert "stage 3" in info["reason"]


# ---------------------------------------------------------------------------
# Arbitrary jumps — no fixed transition graph.
# ---------------------------------------------------------------------------
class TestArbitraryJumps:
    def test_stage1_to_stage3_in_one_eval(self):
        gate = _make_gate()
        info = gate.assign_from_eval(_eval(length=0.99 * 200, final_in_zone=0.90))
        assert info["prev_stage"] == 1
        assert info["stage"] == 3

    def test_stage3_to_stage1_in_one_eval(self):
        gate = _make_gate(initial_stage=3)
        info = gate.assign_from_eval(_eval(length=0.5 * 200, final_in_zone=0.99))
        assert info["prev_stage"] == 3
        assert info["stage"] == 1

    def test_eval_can_oscillate_freely(self):
        gate = _make_gate()
        # Sequences chosen so each step's decision is unambiguous even
        # under stage-3 stickiness: 100 < sticky_len_ratio*200=140, so
        # the second step is below the sticky floor and demotes.
        seq = [
            (200, 0.30, 2),
            (100, 0.99, 1),  # below stage-3 sticky floor regardless of prev
            (200, 0.99, 3),
            (50, 0.50, 1),
        ]
        for length, fiz, expected in seq:
            info = gate.assign_from_eval(_eval(length=length, final_in_zone=fiz))
            assert info["stage"] == expected, (
                f"length={length} final_in_zone={fiz} "
                f"expected stage {expected}, got {info['stage']}"
            )


# ---------------------------------------------------------------------------
# Stage-3 stickiness — the demotion semantics that distinguish "opponent
# landed hits" from "balance regressed".
# ---------------------------------------------------------------------------
class TestStage3Stickiness:
    def test_stage3_stays_when_combat_still_strong(self):
        """Empirical case from curriculum_20260511_143835 u910:
        eval_length=190.62 final_in_zone=1.000 — clearly combat-capable;
        old logic demoted to Stage 1 (length<196). New sticky logic stays.
        """
        gate = _make_gate(initial_stage=3)
        info = gate.assign_from_eval(_eval(length=190.62, final_in_zone=1.0))
        assert info["stage"] == 3
        assert "sticky" in info["reason"]

    def test_stage3_stays_at_sticky_floor(self):
        gate = _make_gate(initial_stage=3)
        # 0.70 * 200 = 140 — exactly at the floor
        info = gate.assign_from_eval(_eval(length=140, final_in_zone=0.95))
        assert info["stage"] == 3

    def test_stage3_demotes_when_length_collapses(self):
        """Below sticky floor — catastrophic, demote regardless of final_in_zone."""
        gate = _make_gate(initial_stage=3)
        info = gate.assign_from_eval(_eval(length=120, final_in_zone=1.0))
        assert info["stage"] == 1
        assert "stage 1" in info["reason"]

    def test_stage3_demotes_when_combat_skill_regresses(self):
        """Length OK but final_in_zone collapsed — stage 2, not sticky stage 3."""
        gate = _make_gate(initial_stage=3)
        info = gate.assign_from_eval(_eval(length=200, final_in_zone=0.20))
        assert info["stage"] == 2

    def test_stickiness_does_not_apply_to_stage1_or_stage2(self):
        # From stage 1 with length=180 < pass_len, still stage 1.
        gate = _make_gate(initial_stage=1)
        info = gate.assign_from_eval(_eval(length=180, final_in_zone=1.0))
        assert info["stage"] == 1, "stickiness must NOT apply from stage 1"

        # From stage 2 with length=180 < pass_len, still stage 1.
        gate = _make_gate(initial_stage=2)
        info = gate.assign_from_eval(_eval(length=180, final_in_zone=1.0))
        assert info["stage"] == 1, "stickiness must NOT apply from stage 2"


# ---------------------------------------------------------------------------
# current_state() — snapshot for non-eval updates.
# ---------------------------------------------------------------------------
class TestCurrentState:
    def test_current_state_before_any_eval(self):
        gate = _make_gate()
        snap = gate.current_state()
        assert snap["stage"] == 1
        assert snap["weights"] == pytest.approx((0.5, 0.5, 0.0, 0.0))
        assert snap["eval_len_ratio"] is None
        assert snap["eval_final_in_zone_ratio"] is None
        assert snap["reason"] == "init"

    def test_current_state_after_eval_carries_last_decision(self):
        gate = _make_gate()
        gate.assign_from_eval(_eval(length=0.99 * 200, final_in_zone=0.85))
        snap = gate.current_state()
        assert snap["stage"] == 3
        assert snap["weights"] == pytest.approx((0.25, 0.25, 0.25, 0.25))
        assert snap["eval_len_ratio"] == pytest.approx(0.99)
        assert snap["eval_final_in_zone_ratio"] == pytest.approx(0.85)
        assert "stage 3" in snap["reason"]


# ---------------------------------------------------------------------------
# Boundary conditions.
# ---------------------------------------------------------------------------
class TestBoundaries:
    def test_exact_pass_len_ratio_qualifies_for_stage2_or_3(self):
        gate = _make_gate(pass_len_ratio=1.0, pass_final_in_zone=0.5)
        info = gate.assign_from_eval(_eval(length=200, final_in_zone=0.20))
        assert info["stage"] == 2

    def test_exact_pass_final_in_zone_qualifies_for_stage3(self):
        gate = _make_gate(pass_len_ratio=1.0, pass_final_in_zone=0.5)
        info = gate.assign_from_eval(_eval(length=200, final_in_zone=0.5))
        assert info["stage"] == 3

    def test_just_below_pass_len_ratio_drops_to_stage1(self):
        gate = _make_gate(pass_len_ratio=1.0)
        info = gate.assign_from_eval(_eval(length=199, final_in_zone=1.0))
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
