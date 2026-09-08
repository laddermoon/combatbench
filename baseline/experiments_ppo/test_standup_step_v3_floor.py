"""Unit tests for the two-stage uncertainty floor scheduling in
``StandupStepV3``.

Covers:
  - Phase 1: floor/coef returned while uncertainty is below floor.
  - Trigger: floor disabled after `floor_confirm_updates` consecutive
    updates at/above floor.
  - Phase 2: zero floor/coef after disable, and stays disabled.
  - Hysteresis: a single spike above floor does NOT trigger disable.
"""
from __future__ import annotations

from types import SimpleNamespace

from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3


def _stats(uncertainty: float):
    """Build a minimal stand-in for ``UpdateStats`` with only the fields
    the experiment's ``on_update`` reads."""
    return SimpleNamespace(policy_stats={"uncertainty": uncertainty})


def test_phase1_returns_configured_floor():
    e = StandupStepV3()
    spec = e.exploration(1)
    assert spec.uncertainty_floor == e.uncertainty_floor
    assert spec.uncertainty_coef == e.uncertainty_coef


def test_floor_disabled_after_confirm_window():
    e = StandupStepV3()
    e.uncertainty_floor = 0.35
    e.floor_confirm_updates = 5
    floor = e.uncertainty_floor

    # Below floor for several updates — should stay active.
    for u in range(1, 11):
        e.on_update(_stats(floor - 0.05), u)
    assert not e._floor_disabled
    assert e.exploration(11).uncertainty_coef == e.uncertainty_coef

    # Cross floor for (confirm-1) updates — should NOT trigger yet.
    for u in range(11, 11 + e.floor_confirm_updates - 1):
        e.on_update(_stats(floor + 0.01), u)
    assert not e._floor_disabled

    # One more above-floor update — should trigger.
    trigger_u = 11 + e.floor_confirm_updates - 1
    e.on_update(_stats(floor + 0.01), trigger_u + 1)
    assert e._floor_disabled
    assert e._floor_disabled_at == trigger_u + 1

    # Phase 2: zero floor/coef.
    spec = e.exploration(trigger_u + 2)
    assert spec.uncertainty_floor == 0.0
    assert spec.uncertainty_coef == 0.0


def test_single_spike_does_not_trigger():
    e = StandupStepV3()
    e.uncertainty_floor = 0.35
    e.floor_confirm_updates = 5
    floor = e.uncertainty_floor

    # 4 below, 1 spike above, 4 below — should not trigger.
    for u in range(1, 5):
        e.on_update(_stats(floor - 0.05), u)
    e.on_update(_stats(floor + 0.10), 5)
    for u in range(6, 10):
        e.on_update(_stats(floor - 0.05), u)
    assert not e._floor_disabled
    assert e.exploration(10).uncertainty_coef == e.uncertainty_coef


def test_disabled_is_one_way():
    e = StandupStepV3()
    e.uncertainty_floor = 0.35
    e.floor_confirm_updates = 3
    floor = e.uncertainty_floor

    # Trigger disable.
    for u in range(1, 4):
        e.on_update(_stats(floor + 0.01), u)
    assert e._floor_disabled

    # Even if uncertainty drops well below floor, floor stays disabled.
    for u in range(4, 20):
        e.on_update(_stats(floor - 0.20), u)
    assert e._floor_disabled
    spec = e.exploration(20)
    assert spec.uncertainty_floor == 0.0
    assert spec.uncertainty_coef == 0.0


def test_empty_stats_does_not_crash():
    """on_update must tolerate missing 'uncertainty' key (defensive)."""
    e = StandupStepV3()
    e.on_update(SimpleNamespace(policy_stats={}), 1)
    assert not e._floor_disabled
