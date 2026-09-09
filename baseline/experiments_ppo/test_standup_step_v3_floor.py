"""Unit tests for stepping-based floor scheduling in ``StandupStepV3``.

Covers:
  - Phase 1: floor/coef returned while stepping not yet demonstrated.
  - Trigger: floor disabled after ``step_confirm_updates`` consecutive
    evals where ALL eval episodes (all agents) demonstrate stepping.
  - Phase 2: zero floor/coef after disable, and stays disabled.
  - Hysteresis: a single successful eval does NOT trigger disable.
  - One-way: once disabled, stays disabled even if stepping regresses.
  - Empty episodes do not crash.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3


# ---------------------------------------------------------------------------
# Mock episode builder
# ---------------------------------------------------------------------------

def _make_episode(
    T: int = 200,
    *,
    h_torso_profile: str = "standup_then_balance",
    step_left: bool = True,
    step_right: bool = True,
) -> SimpleNamespace:
    """Build a minimal mock episode for ``on_eval``.

    ``h_torso_profile``:
      - "standup_then_balance": first 50 steps low, rest at 1.2 (plateau)
      - "never_balance": all steps at 0.5 (never stands up)
      - "balance_no_step": stands up but feet don't lift

    ``step_left`` / ``step_right``: if True, the respective foot lifts
    above 0.05m during BALANCE frames.
    """
    h_torso = np.zeros(T, dtype=np.float32)
    h_left = np.zeros(T, dtype=np.float32)
    h_right = np.zeros(T, dtype=np.float32)
    potential = np.zeros(T, dtype=np.float32)

    if h_torso_profile == "standup_then_balance":
        h_torso[:50] = 0.5
        h_torso[50:] = 1.2  # plateau above 1.0 → BALANCE from step 50
        potential[50:] = 0.95
        if step_left:
            h_left[60] = 0.08  # lifts above 0.05 during BALANCE
        if step_right:
            h_right[70] = 0.06
    elif h_torso_profile == "balance_no_step":
        h_torso[:50] = 0.5
        h_torso[50:] = 1.2
        potential[50:] = 0.95
        # feet stay at 0 → no stepping
    elif h_torso_profile == "never_balance":
        h_torso[:] = 0.5
        # potential stays 0

    observer_outputs = {
        "standing_balance_a": {
            "potential": potential,
            "h_torso": h_torso,
        },
        "foot_state_a": {
            "h_left_foot": h_left,
            "h_right_foot": h_right,
        },
        "standing_balance_b": {
            "potential": potential,
            "h_torso": h_torso,
        },
        "foot_state_b": {
            "h_left_foot": h_left,
            "h_right_foot": h_right,
        },
    }

    return SimpleNamespace(
        num_frames=T,
        observer_outputs=observer_outputs,
    )


def _make_stepping_episodes(n: int = 2) -> list:
    """Build ``n`` episodes where all agents step successfully."""
    return [_make_episode() for _ in range(n)]


def _make_no_step_episodes(n: int = 2) -> list:
    """Build ``n`` episodes where agents never reach BALANCE."""
    return [_make_episode(h_torso_profile="never_balance") for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_phase1_returns_configured_floor():
    """Phase 1: exploration() returns the configured floor/coef."""
    e = StandupStepV3()
    spec = e.exploration(1)
    assert spec.uncertainty_floor == e.uncertainty_floor
    assert spec.uncertainty_coef == e.uncertainty_coef


def test_floor_disabled_after_confirm_evals():
    """Floor disabled after step_confirm_updates consecutive successful evals."""
    e = StandupStepV3()
    e.step_confirm_updates = 3
    eps = _make_stepping_episodes()

    # 2 successful evals — should NOT trigger yet.
    e.on_eval(eps, 1)
    assert not e._floor_disabled
    assert e._step_success_streak == 1
    e.on_eval(eps, 2)
    assert not e._floor_disabled
    assert e._step_success_streak == 2

    # 3rd successful eval — should trigger.
    e.on_eval(eps, 3)
    assert e._floor_disabled
    assert e._floor_disabled_at == 3
    assert e._step_success_streak == 3

    # Phase 2: zero floor/coef.
    spec = e.exploration(4)
    assert spec.uncertainty_floor == 0.0
    assert spec.uncertainty_coef == 0.0


def test_single_success_does_not_trigger():
    """A single successful eval followed by failures does NOT trigger."""
    e = StandupStepV3()
    e.step_confirm_updates = 3
    good_eps = _make_stepping_episodes()
    bad_eps = _make_no_step_episodes()

    # 1 success, then 2 failures — streak resets, no trigger.
    e.on_eval(good_eps, 1)
    assert e._step_success_streak == 1
    e.on_eval(bad_eps, 2)
    assert e._step_success_streak == 0
    e.on_eval(bad_eps, 3)
    assert e._step_success_streak == 0
    assert not e._floor_disabled
    assert e.exploration(4).uncertainty_coef == e.uncertainty_coef


def test_partial_step_success_does_not_trigger():
    """If only one foot lifts (not both), that agent fails stepping."""
    e = StandupStepV3()
    e.step_confirm_updates = 3
    # Only left foot lifts — right stays down → stepping fails
    eps = [_make_episode(step_right=False) for _ in range(2)]

    e.on_eval(eps, 1)
    assert e._step_success_streak == 0
    assert not e._floor_disabled


def test_disabled_is_one_way():
    """Once disabled, stays disabled even if stepping regresses."""
    e = StandupStepV3()
    e.step_confirm_updates = 2
    good_eps = _make_stepping_episodes()
    bad_eps = _make_no_step_episodes()

    # Trigger disable.
    e.on_eval(good_eps, 1)
    e.on_eval(good_eps, 2)
    assert e._floor_disabled

    # Even with bad episodes, floor stays disabled.
    e.on_eval(bad_eps, 3)
    e.on_eval(bad_eps, 4)
    assert e._floor_disabled
    spec = e.exploration(5)
    assert spec.uncertainty_floor == 0.0
    assert spec.uncertainty_coef == 0.0


def test_balance_no_step_does_not_trigger():
    """Reaching BALANCE but not lifting feet does NOT count as stepping."""
    e = StandupStepV3()
    e.step_confirm_updates = 3
    eps = [_make_episode(h_torso_profile="balance_no_step") for _ in range(2)]

    for u in range(1, 6):
        e.on_eval(eps, u)
    assert not e._floor_disabled
    assert e._step_success_streak == 0


def test_empty_episodes_does_not_crash():
    """on_eval must tolerate empty episodes."""
    e = StandupStepV3()
    e.on_eval([], 1)
    assert not e._floor_disabled


def test_zero_frame_episode_does_not_crash():
    """on_eval must tolerate episodes with 0 frames."""
    e = StandupStepV3()
    ep = SimpleNamespace(
        num_frames=0,
        observer_outputs={},
    )
    e.on_eval([ep], 1)
    assert not e._floor_disabled


def test_on_update_no_longer_disables():
    """on_update only records uncertainty; does not disable floor."""
    e = StandupStepV3()
    # High uncertainty for many updates — should NOT disable floor.
    for u in range(1, 20):
        e.on_update(SimpleNamespace(policy_stats={"uncertainty": 0.99}), u)
    assert not e._floor_disabled
    assert e.exploration(20).uncertainty_coef == e.uncertainty_coef


def test_on_update_missing_uncertainty_does_not_crash():
    """on_update must tolerate missing 'uncertainty' key (defensive)."""
    e = StandupStepV3()
    e.on_update(SimpleNamespace(policy_stats={}), 1)
    assert not e._floor_disabled


def test_state_round_trip():
    """state() / load_state() preserve floor scheduling state."""
    e = StandupStepV3()
    e.step_confirm_updates = 2
    good_eps = _make_stepping_episodes()
    e.on_eval(good_eps, 1)
    e.on_eval(good_eps, 2)
    assert e._floor_disabled

    state = e.state()
    assert state["floor_disabled"] is True
    assert state["floor_disabled_at"] == 2
    assert state["step_success_streak"] == 2

    e2 = StandupStepV3()
    e2.load_state(state)
    assert e2._floor_disabled is True
    assert e2._floor_disabled_at == 2
    assert e2._step_success_streak == 2
    spec = e2.exploration(3)
    assert spec.uncertainty_floor == 0.0
    assert spec.uncertainty_coef == 0.0


if __name__ == "__main__":
    test_phase1_returns_configured_floor()
    print("test_phase1_returns_configured_floor: PASS")
    test_floor_disabled_after_confirm_evals()
    print("test_floor_disabled_after_confirm_evals: PASS")
    test_single_success_does_not_trigger()
    print("test_single_success_does_not_trigger: PASS")
    test_partial_step_success_does_not_trigger()
    print("test_partial_step_success_does_not_trigger: PASS")
    test_disabled_is_one_way()
    print("test_disabled_is_one_way: PASS")
    test_balance_no_step_does_not_trigger()
    print("test_balance_no_step_does_not_trigger: PASS")
    test_empty_episodes_does_not_crash()
    print("test_empty_episodes_does_not_crash: PASS")
    test_zero_frame_episode_does_not_crash()
    print("test_zero_frame_episode_does_not_crash: PASS")
    test_on_update_no_longer_disables()
    print("test_on_update_no_longer_disables: PASS")
    test_on_update_missing_uncertainty_does_not_crash()
    print("test_on_update_missing_uncertainty_does_not_crash: PASS")
    test_state_round_trip()
    print("test_state_round_trip: PASS")
