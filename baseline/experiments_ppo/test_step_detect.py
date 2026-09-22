"""Unit tests for ``detect_step_cycles`` — eval-time step detector.

Covers:
  - A clean alternating gait produces cycles + alt_ratio=1.
  - Contact jitter (1-2 frame flaps) does not produce swings.
  - A hop (both feet airborne) is never a step.
  - A swing that ends in a fall (landing not standing) is rejected.
  - A swing that never reaches h_thresh counts as attempt, not cycle.
  - "stepped" requires >=1 valid cycle on EACH foot.
"""
from __future__ import annotations

import numpy as np

from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    detect_step_cycles,
    single_support_mask,
)


def _seq(T: int, events: list) -> tuple:
    """Build (contact_l, contact_r, h_left, h_right) from events.

    events: list of (start, end, foot, h_peak).  During [start, end) the
    foot is airborne (contact False) with height ramping to h_peak.
    """
    cl = np.ones(T, dtype=bool)
    cr = np.ones(T, dtype=bool)
    hl = np.zeros(T, dtype=np.float32)
    hr = np.zeros(T, dtype=np.float32)
    for a, b, foot, hp in events:
        if foot == "left":
            cl[a:b] = False
            hl[a:b] = np.linspace(0.0, hp, b - a)
        else:
            cr[a:b] = False
            hr[a:b] = np.linspace(0.0, hp, b - a)
    return cl, cr, hl, hr


def test_clean_alternating_gait():
    T = 200
    cl, cr, hl, hr = _seq(T, [
        (50, 60, "left", 0.08),
        (70, 80, "right", 0.08),
        (90, 100, "left", 0.08),
        (110, 120, "right", 0.08),
    ])
    standing = np.ones(T, dtype=bool)
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    assert r["n_cycles_left"] == 2
    assert r["n_cycles_right"] == 2
    assert r["n_swings"] == 4
    assert r["alt_ratio"] == 1.0
    assert r["stepped"] is True
    assert abs(r["h_swing_max"] - 0.08) < 1e-6


def test_jitter_is_not_a_step():
    T = 200
    cl, cr, hl, hr = _seq(T, [])
    # 2-frame contact flap on left foot — below min_air_steps=3 AND
    # below the 4-frame debounce hold.
    cl[100:102] = False
    hl[100:102] = 0.10
    standing = np.ones(T, dtype=bool)
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    assert r["n_swings"] == 0
    assert r["stepped"] is False


def test_hop_is_not_a_step():
    T = 200
    cl, cr, hl, hr = _seq(T, [])
    # Both feet airborne for 8 frames — a hop, not a step.
    cl[60:68] = False
    cr[60:68] = False
    hl[60:68] = 0.10
    hr[60:68] = 0.10
    standing = np.ones(T, dtype=bool)
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    assert r["n_swings"] == 0
    assert len(r["cycles"]) == 0
    assert r["stepped"] is False


def test_fall_landing_rejected():
    T = 200
    cl, cr, hl, hr = _seq(T, [(50, 60, "left", 0.08)])
    standing = np.ones(T, dtype=bool)
    standing[58:] = False  # robot fell mid-swing / on landing
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    # Lift-off was standing, landing was not → attempt but no cycle.
    assert r["n_swings"] == 1
    assert len(r["cycles"]) == 0
    assert r["stepped"] is False


def test_shallow_swing_is_attempt_only():
    T = 200
    cl, cr, hl, hr = _seq(T, [(50, 60, "left", 0.03)])  # 3cm < 5cm thresh
    standing = np.ones(T, dtype=bool)
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    assert r["n_swings"] == 1
    assert len(r["cycles"]) == 0
    assert abs(r["h_swing_max"] - 0.03) < 1e-6
    assert r["stepped"] is False


def test_single_foot_only_not_stepped():
    T = 200
    cl, cr, hl, hr = _seq(T, [
        (50, 60, "left", 0.08),
        (80, 90, "left", 0.08),
    ])
    standing = np.ones(T, dtype=bool)
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    assert r["n_cycles_left"] == 2
    assert r["n_cycles_right"] == 0
    assert r["alt_ratio"] == 0.0
    assert r["stepped"] is False


def test_nonstanding_liftoff_rejected():
    T = 200
    cl, cr, hl, hr = _seq(T, [(20, 30, "left", 0.08), (150, 160, "right", 0.08)])
    standing = np.ones(T, dtype=bool)
    standing[:100] = False  # first lift happens before standing
    r = detect_step_cycles(cl, cr, hl, hr, standing)
    assert r["n_cycles_left"] == 0
    assert r["n_cycles_right"] == 1
    assert r["stepped"] is False  # only one foot has a valid cycle


# ----------------------------------------------------------------------
# Phase C lift gate — a swing foot that never reached the threshold must
# keep getting +W late in the phase, not unconditional -W.
# ----------------------------------------------------------------------

def _long_swing(T: int, h_during_swing) -> tuple:
    """DOUBLE for 10 frames, then left foot airborne for 20 frames
    (SUPPORT_R: support_steps 1..20, Phase C begins at step 11 → t=20)."""
    cl = np.ones(T, dtype=bool)
    cr = np.ones(T, dtype=bool)
    hl = np.zeros(T, dtype=np.float32)
    hr = np.zeros(T, dtype=np.float32)
    cl[10:30] = False                      # left airborne t=10..29
    hl[10:30] = h_during_swing
    return cl, cr, hl, hr


def test_phase_c_shallow_swing_keeps_lift_weight():
    T = 60
    cl, cr, hl, hr = _long_swing(T, np.full(20, 0.01, dtype=np.float32))
    wl, wr = compute_foot_weights(cl, cr, T, h_left=hl, h_right=hr)
    # Phase C (t=20..29): h < threshold → +W (keep pushing up)
    assert np.all(wl[20:30] > 0)
    assert np.all(wr[20:30] == 0)


def test_phase_c_lifted_swing_gets_press_weight():
    T = 60
    cl, cr, hl, hr = _long_swing(T, np.full(20, 0.08, dtype=np.float32))
    wl, wr = compute_foot_weights(cl, cr, T, h_left=hl, h_right=hr)
    # Phase C (t=20..29): h >= threshold → -W (come down)
    assert np.all(wl[20:30] < 0)
    assert np.all(wr[20:30] == 0)


def test_phase_c_late_lift_not_punished():
    """Foot crosses threshold mid-Phase-C: +W while rising, -W at crest."""
    T = 60
    h = np.full(20, 0.01, dtype=np.float32)
    h[15:] = 0.08                          # crosses at t=25 (one rising step)
    cl, cr, hl, hr = _long_swing(T, h)
    wl, wr = compute_foot_weights(cl, cr, T, h_left=hl, h_right=hr)
    assert np.all(wl[20:26] > 0)           # below bar / rising → keep lifting
    assert np.all(wl[26:30] < 0)           # crested above bar → descend


def test_phase_c_rising_apex_keeps_lift_weight():
    """A swing foot above threshold but still rising is not pushed down."""
    T = 60
    # rises ~4 mm/frame through the whole swing — crosses the bar mid-C
    h = (0.02 + 0.004 * np.arange(20)).astype(np.float32)   # 0.02 → 0.096
    cl, cr, hl, hr = _long_swing(T, h)
    wl, wr = compute_foot_weights(cl, cr, T, h_left=hl, h_right=hr)
    # Phase C (t=20..29): h crosses threshold but keeps rising → +W
    assert np.all(wl[20:30] > 0)
    assert np.all(wr[20:30] == 0)


def test_phase_c_no_height_data_stays_press():
    """Backward compat: without h arrays Phase C is unconditional -W."""
    T = 60
    cl, cr, _, _ = _long_swing(T, np.zeros(20, dtype=np.float32))
    wl, wr = compute_foot_weights(cl, cr, T)
    assert np.all(wl[20:30] < 0)


# ----------------------------------------------------------------------
# single_support_mask — debounced exactly-one-foot-down mask used to
# exempt commanded swings from the r_potential actor-weight gate.
# ----------------------------------------------------------------------

def test_single_support_mask_basic():
    T = 40
    cl = np.ones(T, dtype=bool); cr = np.ones(T, dtype=bool)
    cl[10:20] = False                       # left airborne → SUPPORT_R
    m = single_support_mask(cl, cr)
    assert m.dtype == bool
    assert np.all(m[10:20])                 # single-support frames True
    assert not np.any(m[:10])               # DOUBLE → False
    assert not np.any(m[20:])


def test_single_support_mask_excludes_flight_and_jitter():
    T = 40
    cl = np.ones(T, dtype=bool); cr = np.ones(T, dtype=bool)
    cl[5:13] = False; cr[5:13] = False      # FLIGHT — not single support
    cl[25:27] = False                       # 2-frame jitter (< debounce)
    m = single_support_mask(cl, cr)
    assert not np.any(m[5:13])              # hop is not a commanded swing
    assert not np.any(m[25:27])             # jitter absorbed by debounce


# ----------------------------------------------------------------------
# step_cycle_bonus — detected cycles add reward over the airborne window
# ----------------------------------------------------------------------

def _make_step_episode(T: int, events: list):
    """Minimal Episode with the observer fields exp_step reads."""
    from baseline.framework.rollout.episode import Episode

    cl, cr, hl, hr = _seq(T, events)
    standing = np.ones(T, dtype=np.float32)
    obs = np.zeros((T, 96), dtype=np.float32)
    obs[:, 45] = 1.2                       # h_torso ≥ 1.1 → standing
    acts = np.zeros((T, 21), dtype=np.float32)
    foot = {
        "h_left_foot": hl, "h_right_foot": hr,
        "left_foot_contact": cl.astype(np.float32),
        "right_foot_contact": cr.astype(np.float32),
    }
    balance = {"potential": standing}
    return Episode(
        base_seed=42, episode_index=0, blueprint_hash="t",
        num_frames=T, episode_options={},
        agent_termination_proposal_records={},
        observations={"robot_a": obs, "robot_b": obs.copy()},
        actions={"robot_a": acts, "robot_b": acts.copy()},
        action_extras={"robot_a": {}, "robot_b": {}},
        explore_factors={"robot_a": np.zeros(T, np.float32),
                         "robot_b": np.zeros(T, np.float32)},
        observer_outputs={
            "foot_state_a": foot, "foot_state_b": dict(foot),
            "standing_balance_a": balance,
            "standing_balance_b": dict(balance),
        },
        final_observation={"robot_a": obs[-1], "robot_b": obs[-1].copy()},
        episode_metrics={},
    )


def test_step_cycle_bonus_adds_reward_on_airborne_window():
    from baseline.experiments_ppo.exp_step import Step

    T = 80
    # two valid cycles: left swing 10..20 (peak .08), right 40..50
    ep = _make_step_episode(
        T, [(10, 20, "left", 0.08), (40, 50, "right", 0.08)])
    exp = Step()
    trajs = exp._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    assert len(trajs) == 1
    ch = trajs[0].channels

    rl = ch["r_left_foot"].reward
    # dense clip(h,0,.05) over the ramp + bonus .5/10 = .05/frame
    expected_l = np.clip(np.linspace(0.0, 0.08, 10), 0, 0.05) + 0.05
    np.testing.assert_allclose(rl[10:20], expected_l, rtol=1e-5, atol=1e-6)
    assert rl[:10].max() == 0.0 and rl[20:].max() < 0.05

    rr = ch["r_right_foot"].reward
    np.testing.assert_allclose(rr[40:50], expected_l, rtol=1e-5, atol=1e-6)
    assert rr[:40].max() < 0.05


def test_step_cycle_bonus_skips_invalid_swings():
    from baseline.experiments_ppo.exp_step import Step

    T = 60
    # shallow swing (peak .02 < .05) — attempt only, no bonus
    ep = _make_step_episode(T, [(10, 20, "left", 0.02)])
    exp = Step()
    trajs = exp._build_agent_trajectory(
        ep, "robot_a", "foot_state_a", "standing_balance_a")
    rl = trajs[0].channels["r_left_foot"].reward
    # dense term only (clip of the ramped height), no bonus
    assert rl[10:20].max() <= 0.02 + 1e-6


def test_clock_foot_weights_alternating():
    """Clock commands: left window +W, right window +W, support -W."""
    from baseline.humanoid21.end2end.stepping_state_machine import (
        clock_foot_weights,
    )
    T, P = 80, 40
    wl, wr = clock_foot_weights(T, period=P)
    # frames 0..19 = left window (cmd_L), 20..39 = right window
    assert (wl[0:15] > 0).all()          # left lift phase
    assert (wl[15:20] < 0).all()         # left land phase (τ>=0.75)
    assert (wr[0:20] < 0).all()          # right is support
    assert (wr[20:35] > 0).all()         # right lift phase
    assert (wl[20:40] < 0).all()         # left is support
    # cycle repeats
    assert (wl[40:55] > 0).all()


def test_clock_foot_weights_apex_coast():
    """Commanded foot already high & descending mid-window -> 0 (coast)."""
    from baseline.humanoid21.end2end.stepping_state_machine import (
        clock_foot_weights,
    )
    T, P = 40, 40
    hl = np.zeros(T); hr = np.zeros(T)
    # left window is frames 0..19; make h high & descending at t=5
    hl[:] = 0.0
    hl[4] = 0.06; hl[5] = 0.05  # above thresh, falling -> coast (0)
    wl, wr = clock_foot_weights(
        T, h_left=hl, h_right=hr, period=P)
    assert wl[4] > 0       # rising into apex
    assert wl[5] == 0      # high + descending mid-window -> coast
    assert wl[16] < 0      # land phase
    assert wr[5] < 0       # support always pressed
