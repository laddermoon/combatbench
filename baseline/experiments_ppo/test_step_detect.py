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
    detect_step_cycles,
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
