"""Stepping state machine v1 — expected-swing alternating gait schedule.

This is the original version used in the training run
``train_basic_balance_step_ppo_20260824_175705`` (commit 88ca9e58).

It differs from the current ``stepping_state_machine.py`` (v2) in several
ways:

  - **Expected-swing correction**: always pushes ``opposite(last_swing)``
    up and the other foot down, regardless of which foot is *actually*
    airborne.  If the robot lifts the wrong foot, it is corrected toward
    the expected one.  (v2 follows the actual swing/support feet and
    only corrects at DOUBLE transitions.)
  - **Single grace threshold**: ``support_steps < MIN_SUPPORT_STEPS``
    (default 10) → zero weights.  No Phase A/B/C sub-phases.
  - **FLIGHT → zero**: both feet off the ground resets to zero weights.
    (v2 continues the previous state.)
  - **No DOUBLE grace**: DOUBLE immediately pushes expected_swing up
    and the other foot down.  (v2 has a 6-step grace on DOUBLE.)
  - **FOOT_HEIGHT_CLIP = 0.2**: wider saturation than v2's 0.1.

Physical state from ``(contact_l, contact_r)``::

    (T, T) → DOUBLE      both feet down
    (T, F) → SUPPORT_L   left supports,  right swings
    (F, T) → SUPPORT_R   right supports, left swings
    (F, F) → FLIGHT      neither foot down

Bookkeeping, per frame::

    if state is SUPPORT_*:
        current_swing = the airborne foot
        last_swing    = current_swing        # updated unconditionally
        support_steps = support_steps + 1 if same state as previous else 1
    else:
        current_swing = None
        support_steps = 0

    expected_swing = opposite(last_swing)    # None until the first SUPPORT_*

Weights (W = 1.0)::

    initial DOUBLE (last_swing is None)  →  w_L = +W, w_R = +W
    grace (SUPPORT_* and steps < 10)     →  w_L =  0, w_R =  0
    FLIGHT                               →  w_L =  0, w_R =  0
    otherwise                            →  w[expected_swing] = +W
                                            w[other foot]     = -W

Self-correction property
------------------------
``last_swing`` is updated unconditionally to whichever foot is actually
airborne, and ``expected_swing = opposite(last_swing)``.  With only two feet,
if the robot lifts the *wrong* foot then ``opposite(wrong) == expected``, so
``expected_swing`` is unchanged and the robot keeps being pushed toward the
correct foot.  Lifting the same foot twice therefore earns no reward (its
weight was -W in DOUBLE) and needs no special-case branch.

Negative actor_weight relies on the ``!= 0.0`` skip predicate in
``ppo_trainer_v2.ppo_update_v2`` (a channel whose weights are all <= 0 was
previously dropped silently).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# --- Physical states ---
STATE_DOUBLE = "double"
STATE_SUPPORT_L = "support_l"   # left supports, right swings
STATE_SUPPORT_R = "support_r"   # right supports, left swings
STATE_FLIGHT = "flight"

# --- Stepping state machine parameters ---
FOOT_WEIGHT: float = 1.0
"""Base actor_weight magnitude W for the two foot channels."""

MIN_SUPPORT_STEPS: int = 10
"""Single-support must last this many steps (0.5 s @ 20 Hz) before the
switch instruction (lift support foot / lower swing foot) kicks in."""

FOOT_HEIGHT_CLIP: float = 0.2
"""Foot height reward saturation (m).  Lifting beyond this earns nothing
more, preventing a degenerate 'raise the knee as high as possible' policy."""


def compute_foot_weights(
    contact_l: np.ndarray,
    contact_r: np.ndarray,
    T: int,
    weight: float = FOOT_WEIGHT,
    min_support_steps: int = MIN_SUPPORT_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-hoc scan producing per-frame actor weights for both feet.

    Returns ``(w_left, w_right)``, each shape ``(T,)`` float32.
    See the module docstring for the full rule table.
    """
    w_left = np.zeros(T, dtype=np.float32)
    w_right = np.zeros(T, dtype=np.float32)

    last_swing: Optional[str] = None
    prev_state: Optional[str] = None
    support_steps: int = 0

    for t in range(T):
        cl = bool(contact_l[t])
        cr = bool(contact_r[t])

        if cl and cr:
            state = STATE_DOUBLE
        elif cl and not cr:
            state = STATE_SUPPORT_L
        elif cr and not cl:
            state = STATE_SUPPORT_R
        else:
            state = STATE_FLIGHT

        # --- Bookkeeping ---
        if state == STATE_SUPPORT_L:
            current_swing = "right"
        elif state == STATE_SUPPORT_R:
            current_swing = "left"
        else:
            current_swing = None

        if current_swing is not None:
            last_swing = current_swing
            support_steps = support_steps + 1 if state == prev_state else 1
        else:
            support_steps = 0

        expected_swing = None
        if last_swing is not None:
            expected_swing = "right" if last_swing == "left" else "left"

        # --- Weights ---
        if state == STATE_FLIGHT:
            # Neither foot down: don't inject a direction, let r_fall lead.
            pass
        elif current_swing is not None and support_steps < min_support_steps:
            # Grace: single support has not lasted long enough yet.
            pass
        elif last_swing is None:
            # Initial double support, no step taken yet: lift either foot.
            w_left[t] = weight
            w_right[t] = weight
        else:
            # Push the expected swing foot up, the other one down.
            if expected_swing == "left":
                w_left[t] = weight
                w_right[t] = -weight
            else:
                w_left[t] = -weight
                w_right[t] = weight

        prev_state = state

    return w_left, w_right
