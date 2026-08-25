"""Stepping state machine for humanoid21 foot-height reward channels.

Post-hoc scan over the per-frame contact sequence (contact_l, contact_r)
that produces per-frame actor weights for the left and right foot reward
channels.  The weights encode a gait schedule that encourages alternating
steps with a grace period on double support.

Physical state from ``(contact_l, contact_R)``::

    (T, T) → DOUBLE      both feet down
    (T, F) → SUPPORT_L   left supports,  right swings
    (F, T) → SUPPORT_R   right supports, left swings
    (F, F) → FLIGHT      neither foot down

Bookkeeping, per frame::

    if state is SUPPORT_*:
        current_swing = the airborne foot
        last_swing    = current_swing        # updated unconditionally
        support_steps = support_steps + 1 if same state as previous else 1
    elif state is DOUBLE:
        double_steps  = double_steps + 1 if same state as previous else 1
    else:
        current_swing = None
        support_steps = 0
        double_steps  = 0

Weights (W = 1.0)::

    initial DOUBLE (last_swing is None)
        steps 1..6   (grace)              →  w_L =  0, w_R =  0
        steps 7+                           →  w_L = +W, w_R = +W
    FLIGHT                               →  continues previous state
    SUPPORT_*  steps 1..2   (Phase A)    →  w[support] = -W
                                            w[swing]   = +W if h_swing < SWING_LIFT_THRESHOLD else 0
    SUPPORT_*  steps 3..10  (Phase B)    →  w[swing]   = +W if h_swing < SWING_LIFT_THRESHOLD else 0
                                            w[support] =  0
    SUPPORT_*  steps 11+    (Phase C)    →  w[swing]   = -W
                                            w[support] =  0
    DOUBLE transition (last_swing set)
        steps 1..6   (grace)              →  w[prev_support] =  0
                                            w[prev_swing]   = -W
        steps 7+                           →  w[prev_support] = +W
                                            w[prev_swing]   = -W

Phase semantics
---------------
Phase A (steps 1..2, ~0.1 s @ 20 Hz):
    Press the support foot down (w[support] = -W).  The swing foot gets
    +W if its height is below SWING_LIFT_THRESHOLD (it hasn't lifted
    enough yet), otherwise w = 0 (lift was already encouraged by the
    preceding DOUBLE transition).  Goal: weight transfer onto the new
    support foot + ensure the swing foot actually leaves the ground.

Phase B (steps 3..10, ~0.4 s):
    Coast — no support-foot encouragement.  The swing foot still gets
    +W if below SWING_LIFT_THRESHOLD, ensuring it stays airborne.
    Once lifted enough, let physics carry it naturally.

Phase C (steps 11+, ~0.55 s+):
    Encourage the swing foot to lower (prepare for landing).  The support
    foot is left alone (w = 0): it should stay planted, not start lifting
    prematurely.

DOUBLE grace period (steps 1..6):
    Allow the robot to settle on both feet without being pushed to lift.
    The landing foot (prev_swing) is still encouraged to press down
    (w = -W), but the other foot (prev_support) is left alone (w = 0).
    After the grace period, the next step is initiated.

FLIGHT continuation:
    When both feet momentarily leave the ground (brief hop, gait
    oscillation), the state is inherited from the previous frame rather
    than resetting the gait schedule.  This means a SUPPORT_* → FLIGHT
    transition continues counting support_steps, and a DOUBLE → FLIGHT
    transition continues counting double_steps.  Only a FLIGHT at the
    very start of the episode (no previous state) produces zero weights.

Self-correction property
------------------------
``last_swing`` is updated unconditionally to whichever foot is actually
airborne.  In the DOUBLE transition the weights point at
``opposite(last_swing)`` (previous support → lift) and ``last_swing``
(previous swing → lower), so if the robot lifted the *wrong* foot on the
previous step it is pushed back toward the correct foot during DOUBLE.

Inside SUPPORT_* the weights follow the *actual* swing/support feet, not
an expected-swing target.  This means that if the robot commits to
lifting the same foot twice, Phase A will reinforce that choice (+W on
the swing foot).  This is by design: once a foot is committed, let the
gait cycle complete; correction happens at the next DOUBLE transition.
Occasionally repeating the same foot is acceptable, and forcing a
mid-stride correction would fight the physics.

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

PHASE_A_STEPS: int = 2
"""Phase A duration (steps 1..PHASE_A_STEPS): press support foot down."""

PHASE_B_END: int = 10
"""Phase B ends at this step (steps PHASE_A_STEPS+1 .. PHASE_B_END): coast,
no encouragement — let the swing foot travel naturally."""

DOUBLE_GRACE_STEPS: int = 6
"""Grace period (steps) at the start of DOUBLE support.  During the first
DOUBLE_GRACE_STEPS frames the robot is allowed to settle on both feet
without being pushed to lift a foot.  After the grace period, the
DOUBLE weights resume encouraging the next step."""

FOOT_HEIGHT_CLIP: float = 0.1
"""Foot height reward saturation (m).  Lifting beyond this earns nothing
more, preventing a degenerate 'raise the knee as high as possible' policy."""

SWING_LIFT_THRESHOLD: float = 0.05
"""Minimum swing-foot height (m) during Phase A/B before the lift
encouragement turns off.  If the swing foot hasn't risen above this,
a +W actor weight is applied to keep pushing it up."""


def compute_foot_weights(
    contact_l: np.ndarray,
    contact_r: np.ndarray,
    T: int,
    h_left: Optional[np.ndarray] = None,
    h_right: Optional[np.ndarray] = None,
    weight: float = FOOT_WEIGHT,
    phase_a_steps: int = PHASE_A_STEPS,
    phase_b_end: int = PHASE_B_END,
    double_grace_steps: int = DOUBLE_GRACE_STEPS,
    swing_lift_threshold: float = SWING_LIFT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-hoc scan producing per-frame actor weights for both feet.

    Returns ``(w_left, w_right)``, each shape ``(T,)`` float32.
    See the module docstring for the full rule table.

    ``h_left`` / ``h_right`` are per-frame foot heights (m) used for the
    swing-lift gate in Phase A/B.  If omitted, the gate is disabled (as
    if the swing foot is always above threshold).
    """
    w_left = np.zeros(T, dtype=np.float32)
    w_right = np.zeros(T, dtype=np.float32)

    last_swing: Optional[str] = None
    prev_state: Optional[str] = None
    support_steps: int = 0
    double_steps: int = 0

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

        # --- FLIGHT is a continuation of the previous state ---
        # A momentary loss of both contacts (brief hop, gait oscillation)
        # should not reset the gait schedule.  Inherit the previous state
        # so counters and weights continue as if the contact never left.
        if state == STATE_FLIGHT and prev_state is not None and prev_state != STATE_FLIGHT:
            state = prev_state

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
            double_steps = 0
        elif state == STATE_DOUBLE:
            double_steps = double_steps + 1 if state == prev_state else 1
            support_steps = 0
        else:
            support_steps = 0
            double_steps = 0

        # --- Weights ---
        if state == STATE_FLIGHT:
            # Neither foot down: don't inject a direction, let r_fall lead.
            pass
        elif current_swing is not None:
            # Single support — three sub-phases based on support_steps.
            swing_is_left = current_swing == "left"

            # Swing-lift gate: if the swing foot hasn't risen above
            # threshold, keep pushing it up (+W).
            if h_left is not None and h_right is not None:
                h_swing = float(h_left[t]) if swing_is_left else float(h_right[t])
                swing_needs_lift = h_swing < swing_lift_threshold
            else:
                swing_needs_lift = False

            if support_steps <= phase_a_steps:
                # Phase A: press support foot down.
                if swing_is_left:
                    w_right[t] = -weight     # support down
                else:
                    w_left[t] = -weight
                # Swing foot: +W if not lifted enough yet.
                if swing_needs_lift:
                    if swing_is_left:
                        w_left[t] = weight
                    else:
                        w_right[t] = weight
            elif support_steps <= phase_b_end:
                # Phase B: coast on support foot.
                # Swing foot: +W if not lifted enough yet.
                if swing_needs_lift:
                    if swing_is_left:
                        w_left[t] = weight
                    else:
                        w_right[t] = weight
            else:
                # Phase C: encourage swing foot down, leave support alone.
                if swing_is_left:
                    w_left[t] = -weight     # swing down
                else:
                    w_right[t] = -weight
        elif last_swing is None:
            # Initial double support, no step taken yet.
            # Grace: allow the robot to settle before pushing to lift.
            if double_steps > double_grace_steps:
                w_left[t] = weight
                w_right[t] = weight
        else:
            # DOUBLE transition: previous_swing == last_swing,
            # previous_support == opposite(last_swing).
            # Grace: let the landing foot settle (prev_swing=-W) but
            # don't push the other foot up (prev_support=0) yet.
            if double_steps > double_grace_steps:
                if last_swing == "left":
                    w_left[t] = -weight     # previous swing down
                    w_right[t] = weight     # previous support up
                else:
                    w_right[t] = -weight
                    w_left[t] = weight
            else:
                # Grace period: only push prev_swing down.
                if last_swing == "left":
                    w_left[t] = -weight     # previous swing down
                else:
                    w_right[t] = -weight

        prev_state = state

    return w_left, w_right
