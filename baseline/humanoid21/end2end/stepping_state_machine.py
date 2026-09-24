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
                                            w[swing]   = +W if h_swing < SWING_LIFT_THRESHOLD
                                                          or still rising; else 0
    SUPPORT_*  steps 3..10  (Phase B)    →  w[swing]   = +W if h_swing < SWING_LIFT_THRESHOLD
                                                          or still rising; else 0
                                            w[support] =  0
    SUPPORT_*  steps 11+    (Phase C)    →  w[swing]   = -W if h_swing >= SWING_LIFT_THRESHOLD
                                                          and no longer rising
                                                          else +W (late lift / rising apex
                                                          still pushed up)
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
    The swing foot is overdue.  If it reached SWING_LIFT_THRESHOLD,
    encourage it to lower (prepare for landing).  If it never lifted
    enough, keep pushing it up (+W) — a foot that is still below the bar
    must not be punished for making slow progress.  The support foot is
    left alone (w = 0): it should stay planted, not start lifting
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
``ppo.trainer.ppo_update`` (a channel whose weights are all <= 0 was
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

CONTACT_HOLD_STEPS: int = 4
"""Minimum burst length for a contact signal to be kept.  Contact bursts
shorter than this (1-3 frame spikes from MuJoCo jitter) are removed
entirely.  Bursts ≥ CONTACT_HOLD_STEPS are kept as-is with no delay.
This is a post-hoc filter (uses future frames), applied on the full
trajectory before the state machine scan."""

FOOT_HEIGHT_CLIP: float = 0.05
"""Foot height reward saturation (m).  Lifting beyond this earns nothing
more, preventing a degenerate 'raise the knee as high as possible' policy."""

SWING_LIFT_THRESHOLD: float = 0.05
"""Minimum swing-foot height (m) during Phase A/B before the lift
encouragement turns off.  If the swing foot hasn't risen above this,
a +W actor weight is applied to keep pushing it up."""

SOLE_CLEAR_THRESHOLD: float = 0.03
"""Minimum sole clearance (m) for a swing to count as a real step.
Sole clearance = min world-z over the four foot capsule endpoints minus
geom radius: ~0 while any sole edge is at ground level, so foot-rocking
(pivot on toe/side edge) cannot fake it — unlike midpoint ``h``, which
rises under a pure tilt (run 064853 exploited exactly this)."""


def _hold_filter(contact: np.ndarray, hold: int) -> np.ndarray:
    """Remove short bursts in both directions (post-hoc, no delay).

    Symmetric filter: runs of either True or False shorter than ``hold``
    frames are absorbed into the surrounding signal.

    - Short True bursts (< hold) → set to False  (contact jitter spikes)
    - Short False bursts (< hold) → set to True   (contact jitter gaps)

    Example (hold=4):
        001000  → 000000   (1-frame True spike removed)
        011110  → 011110   (4-frame True burst kept)
        111101111 → 111111111  (1-frame False gap filled)
        1110000111 → 1110000111 (3-frame False gap kept)

    This is a post-hoc filter — it uses the full signal and may look
    ahead.  Safe for trajectory post-processing.
    """
    T = len(contact)
    if T == 0:
        return contact.copy()

    # Find all runs (alternating True/False), record (start, length, value)
    runs = []
    t = 0
    while t < T:
        val = bool(contact[t])
        run_end = t
        while run_end < T and bool(contact[run_end]) == val:
            run_end += 1
        runs.append((t, run_end - t, val))
        t = run_end

    if len(runs) <= 1:
        return contact.copy()

    out = contact.copy()
    # Interior short runs (not first, not last) get absorbed into neighbors
    for i in range(1, len(runs) - 1):
        start, length, val = runs[i]
        if length < hold:
            # Absorb: set this run to the opposite of its value
            # (i.e. merge into surrounding runs)
            out[start:start + length] = not val
    return out


def single_support_mask(
    contact_l: np.ndarray,
    contact_r: np.ndarray,
    hold: int = CONTACT_HOLD_STEPS,
) -> np.ndarray:
    """Debounced single-support mask: True where exactly one foot is down.

    Uses the same ``_hold_filter`` view as ``compute_foot_weights`` so the
    mask is consistent with the gait schedule.  Unlike the state machine,
    FLIGHT is *not* inherited — both-feet-off frames are False (a hop is
    not a commanded swing and must not be exempt from e.g. a standing-
    potential actor-weight gate).
    """
    cl = _hold_filter(np.asarray(contact_l, dtype=bool), hold)
    cr = _hold_filter(np.asarray(contact_r, dtype=bool), hold)
    return cl ^ cr


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

    # Debounce contact signals: require CONTACT_HOLD_STEPS consecutive
    # frames to confirm or release a contact.  Eliminates MuJoCo
    # contact jitter (1-2 frame spikes/drops) that would cause the
    # state machine to spuriously switch states.
    contact_l = _hold_filter(contact_l, CONTACT_HOLD_STEPS)
    contact_r = _hold_filter(contact_r, CONTACT_HOLD_STEPS)

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
                h_swing_prev = (
                    float(h_left[t - 1]) if swing_is_left
                    else float(h_right[t - 1])
                ) if t > 0 else h_swing
                swing_needs_lift = h_swing < swing_lift_threshold
                # Still rising toward/through the apex — the u50 dump
                # showed that penalizing the ascent (h>=thresh & rising)
                # puts negative weight on the frames with the highest
                # foot advantage, discouraging exactly the apex.
                swing_rising = h_swing > h_swing_prev
            else:
                swing_needs_lift = False
                swing_rising = False

            if support_steps <= phase_a_steps:
                # Phase A: press support foot down.
                if swing_is_left:
                    w_right[t] = -weight     # support down
                else:
                    w_left[t] = -weight
                # Swing foot: +W while below threshold or still rising.
                if swing_needs_lift or swing_rising:
                    if swing_is_left:
                        w_left[t] = weight
                    else:
                        w_right[t] = weight
            elif support_steps <= phase_b_end:
                # Phase B: coast on support foot.
                # Swing foot: +W while below threshold or still rising.
                if swing_needs_lift or swing_rising:
                    if swing_is_left:
                        w_left[t] = weight
                    else:
                        w_right[t] = weight
            else:
                # Phase C: swing overdue.  If the foot reached the lift
                # threshold and has crested (no longer rising), encourage
                # descent; otherwise keep pushing up — a late lift or a
                # still-rising apex must not be punished for making
                # progress.
                if swing_needs_lift or swing_rising:
                    if swing_is_left:
                        w_left[t] = weight
                    else:
                        w_right[t] = weight
                elif swing_is_left:
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


# ----------------------------------------------------------------------
# Clock-driven commands (observable gait schedule)
# ----------------------------------------------------------------------

GAIT_PERIOD: int = 40
"""Action steps per full L+R command cycle (must match the simulator's
``gait_period``).  Each foot owns half the cycle — e.g. 40 @ 20 Hz gives
a 2 s cycle, 1 s per foot."""

GAIT_LAND_FRAC: float = 0.75
"""Within a foot's window, the last (1 - GAIT_LAND_FRAC) commands landing
(-W) so the foot is back down before the other foot's window starts."""


def clock_foot_weights(
    T: int,
    sole_left: Optional[np.ndarray] = None,
    sole_right: Optional[np.ndarray] = None,
    *,
    period: int = GAIT_PERIOD,
    land_frac: float = GAIT_LAND_FRAC,
    weight: float = FOOT_WEIGHT,
    lift_threshold: float = SOLE_CLEAR_THRESHOLD,
    stable: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame actor weights driven by the observable gait clock.

    Unlike :func:`compute_foot_weights` (which reacts to measured
    contacts), the commanded foot is a deterministic function of the
    frame index — identical to the ``cmd_L``/``cmd_R``/``wprog``
    observation dims appended by ``GaitClockSimulator``.  The policy can
    therefore learn "when cmd_L=1 → lift left foot" as an ordinary
    state→action mapping instead of inferring invisible intent.

    Rules per frame ``t`` (later gated by φ² in the experiment — the
    raw schedule runs regardless of standing state):

    - commanded foot: ``+W`` while sole clearance is below
      ``lift_threshold`` or still rising (apex-aligned); once the window
      passes ``land_frac`` of its length the command flips to ``-W`` so
      the foot lands before the window ends.  A commanded foot that is
      already clear and no longer rising mid-window gets 0 (coast).
    - support foot: ``-W`` — it must stay down; an uncommanded lift is
      always punished (self-correction built in).
    - stability gate: when ``stable`` is given, a ``+W`` command on an
      unstable frame flips to ``-W`` — lifting while wobbly is actively
      punished instead of merely unrewarded.  The u1801 dump showed
      ~5% of liftoffs happen at φ<0.5 and a quarter of airborne frames
      sit at φ<0.23 (body already falling, foot still up); under the
      unconditional clock those frames still earned +W.

    ``sole_*`` must be sole clearance (min capsule-endpoint z − radius),
    NOT midpoint height — midpoint rises under foot-rocking and would
    mark a pivot as "lifted".
    """
    w_left = np.zeros(T, dtype=np.float32)
    w_right = np.zeros(T, dtype=np.float32)
    half = max(1, period // 2)
    sl = np.asarray(sole_left, dtype=np.float32) if sole_left is not None else None
    sr = np.asarray(sole_right, dtype=np.float32) if sole_right is not None else None

    for t in range(T):
        pos = t % period
        if pos < half:
            cmd_left, wprog = True, pos / half
        else:
            cmd_left, wprog = False, (pos - half) / half

        h_cmd = (sl if cmd_left else sr)
        h_cmd_t = float(h_cmd[t]) if h_cmd is not None else None
        h_prev = (
            float(h_cmd[t - 1]) if h_cmd is not None and t > 0 else h_cmd_t
        )
        rising = h_cmd_t is not None and h_cmd_t > h_prev
        needs_lift = h_cmd_t is not None and h_cmd_t < lift_threshold

        if wprog < land_frac:
            w_cmd = weight if (h_cmd_t is None or needs_lift or rising) else 0.0
        else:
            w_cmd = -weight
        if w_cmd > 0 and stable is not None and not stable[t]:
            w_cmd = -weight
        if cmd_left:
            w_left[t], w_right[t] = w_cmd, -weight
        else:
            w_right[t], w_left[t] = w_cmd, -weight

    return w_left, w_right


# ----------------------------------------------------------------------
# Step-cycle detection (eval/diagnostic, shares the debounced-contact view)
# ----------------------------------------------------------------------

def detect_step_cycles(
    contact_l: np.ndarray,
    contact_r: np.ndarray,
    h_left: np.ndarray,
    h_right: np.ndarray,
    standing: np.ndarray,
    sole_left: Optional[np.ndarray] = None,
    sole_right: Optional[np.ndarray] = None,
    *,
    hold: int = CONTACT_HOLD_STEPS,
    min_air_steps: int = 3,
    h_thresh: float = SWING_LIFT_THRESHOLD,
    sole_thresh: float = SOLE_CLEAR_THRESHOLD,
) -> dict:
    """Post-hoc detection of valid step cycles on debounced contacts.

    A *swing attempt* for foot F is a contiguous run of the SUPPORT state
    where F is airborne, starting on a standing frame and lasting at
    least ``min_air_steps`` frames.  A *valid step cycle* additionally
    requires:

      1. peak foot midpoint height during the swing >= ``h_thresh``;
      2. when ``sole_*`` clearance arrays are given, peak sole clearance
         during the swing >= ``sole_thresh`` — this rejects foot-rocking
         (pivot on toe/side edge lifts the midpoint but leaves sole
         clearance ~0);
      3. the run ends with F regaining contact (lands) — runs that end in
         FLIGHT (hop/stumble) or at episode end do not count;
      4. the landing frame is also a standing frame (falling down is not
         a step).

    Because the run is contiguous SUPPORT_* by construction, the other
    foot stays on the ground for the whole swing — a hop (both feet
    airborne) can never be counted.

    All inputs are per-frame arrays of length T.  ``standing`` is the
    caller's standing gate (e.g. ``phi >= 0.9``).

    Returns a dict::

        cycles         list of (foot, t_off, t_land, h_peak)
        n_cycles_left  valid cycles completed by the left foot
        n_cycles_right valid cycles completed by the right foot
        n_swings       swing attempts (standing + min duration, any height)
        h_swing_max    mean peak swing height over attempts (0 if none)
        alt_ratio      fraction of consecutive cycles on opposite feet
                       (None when fewer than 2 cycles)
        stepped        True iff both feet completed >= 1 valid cycle
    """
    contact_l = _hold_filter(np.asarray(contact_l, dtype=bool), hold)
    contact_r = _hold_filter(np.asarray(contact_r, dtype=bool), hold)
    h_left = np.asarray(h_left, dtype=np.float32)
    h_right = np.asarray(h_right, dtype=np.float32)
    standing = np.asarray(standing, dtype=bool)
    sole_l = np.asarray(sole_left, dtype=np.float32) if sole_left is not None else None
    sole_r = np.asarray(sole_right, dtype=np.float32) if sole_right is not None else None
    T = len(contact_l)

    cycles = []
    n_swings = 0
    h_peaks = []
    sole_peaks = []

    t = 0
    while t < T:
        cl, cr = bool(contact_l[t]), bool(contact_r[t])
        if cl == cr:
            t += 1
            continue
        # SUPPORT_L: left down, right swings.  SUPPORT_R: right down, left swings.
        swing_is_left = cr
        a = t
        while t < T and bool(contact_l[t]) == cl and bool(contact_r[t]) == cr:
            t += 1
        b = t  # run = [a, b)

        if not standing[a] or (b - a) < min_air_steps:
            continue
        h_swing = h_left if swing_is_left else h_right
        h_peak = float(h_swing[a:b].max())
        sole_swing = sole_l if swing_is_left else sole_r
        sole_peak = (
            float(sole_swing[a:b].max()) if sole_swing is not None else None
        )
        n_swings += 1
        h_peaks.append(h_peak)
        if sole_peak is not None:
            sole_peaks.append(sole_peak)

        landed = b < T and (
            bool(contact_l[b]) if swing_is_left else bool(contact_r[b])
        )
        sole_ok = sole_peak is None or sole_peak >= sole_thresh
        if landed and standing[b] and h_peak >= h_thresh and sole_ok:
            cycles.append(("left" if swing_is_left else "right", a, b, h_peak))

    n_alt = sum(
        1 for i in range(1, len(cycles)) if cycles[i][0] != cycles[i - 1][0]
    )
    n_left = sum(1 for c in cycles if c[0] == "left")

    return {
        "cycles": cycles,
        "n_cycles_left": n_left,
        "n_cycles_right": len(cycles) - n_left,
        "n_swings": n_swings,
        "h_swing_max": float(np.mean(h_peaks)) if h_peaks else 0.0,
        "sole_swing_max": (
            float(np.mean(sole_peaks)) if sole_peaks else None
        ),
        "alt_ratio": (n_alt / (len(cycles) - 1)) if len(cycles) >= 2 else None,
        "stepped": n_left >= 1 and (len(cycles) - n_left) >= 1,
    }


def count_falls(
    phi: np.ndarray,
    *,
    stand_hi: float = 0.7,
    fall_lo: float = 0.5,
    min_frames: int = 10,
) -> int:
    """Count real falls: standing (φ≥``stand_hi``) then φ<``fall_lo``
    sustained for at least ``min_frames`` consecutive frames.

    Single/double-frame φ dips are stumbles or sensor noise, not falls —
    a genuine knockdown keeps φ low for many frames before the standup
    policy recovers it (~40 frames).  The u1801 dump used these exact
    thresholds: 7812 raw dips vs 1192 sustained events (~1.16/traj).
    """
    phi = np.asarray(phi, dtype=np.float32)
    n = len(phi)
    falls = 0
    j = 0
    while j < n - 1:
        if phi[j] >= stand_hi and phi[j + 1] < fall_lo:
            k = j + 1
            while k < n and phi[k] < fall_lo:
                k += 1
            if k - (j + 1) >= min_frames:
                falls += 1
            j = k
        else:
            j += 1
    return falls
