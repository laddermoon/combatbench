"""Arm punch state machine — post-hoc scan producing per-step actor weights.

Each arm has three states; a global ``turn`` variable enforces strict
left/right alternation of attacks:

    ATTACK — extending the elbow to punch out
    FLEX   — retracting the elbow after a punch
    READY  — fully retracted, waiting for its turn to attack

Because ``turn`` only flips when the current attacker completes its punch,
at most one arm is in ATTACK at any step.

State machine (per step, in order):

    1. ATTACK complete:  elbow <= EXTEND_THRESHOLD  →  state = FLEX,
                                                       turn = other arm
    2. FLEX complete:    elbow >= FLEX_THRESHOLD    →  state = READY
    3. READY starts:     turn == self               →  state = ATTACK

Key behaviors:

  - After punching out, the attacking arm starts retracting **immediately**
    and does not wait for the other arm.
  - The turn passes to the other arm as soon as the punch completes, but
    that arm can only start attacking once it is fully retracted (READY).
    If it is still retracting, there is a gap with no arm attacking.
  - An arm cannot punch twice in a row: it must wait for the other arm to
    complete its punch first.

Initial state (first step of each valid segment):
    The hand closer to the opponent's head becomes the attacker
    (state = ATTACK, turn = that arm); the other arm starts in FLEX.

The state machine only runs on **valid segments** — contiguous stretches
where the robot is in attack range and facing the opponent.  Each valid
segment runs the state machine independently with a fresh initial state.
Invalid steps get zero weight.

Actor weights per state:

    ATTACK       →  aw_elbow = +W,  aw_hand_dist = -W
    FLEX / READY →  aw_elbow = -W,  aw_hand_dist = +W
    invalid step →  aw_elbow =  0,  aw_hand_dist =  0

Elbow reward mapping (applied in the experiment, not here):
    r_elbow = (1 - elbow_norm) / 2   →   range [0, 1]
    norm=-1 (fully extended/伸直) → r=1.0 (punched out)
    norm=+1 (fully flexed/收回)   → r=0.0 (chambered)

    ATTACK:  aw=+W → PPO maximizes → reward↑ → norm↓ → elbow extends
    FLEX:    aw=-W → PPO minimizes → reward↓ → norm↑ → elbow retracts

Hand distance reward is the raw 3D distance to the opponent head (meters):
    ATTACK:  aw=-W → encourages distance↓ (hand → opp head)
    FLEX:    aw=+W → encourages distance↑ (hand away from opp)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# Elbow normalized [-1, 1]: -1 = fully extended (伸直), +1 = fully flexed (收回).
# Small margins are left at both ends so the thresholds are reachable.
ELBOW_EXTEND_THRESHOLD: float = -0.95
"""ATTACK completes when elbow_norm <= this (nearly straight)."""

ELBOW_FLEX_THRESHOLD: float = 0.95
"""FLEX completes when elbow_norm >= this (nearly fully retracted)."""

# Default arm actor weight magnitude.
ARM_WEIGHT: float = 1.0

# Arm states
_ATTACK = 0
_FLEX = 1
_READY = 2

# Turn owner
_LEFT = 0
_RIGHT = 1


def _find_segments(valid: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True segments in a boolean array.

    Returns list of (start, end) pairs, end exclusive.
    """
    segments: List[Tuple[int, int]] = []
    T = len(valid)
    t = 0
    while t < T:
        if valid[t]:
            start = t
            while t < T and valid[t]:
                t += 1
            segments.append((start, t))
        else:
            t += 1
    return segments


def _scan(
    left_elbow: np.ndarray,
    right_elbow: np.ndarray,
    left_hand_to_opp_head: np.ndarray,
    right_hand_to_opp_head: np.ndarray,
    valid_mask: np.ndarray,
    extend_threshold: float,
    flex_threshold: float,
    W: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Core scan: returns four (T,) weight arrays plus punch statistics."""
    T = len(left_elbow)

    w_left_elbow = np.zeros(T, dtype=np.float32)
    w_right_elbow = np.zeros(T, dtype=np.float32)
    w_left_hand_dist = np.zeros(T, dtype=np.float32)
    w_right_hand_dist = np.zeros(T, dtype=np.float32)

    n_left_punches = 0
    n_right_punches = 0
    segments = _find_segments(valid_mask)

    for start, end in segments:
        # --- Fresh initial state per segment ---
        # Hand closer to the opponent's head attacks first.
        if left_hand_to_opp_head[start] <= right_hand_to_opp_head[start]:
            left_state, right_state, turn = _ATTACK, _FLEX, _LEFT
        else:
            left_state, right_state, turn = _FLEX, _ATTACK, _RIGHT

        for t in range(start, end):
            # --- 1. Emit weights for the current states ---
            if left_state == _ATTACK:
                w_left_elbow[t] = W
                w_left_hand_dist[t] = -W
            else:  # FLEX or READY
                w_left_elbow[t] = -W
                w_left_hand_dist[t] = W

            if right_state == _ATTACK:
                w_right_elbow[t] = W
                w_right_hand_dist[t] = -W
            else:
                w_right_elbow[t] = -W
                w_right_hand_dist[t] = W

            # --- 2a. ATTACK complete → FLEX, turn passes to the other arm ---
            if left_state == _ATTACK and float(left_elbow[t]) <= extend_threshold:
                left_state = _FLEX
                turn = _RIGHT
                n_left_punches += 1
            elif right_state == _ATTACK and float(right_elbow[t]) <= extend_threshold:
                right_state = _FLEX
                turn = _LEFT
                n_right_punches += 1

            # --- 2b. FLEX complete → READY ---
            if left_state == _FLEX and float(left_elbow[t]) >= flex_threshold:
                left_state = _READY
            if right_state == _FLEX and float(right_elbow[t]) >= flex_threshold:
                right_state = _READY

            # --- 2c. READY and it is my turn → ATTACK ---
            if left_state == _READY and turn == _LEFT:
                left_state = _ATTACK
            if right_state == _READY and turn == _RIGHT:
                right_state = _ATTACK

    n_valid = int(np.count_nonzero(valid_mask))
    stats: Dict[str, float] = {
        "n_punches": float(n_left_punches + n_right_punches),
        "n_left_punches": float(n_left_punches),
        "n_right_punches": float(n_right_punches),
        "n_valid_steps": float(n_valid),
        "n_segments": float(len(segments)),
    }
    return (
        w_left_elbow, w_right_elbow,
        w_left_hand_dist, w_right_hand_dist,
        stats,
    )


def compute_arm_weights(
    left_elbow: np.ndarray,               # (T,) normalized [-1, 1]
    right_elbow: np.ndarray,              # (T,) normalized [-1, 1]
    left_hand_to_opp_head: np.ndarray,    # (T,) meters
    right_hand_to_opp_head: np.ndarray,   # (T,) meters
    *,
    valid_mask: np.ndarray = None,        # (T,) bool — where the machine runs
    extend_threshold: float = ELBOW_EXTEND_THRESHOLD,
    flex_threshold: float = ELBOW_FLEX_THRESHOLD,
    arm_weight: float = ARM_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-hoc scan over a trajectory to produce arm actor weights.

    The state machine runs independently on each contiguous valid segment
    (where ``valid_mask`` is True).  Invalid steps get zero weight.

    Args:
        valid_mask: (T,) boolean array.  If None, the whole trajectory is
                    treated as one valid segment.

    Returns:
        (w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist)
        Each is (T,) float32 with values in {-arm_weight, 0, +arm_weight}.
    """
    w_le, w_re, w_lhd, w_rhd, _ = compute_arm_weights_with_stats(
        left_elbow, right_elbow,
        left_hand_to_opp_head, right_hand_to_opp_head,
        valid_mask=valid_mask,
        extend_threshold=extend_threshold,
        flex_threshold=flex_threshold,
        arm_weight=arm_weight,
    )
    return w_le, w_re, w_lhd, w_rhd


def compute_arm_weights_with_stats(
    left_elbow: np.ndarray,
    right_elbow: np.ndarray,
    left_hand_to_opp_head: np.ndarray,
    right_hand_to_opp_head: np.ndarray,
    *,
    valid_mask: np.ndarray = None,
    extend_threshold: float = ELBOW_EXTEND_THRESHOLD,
    flex_threshold: float = ELBOW_FLEX_THRESHOLD,
    arm_weight: float = ARM_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Same as :func:`compute_arm_weights` but also returns punch statistics.

    Stats keys: ``n_punches``, ``n_left_punches``, ``n_right_punches``,
    ``n_valid_steps``, ``n_segments``.
    """
    T = len(left_elbow)
    W = float(arm_weight)

    if T == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy(), empty.copy(), empty.copy(), {
            "n_punches": 0.0, "n_left_punches": 0.0, "n_right_punches": 0.0,
            "n_valid_steps": 0.0, "n_segments": 0.0,
        }

    if valid_mask is None:
        valid = np.ones(T, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)

    return _scan(
        np.asarray(left_elbow, dtype=np.float64),
        np.asarray(right_elbow, dtype=np.float64),
        np.asarray(left_hand_to_opp_head, dtype=np.float64),
        np.asarray(right_hand_to_opp_head, dtype=np.float64),
        valid,
        float(extend_threshold),
        float(flex_threshold),
        W,
    )
