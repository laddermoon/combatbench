"""Arm punch state machine — post-hoc scan producing per-step actor weights.

Two-state machine with coupled arms (always opposite):

    State A: left=ATTACK,  right=PREPARE
    State B: right=ATTACK, left=PREPARE

The state machine only runs on **valid segments** — contiguous stretches
where the robot is in attack range and facing the opponent.  Each valid
segment runs the state machine independently (fresh initial state at the
first step of each segment).  Invalid steps get zero weight.

Initial state (first step of each valid segment):
    Compare left_hand_to_opp_head vs right_hand_to_opp_head.
    The hand closer to the opponent's head enters ATTACK.

State transitions (only triggered by the attacking hand's completion):

    State A → State B:
        Left attack completes when:
            left_elbow_norm <= ELBOW_EXTEND_THRESHOLD   (elbow fully extended, norm → -1)
            OR
            left_hand_to_shoulder > opp_head_to_left_shoulder  (can't reach opp)

    State B → State A:
        Right attack completes when:
            right_elbow_norm <= ELBOW_EXTEND_THRESHOLD
            OR
            right_hand_to_shoulder > opp_head_to_right_shoulder

The preparing hand's completion does NOT trigger any transition.

Output: four (T,) float32 arrays of actor weights, each in {-W, 0, +W}:

    w_left_elbow      — -W when left=ATTACK (encourage norm → -1, elbow straight/伸直),
                        +W when left=PREPARE (encourage norm → +1, elbow flexed/收回),
                        0 on invalid steps
    w_right_elbow     — -W when right=ATTACK,
                        +W when right=PREPARE,
                        0 on invalid steps
    w_left_hand_dist  — -W when left=ATTACK (encourage hand → opp head),
                        +W when left=PREPARE (encourage hand away from opp),
                        0 on invalid steps
    w_right_hand_dist — -W when right=ATTACK,
                        +W when right=PREPARE,
                        0 on invalid steps

Elbow normalization (from simulator):
    -1 = fully extended/straight (伸直, -100°) — punch extended
    +1 = fully flexed/retracted (收回, +50°)   — chambered/retracted
    ATTACK rewards extension (norm → -1), PREPARE rewards flexion (norm → +1).

The experiment applies additional gating (phi_height², balance_mask) on top
of these weights.  The distance + facing gate is applied HERE (as the valid
mask) so that the state machine only runs on valid segments.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# Default threshold for elbow fully extended (normalized [-1, 1]).
# -0.97 → close to -1 (fully straight/伸直).
ELBOW_EXTEND_THRESHOLD: float = -0.97

# Default arm actor weight magnitude.
ARM_WEIGHT: float = 1.0


def _find_segments(valid: np.ndarray) -> list:
    """Find contiguous True segments in a boolean array.

    Returns list of (start, end) pairs (end exclusive).
    """
    segments = []
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


def _run_segment(
    left_elbow: np.ndarray,
    right_elbow: np.ndarray,
    left_hand_to_opp_head: np.ndarray,
    right_hand_to_opp_head: np.ndarray,
    left_hand_to_shoulder: np.ndarray,
    right_hand_to_shoulder: np.ndarray,
    opp_head_to_left_shoulder: np.ndarray,
    opp_head_to_right_shoulder: np.ndarray,
    start: int,
    end: int,
    W: float,
    elbow_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run state machine on one valid segment [start, end).

    Returns 4 arrays of length (end - start).
    """
    seg_len = end - start
    w_le = np.full(seg_len, W, dtype=np.float32)
    w_re = np.full(seg_len, W, dtype=np.float32)
    w_lhd = np.full(seg_len, W, dtype=np.float32)
    w_rhd = np.full(seg_len, W, dtype=np.float32)

    if seg_len == 0:
        return w_le, w_re, w_lhd, w_rhd

    # --- Initial state: hand closer to opp head → ATTACK ---
    if left_hand_to_opp_head[start] <= right_hand_to_opp_head[start]:
        state_a = True   # left attacks first
    else:
        state_a = False  # right attacks first

    for i, t in enumerate(range(start, end)):
        if state_a:
            # left=ATTACK, right=PREPARE
            w_le[i] = -W
            w_lhd[i] = -W
            w_re[i] = W
            w_rhd[i] = W

            # Check if left attack is complete
            elbow_done = float(left_elbow[t]) <= elbow_threshold
            cant_reach = (
                float(left_hand_to_shoulder[t]) >
                float(opp_head_to_left_shoulder[t])
            )
            if elbow_done or cant_reach:
                state_a = False  # switch to State B
        else:
            # right=ATTACK, left=PREPARE
            w_le[i] = W
            w_lhd[i] = W
            w_re[i] = -W
            w_rhd[i] = -W

            # Check if right attack is complete
            elbow_done = float(right_elbow[t]) <= elbow_threshold
            cant_reach = (
                float(right_hand_to_shoulder[t]) >
                float(opp_head_to_right_shoulder[t])
            )
            if elbow_done or cant_reach:
                state_a = True  # switch to State A

    return w_le, w_re, w_lhd, w_rhd


def compute_arm_weights(
    left_elbow: np.ndarray,               # (T,) normalized [-1, 1]
    right_elbow: np.ndarray,              # (T,) normalized [-1, 1]
    left_hand_to_opp_head: np.ndarray,    # (T,) meters
    right_hand_to_opp_head: np.ndarray,   # (T,) meters
    left_hand_to_shoulder: np.ndarray,    # (T,) meters
    right_hand_to_shoulder: np.ndarray,   # (T,) meters
    opp_head_to_left_shoulder: np.ndarray,   # (T,) meters
    opp_head_to_right_shoulder: np.ndarray,  # (T,) meters
    *,
    valid_mask: np.ndarray = None,        # (T,) bool — where state machine runs
    elbow_threshold: float = ELBOW_EXTEND_THRESHOLD,
    arm_weight: float = ARM_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-hoc scan over a trajectory to produce arm actor weights.

    The state machine runs independently on each contiguous valid segment
    (where ``valid_mask`` is True).  Invalid steps get zero weight.

    Args:
        valid_mask: (T,) boolean array. If None, all steps are valid
                    (same behavior as before — runs on entire trajectory).

    Returns:
        (w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist)
        Each is (T,) float32 with values in {-arm_weight, 0, +arm_weight}.
    """
    T = len(left_elbow)
    W = float(arm_weight)

    # Default: all zeros (invalid steps)
    w_left_elbow = np.zeros(T, dtype=np.float32)
    w_right_elbow = np.zeros(T, dtype=np.float32)
    w_left_hand_dist = np.zeros(T, dtype=np.float32)
    w_right_hand_dist = np.zeros(T, dtype=np.float32)

    if T == 0:
        return w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist

    # If no valid_mask provided, treat entire trajectory as one valid segment
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    # --- Find contiguous valid segments and run state machine on each ---
    segments = _find_segments(valid_mask)

    for start, end in segments:
        seg_len = end - start
        if seg_len == 0:
            continue

        s_le, s_re, s_lhd, s_rhd = _run_segment(
            left_elbow, right_elbow,
            left_hand_to_opp_head, right_hand_to_opp_head,
            left_hand_to_shoulder, right_hand_to_shoulder,
            opp_head_to_left_shoulder, opp_head_to_right_shoulder,
            start, end, W, elbow_threshold,
        )
        w_left_elbow[start:end] = s_le
        w_right_elbow[start:end] = s_re
        w_left_hand_dist[start:end] = s_lhd
        w_right_hand_dist[start:end] = s_rhd

    return w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist
