"""Arm punch state machine — post-hoc scan producing per-step actor weights.

Two-state machine with coupled arms (always opposite):

    State A: left=ATTACK,  right=PREPARE
    State B: right=ATTACK, left=PREPARE

Initial state (step 0):
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

Output: four (T,) float32 arrays of actor weights, each in {-W, +W}:

    w_left_elbow      — -W when left=ATTACK (encourage norm → -1, elbow straight/伸直),
                        +W when left=PREPARE (encourage norm → +1, elbow flexed/收回)
    w_right_elbow     — -W when right=ATTACK,
                        +W when right=PREPARE
    w_left_hand_dist  — -W when left=ATTACK (encourage hand → opp head),
                        +W when left=PREPARE (encourage hand away from opp)
    w_right_hand_dist — -W when right=ATTACK,
                        +W when right=PREPARE

Elbow normalization (from simulator):
    -1 = fully extended/straight (伸直, -100°) — punch extended
    +1 = fully flexed/retracted (收回, +50°)   — chambered/retracted
    ATTACK rewards extension (norm → -1), PREPARE rewards flexion (norm → +1).

The experiment applies additional gating (distance, facing, phi_height²,
balance_mask) on top of these raw ±W weights.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# Default threshold for elbow fully extended (normalized [-1, 1]).
# -0.97 → close to -1 (fully straight/伸直).
ELBOW_EXTEND_THRESHOLD: float = -0.97

# Default arm actor weight magnitude.
ARM_WEIGHT: float = 1.0


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
    elbow_threshold: float = ELBOW_EXTEND_THRESHOLD,
    arm_weight: float = ARM_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-hoc scan over a trajectory to produce arm actor weights.

    Returns:
        (w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist)
        Each is (T,) float32 with values in {-arm_weight, +arm_weight}.
    """
    T = len(left_elbow)
    W = float(arm_weight)

    # Defaults: both PREPARE (elbow aw = +W → encourage flexion/收回)
    w_left_elbow = np.full(T, W, dtype=np.float32)
    w_right_elbow = np.full(T, W, dtype=np.float32)
    w_left_hand_dist = np.full(T, W, dtype=np.float32)
    w_right_hand_dist = np.full(T, W, dtype=np.float32)

    if T == 0:
        return w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist

    # --- Determine initial state ---
    # State A: left=ATTACK, right=PREPARE
    # State B: right=ATTACK, left=PREPARE
    #
    # The hand closer to the opponent's head enters ATTACK.
    if left_hand_to_opp_head[0] <= right_hand_to_opp_head[0]:
        state_a = True   # left attacks first
    else:
        state_a = False  # right attacks first

    for t in range(T):
        if state_a:
            # left=ATTACK, right=PREPARE
            # ATTACK: elbow aw = -W (encourage norm → -1, straight/伸直)
            # PREPARE: elbow aw = +W (encourage norm → +1, flexed/收回)
            w_left_elbow[t] = -W
            w_left_hand_dist[t] = -W
            w_right_elbow[t] = W
            w_right_hand_dist[t] = W

            # Check if left attack is complete (elbow fully extended → norm <= threshold)
            elbow_done = float(left_elbow[t]) <= elbow_threshold
            cant_reach = (
                float(left_hand_to_shoulder[t]) >
                float(opp_head_to_left_shoulder[t])
            )
            if elbow_done or cant_reach:
                state_a = False  # switch to State B
        else:
            # right=ATTACK, left=PREPARE
            w_left_elbow[t] = W
            w_left_hand_dist[t] = W
            w_right_elbow[t] = -W
            w_right_hand_dist[t] = -W

            # Check if right attack is complete
            elbow_done = float(right_elbow[t]) <= elbow_threshold
            cant_reach = (
                float(right_hand_to_shoulder[t]) >
                float(opp_head_to_right_shoulder[t])
            )
            if elbow_done or cant_reach:
                state_a = True  # switch to State A

    return w_left_elbow, w_right_elbow, w_left_hand_dist, w_right_hand_dist
