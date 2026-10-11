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
    ``turn`` persists across segments — whichever arm was due to attack
    next keeps the initiative when punching resumes, so segment breaks
    can't reset the alternation.  (The previous per-segment "closer hand
    attacks first" re-init was a ratchet: the arm that punched last is
    usually still closer, so it claimed ATTACK at every re-init and the
    other arm starved.)  Only the very first segment uses the closer
    hand for the initial turn.

    An arm already extended (elbow <= extend_threshold) at segment
    start goes to FLEX and must retract before attacking — it can never
    hold or take the turn while extended.  This prevents fake "instant
    punches" at segment boundaries and the held-out-arm ATTACK lock.

ATTACK timeout: an arm that stays in ATTACK for more than
``attack_timeout_steps`` without completing is demoted to FLEX and
the turn passes to the other arm (no punch counted).  This prevents
deadlock when an arm hovers just above the extend threshold —
visually straight but never completing — which would otherwise pin
the other arm in FLEX forever.

The state machine only runs on **valid segments** — contiguous stretches
where the robot is in attack range and facing the opponent.  States
re-initialize at each segment boundary (see above); ``turn`` carries
over.  Invalid steps get zero weight.

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
ARM_WEIGHT: float = 3.0

ATTACK_TIMEOUT_STEPS: int = 60
"""Max consecutive ATTACK steps before the arm is demoted to FLEX (no
punch counted) and the turn passes — guarantees alternation even when
an arm stalls just above the extend threshold."""

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
    attack_timeout_steps: int,
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

    # ``turn`` persists across segments — see module docstring.
    turn = -1  # -1 = uninitialized (first segment picks the closer hand)

    for start, end in segments:
        # --- Segment re-init: states reset, turn carries over ---
        # An arm already extended at segment start goes to FLEX (it must
        # retract before attacking) and can never hold the turn while
        # extended — the turn passes to the other arm if needed.
        left_ext = float(left_elbow[start]) <= extend_threshold
        right_ext = float(right_elbow[start]) <= extend_threshold
        if turn == -1:
            closer = (
                _LEFT
                if left_hand_to_opp_head[start]
                <= right_hand_to_opp_head[start]
                else _RIGHT
            )
            if left_ext and not right_ext:
                turn = _RIGHT
            elif right_ext and not left_ext:
                turn = _LEFT
            else:
                turn = closer
        elif left_ext and turn == _LEFT:
            turn = _RIGHT
        elif right_ext and turn == _RIGHT:
            turn = _LEFT

        left_state = (
            _FLEX if left_ext else (_ATTACK if turn == _LEFT else _READY)
        )
        right_state = (
            _FLEX if right_ext else (_ATTACK if turn == _RIGHT else _READY)
        )

        attack_age = 0

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
            if left_state == _ATTACK:
                attack_age += 1
                if float(left_elbow[t]) <= extend_threshold:
                    left_state = _FLEX
                    turn = _RIGHT
                    n_left_punches += 1
                    attack_age = 0
                elif attack_age > attack_timeout_steps:
                    # Stalled attack — demote without a punch, pass turn.
                    left_state = _FLEX
                    turn = _RIGHT
                    attack_age = 0
            elif right_state == _ATTACK:
                attack_age += 1
                if float(right_elbow[t]) <= extend_threshold:
                    right_state = _FLEX
                    turn = _LEFT
                    n_right_punches += 1
                    attack_age = 0
                elif attack_age > attack_timeout_steps:
                    right_state = _FLEX
                    turn = _LEFT
                    attack_age = 0
            else:
                attack_age = 0

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
    attack_timeout_steps: int = ATTACK_TIMEOUT_STEPS,
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
        attack_timeout_steps=attack_timeout_steps,
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
    attack_timeout_steps: int = ATTACK_TIMEOUT_STEPS,
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
        int(attack_timeout_steps),
    )
