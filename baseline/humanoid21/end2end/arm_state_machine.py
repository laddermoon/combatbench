"""Arm punch state machine — post-hoc scan producing per-step actor weights.

v3: reactive per-arm design (no global ``turn``).

Earlier versions enforced strict alternation through a hidden ``turn``
variable — but the policy cannot observe ``turn``, so it could never
phase-lock its swings to the machine's assignment: both arms extended
deeply but always *out of turn*, during the other arm's window (or
during FLEX), and punch counts stayed near zero despite healthy arm
motion.  Strict alternation enforced by an invisible conductor is an
unlearnable credit-assignment problem.

Each arm now runs an independent two-state cycle:

    READY   — chambered; extending is rewarded (aw=+W elbow,
              aw=-W hand_dist).  elbow <= extend_threshold completes a
              punch (counted) and moves the arm to RETRACT.
    RETRACT — just punched; retracting is rewarded (aw=-W elbow,
              aw=+W hand_dist).  elbow >= flex_threshold moves the arm
              back to READY.

Alternation is expected to *emerge* from the walking arm-swing rhythm
rather than be imposed — both arms get symmetric incentives, and any
arm that chambers earns its next punch.  If single-arm dominance or
double-punching shows up in video review, a soft turn-preference can
be re-added on top.

Anti-lock is intrinsic: an arm extended (elbow <= extend_threshold)
outside valid punching context is in/enters RETRACT and must chamber
before earning extension weights — a statically held-out arm can never
farm punches.  Punches count only on valid frames; state still tracks
the elbow through invalid frames so re-entry is consistent.

Actor weights per state:

    READY        →  aw_elbow = +W,  aw_hand_dist = -W
    RETRACT      →  aw_elbow = -W,  aw_hand_dist = +W
    invalid step →  aw_elbow =  0,  aw_hand_dist =  0

Elbow reward mapping (applied in the experiment, not here):
    r_elbow = (1 - elbow_norm) / 2   →   range [0, 1]
    norm=-1 (fully extended/伸直) → r=1.0 (punched out)
    norm=+1 (fully flexed/收回)   → r=0.0 (chambered)

    READY:   aw=+W → PPO maximizes → reward↑ → norm↓ → elbow extends
    RETRACT: aw=-W → PPO minimizes → reward↓ → norm↑ → elbow retracts

Hand distance reward is the raw 3D distance to the opponent head (meters):
    READY:   aw=-W → encourages distance↓ (hand → opp head)
    RETRACT: aw=+W → encourages distance↑ (hand away from opp)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# Elbow normalized [-1, 1]: -1 = fully extended (伸直), +1 = fully flexed (收回).
# Small margins are left at both ends so the thresholds are reachable.
ELBOW_EXTEND_THRESHOLD: float = -0.95
"""A punch completes when elbow_norm <= this (nearly straight)."""

ELBOW_FLEX_THRESHOLD: float = 0.95
"""RETRACT completes when elbow_norm >= this (nearly fully chambered)."""

# Default arm actor weight magnitude.
ARM_WEIGHT: float = 3.0

ATTACK_TIMEOUT_STEPS: int = 60
"""Unused in v3 (reactive machine needs no turn timeouts) — kept in the
public signatures for call-site compatibility."""

# Arm states (reactive per-arm cycle)
_READY = 0    # chambered — extension is rewarded
_RETRACT = 1  # punched out — retraction is rewarded until re-chambered


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
    n_segments = len(_find_segments(valid_mask))

    # Per-arm persistent states — reactive READY/RETRACT cycle.
    # An arm that starts (or drifts) extended enters RETRACT: it must
    # chamber before earning extension weights (anti-lock).
    left_state = (
        _RETRACT if float(left_elbow[0]) <= extend_threshold else _READY
    )
    right_state = (
        _RETRACT if float(right_elbow[0]) <= extend_threshold else _READY
    )

    for t in range(T):
        le = float(left_elbow[t])
        re_ = float(right_elbow[t])

        # --- State transitions track the elbow continuously, even on
        # invalid frames (no weights there, no punch counts either) ---
        if left_state == _READY and le <= extend_threshold:
            if valid_mask[t]:
                n_left_punches += 1
            left_state = _RETRACT
        elif left_state == _RETRACT and le >= flex_threshold:
            left_state = _READY

        if right_state == _READY and re_ <= extend_threshold:
            if valid_mask[t]:
                n_right_punches += 1
            right_state = _RETRACT
        elif right_state == _RETRACT and re_ >= flex_threshold:
            right_state = _READY

        # --- Emit weights on valid frames only ---
        if not valid_mask[t]:
            continue
        if left_state == _READY:
            w_left_elbow[t] = W
            w_left_hand_dist[t] = -W
        else:
            w_left_elbow[t] = -W
            w_left_hand_dist[t] = W
        if right_state == _READY:
            w_right_elbow[t] = W
            w_right_hand_dist[t] = -W
        else:
            w_right_elbow[t] = -W
            w_right_hand_dist[t] = W

    n_valid = int(np.count_nonzero(valid_mask))
    stats: Dict[str, float] = {
        "n_punches": float(n_left_punches + n_right_punches),
        "n_left_punches": float(n_left_punches),
        "n_right_punches": float(n_right_punches),
        "n_valid_steps": float(n_valid),
        "n_segments": float(n_segments),
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
