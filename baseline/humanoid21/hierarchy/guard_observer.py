"""Guard-posture observer — scores how well each hand sits in a
fighting guard position.

A good guard means:
  * Elbow flexed (chambered):  elbow_norm >= 0.5  (1 = fully flexed)
  * Hand near chest/chin height relative to the torso root
  * Hand not too far from the body (not flailing wide)

Outputs (per step, all ∈ [0, 1] unless noted):

  left_elbow_norm       — left elbow normalised angle [-1, 1]
  right_elbow_norm      — right elbow normalised angle [-1, 1]
  left_guard            — left-hand guard score [0, 1]
  right_guard           — right-hand guard score [0, 1]
  guard_score           — mean(left_guard, right_guard)  [0, 1]
  left_hand_height      — hand z relative to torso root (m)
  right_hand_height     — same for right hand
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

# Joint indices within joint_pos_norm[21] (CONTROLLED_JOINTS order).
ELBOW_RIGHT_IDX: int = 17
ELBOW_LEFT_IDX: int = 20

# Guard thresholds
ELBOW_GUARD_THRESH: float = 0.5      # elbow_norm >= this counts as flexed
HAND_HEIGHT_LO: float = -0.25        # below torso root — too low (dangling)
HAND_HEIGHT_HI: float = 0.35         # above torso root — too high
HAND_DIST_MAX: float = 0.55          # hand→shoulder dist at which guard → 0


class GuardPostureObserver(BaseObserverPlugin):
    """Per-agent guard-posture scorer for r_guard channel."""

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = str(agent_id)
        self._left_elbow: float = 0.0
        self._right_elbow: float = 0.0
        self._left_guard: float = 0.0
        self._right_guard: float = 0.0
        self._left_hand_h: float = 0.0
        self._right_hand_h: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._left_elbow = 0.0
        self._right_elbow = 0.0
        self._left_guard = 0.0
        self._right_guard = 0.0
        self._left_hand_h = 0.0
        self._right_hand_h = 0.0

    @staticmethod
    def _score_hand(
        elbow_norm: float,
        hand_height: float,
        hand_shoulder_dist: float,
    ) -> float:
        """Score one hand's guard quality ∈ [0, 1]."""
        # Elbow flexion: 0 when straight, 1 when >= threshold
        elbow_score = min(1.0, max(0.0, elbow_norm / ELBOW_GUARD_THRESH))
        # Height band: peak at midpoint, taper at edges
        mid = (HAND_HEIGHT_LO + HAND_HEIGHT_HI) / 2
        half = (HAND_HEIGHT_HI - HAND_HEIGHT_LO) / 2
        height_score = max(0.0, 1.0 - abs(hand_height - mid) / half)
        # Proximity: hand shouldn't be too far from shoulder (extended)
        prox_score = max(0.0, 1.0 - hand_shoulder_dist / HAND_DIST_MAX)
        return elbow_score * height_score * prox_score

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        self_core = core_state.get(self.agent_id, {})
        joint_pos_norm = self_core.get("joint_pos_norm")
        if joint_pos_norm is not None:
            jpn = np.asarray(joint_pos_norm, dtype=np.float64)
            if jpn.shape[0] > ELBOW_LEFT_IDX:
                self._left_elbow = float(jpn[ELBOW_LEFT_IDX])
                self._right_elbow = float(jpn[ELBOW_RIGHT_IDX])

        derived = ctx.accessor.get_derived_state([self.agent_id])
        self_view = derived.get(self.agent_id, {})
        bx = self_view.get("body_xpos", {})
        torso_key = f"torso_{self.agent_id[-1]}"
        torso_pos = bx.get(torso_key)
        if torso_pos is None:
            # fallback: use root z
            self._left_hand_h = 0.0
            self._right_hand_h = 0.0
            self._left_guard = 0.0
            self._right_guard = 0.0
            return

        suffix = self.agent_id[-1]
        hl = bx.get(f"hand_left_{suffix}")
        hr = bx.get(f"hand_right_{suffix}")
        sl = bx.get(f"upper_arm_left_{suffix}")
        sr = bx.get(f"upper_arm_right_{suffix}")

        torso_z = float(np.asarray(torso_pos)[2])

        if hl is not None and sl is not None:
            hl = np.asarray(hl, dtype=np.float64)
            sl = np.asarray(sl, dtype=np.float64)
            self._left_hand_h = float(hl[2] - torso_z)
            d = float(np.linalg.norm(hl - sl))
            self._left_guard = self._score_hand(
                self._left_elbow, self._left_hand_h, d)
        else:
            self._left_hand_h = 0.0
            self._left_guard = 0.0

        if hr is not None and sr is not None:
            hr = np.asarray(hr, dtype=np.float64)
            sr = np.asarray(sr, dtype=np.float64)
            self._right_hand_h = float(hr[2] - torso_z)
            d = float(np.linalg.norm(hr - sr))
            self._right_guard = self._score_hand(
                self._right_elbow, self._right_hand_h, d)
        else:
            self._right_hand_h = 0.0
            self._right_guard = 0.0

    def get_output(self) -> Dict[str, Any]:
        return {
            "left_elbow_norm": self._left_elbow,
            "right_elbow_norm": self._right_elbow,
            "left_guard": self._left_guard,
            "right_guard": self._right_guard,
            "guard_score": (self._left_guard + self._right_guard) / 2,
            "left_hand_height": self._left_hand_h,
            "right_hand_height": self._right_hand_h,
        }
