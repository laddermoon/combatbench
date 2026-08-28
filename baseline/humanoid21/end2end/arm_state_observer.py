"""Arm-state observer plugin for humanoid21 punch shaping.

Outputs eight raw physical quantities per step (no state-machine logic):

  Left / right elbow normalized angle [-1, 1]:
    -1 = fully flexed (max retraction), +1 = fully extended (max extension)

  Left / right hand to opponent head 3D distance (meters, unnormalized)

  Left / right hand to own shoulder 3D distance (meters, unnormalized)

  Opponent head to own left / right shoulder 3D distance (meters, unnormalized)

The experiment's post-hoc state machine (``arm_state_machine.py``) consumes
these to produce per-step actor weights for four reward channels
(``r_left_elbow``, ``r_right_elbow``, ``r_left_hand_dist``,
``r_right_hand_dist``).

Joint index mapping (CONTROLLED_JOINTS order, 0-based):
    17 = elbow_right
    20 = elbow_left
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# Joint indices within joint_pos_norm[21] (CONTROLLED_JOINTS order).
ELBOW_RIGHT_IDX: int = 17
ELBOW_LEFT_IDX: int = 20


class ArmStateObserver(BaseObserverPlugin):
    """Per-agent arm state observer for punch-shaping rewards.

    Outputs (per step):
      left_elbow_norm          — left elbow normalized angle [-1, 1]
      right_elbow_norm         — right elbow normalized angle [-1, 1]
      left_hand_to_opp_head    — 3D distance, left hand → opponent head (m)
      right_hand_to_opp_head   — 3D distance, right hand → opponent head (m)
      left_hand_to_shoulder    — 3D distance, left hand → left shoulder (m)
      right_hand_to_shoulder   — 3D distance, right hand → right shoulder (m)
      opp_head_to_left_shoulder  — 3D distance, opponent head → left shoulder (m)
      opp_head_to_right_shoulder — 3D distance, opponent head → right shoulder (m)
    """

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = str(agent_id)
        self.opp_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"

        self._left_elbow: float = 0.0
        self._right_elbow: float = 0.0
        self._left_hand_opp_head: float = 0.0
        self._right_hand_opp_head: float = 0.0
        self._left_hand_shoulder: float = 0.0
        self._right_hand_shoulder: float = 0.0
        self._opp_head_left_shoulder: float = 0.0
        self._opp_head_right_shoulder: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._left_elbow = 0.0
        self._right_elbow = 0.0
        self._left_hand_opp_head = 0.0
        self._right_hand_opp_head = 0.0
        self._left_hand_shoulder = 0.0
        self._right_hand_shoulder = 0.0
        self._opp_head_left_shoulder = 0.0
        self._opp_head_right_shoulder = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        # --- Joint angles from core_state ---
        core_state = ctx.accessor.get_core_state()
        self_core = core_state.get(self.agent_id, {})
        joint_pos_norm = self_core.get("joint_pos_norm")
        if joint_pos_norm is not None:
            jpn = np.asarray(joint_pos_norm, dtype=np.float64)
            if jpn.shape[0] > ELBOW_LEFT_IDX:
                self._left_elbow = float(jpn[ELBOW_LEFT_IDX])
                self._right_elbow = float(jpn[ELBOW_RIGHT_IDX])
            else:
                self._left_elbow = 0.0
                self._right_elbow = 0.0
        else:
            self._left_elbow = 0.0
            self._right_elbow = 0.0

        # --- Body positions from derived_state ---
        derived = ctx.accessor.get_derived_state([self.agent_id, self.opp_id])
        self_view = derived.get(self.agent_id, {})
        opp_view = derived.get(self.opp_id, {})

        self_bx = self_view.get("body_xpos", {})
        opp_bx = opp_view.get("body_xpos", {})

        suffix = self.agent_id[-1]   # 'a' or 'b'
        opp_suffix = self.opp_id[-1]

        # Self body names
        hand_left_key = f"hand_left_{suffix}"
        hand_right_key = f"hand_right_{suffix}"
        shoulder_left_key = f"upper_arm_left_{suffix}"
        shoulder_right_key = f"upper_arm_right_{suffix}"

        # Opp body name
        opp_head_key = f"head_{opp_suffix}"

        hand_left = self_bx.get(hand_left_key)
        hand_right = self_bx.get(hand_right_key)
        shoulder_left = self_bx.get(shoulder_left_key)
        shoulder_right = self_bx.get(shoulder_right_key)
        opp_head = opp_bx.get(opp_head_key)

        if hand_left is None or hand_right is None or opp_head is None:
            self._left_hand_opp_head = 0.0
            self._right_hand_opp_head = 0.0
        else:
            hl = np.asarray(hand_left, dtype=np.float64)
            hr = np.asarray(hand_right, dtype=np.float64)
            oh = np.asarray(opp_head, dtype=np.float64)
            self._left_hand_opp_head = float(np.linalg.norm(hl - oh))
            self._right_hand_opp_head = float(np.linalg.norm(hr - oh))

        if hand_left is None or shoulder_left is None:
            self._left_hand_shoulder = 0.0
        else:
            hl = np.asarray(hand_left, dtype=np.float64)
            sl = np.asarray(shoulder_left, dtype=np.float64)
            self._left_hand_shoulder = float(np.linalg.norm(hl - sl))

        if hand_right is None or shoulder_right is None:
            self._right_hand_shoulder = 0.0
        else:
            hr = np.asarray(hand_right, dtype=np.float64)
            sr = np.asarray(shoulder_right, dtype=np.float64)
            self._right_hand_shoulder = float(np.linalg.norm(hr - sr))

        if opp_head is None or shoulder_left is None:
            self._opp_head_left_shoulder = 0.0
        else:
            oh = np.asarray(opp_head, dtype=np.float64)
            sl = np.asarray(shoulder_left, dtype=np.float64)
            self._opp_head_left_shoulder = float(np.linalg.norm(oh - sl))

        if opp_head is None or shoulder_right is None:
            self._opp_head_right_shoulder = 0.0
        else:
            oh = np.asarray(opp_head, dtype=np.float64)
            sr = np.asarray(shoulder_right, dtype=np.float64)
            self._opp_head_right_shoulder = float(np.linalg.norm(oh - sr))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_output(self) -> Dict[str, float]:
        return {
            "left_elbow_norm": self._left_elbow,
            "right_elbow_norm": self._right_elbow,
            "left_hand_to_opp_head": self._left_hand_opp_head,
            "right_hand_to_opp_head": self._right_hand_opp_head,
            "left_hand_to_shoulder": self._left_hand_shoulder,
            "right_hand_to_shoulder": self._right_hand_shoulder,
            "opp_head_to_left_shoulder": self._opp_head_left_shoulder,
            "opp_head_to_right_shoulder": self._opp_head_right_shoulder,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ArmStateObserver":
        return cls(**config)
