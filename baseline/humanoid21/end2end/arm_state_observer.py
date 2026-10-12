"""Arm-state observer plugin for humanoid21 punch shaping.

Outputs eight raw physical quantities per step (no state-machine logic):

  Left / right elbow normalized angle [-1, 1]:
    -1 = fully extended (punched out), +1 = fully flexed (chambered)

  Left / right hand to opponent head 3D distance (meters, unnormalized)

  Left / right hand to own shoulder 3D distance (meters, unnormalized)

  Opponent head to own left / right shoulder 3D distance (meters, unnormalized)

  Left / right hand→opponent-head CLOSING SPEED (m/s, signed):
    -d(hand→opp_head dist)/dt — how fast the fist is closing on the
    target.  NOTE: this includes the OPPONENT's own motion — kept as a
    diagnostic field only.

  Left / right hand SPEED-TOWARD-OPPONENT (m/s, signed):
    v_hand · (opp_head − hand)/|·| — the hand's OWN world-frame
    velocity projected onto the hand→opp-head direction.  Arm swing
    AND body lunge both count (a real punch is a kinetic chain), but
    the opponent's motion does NOT — a stationary hand while the
    target drifts closer earns nothing.  Positive = fist moving at
    the target.  Unbounded — the punch_power stage's "as hard as
    possible" signal.

  Left / right SHOULDER SPEED-TOWARD-OPPONENT (m/s, signed):
    v_shoulder · (opp_head − shoulder)/|·| — the punching-side
    shoulder's own world velocity toward the target.  Shoulder speed
    is driven by hip rotation and leg drive — the proximal end of
    the kinetic chain.  An elbow-only flick leaves it near zero.

  TORSO YAW RATE (rad/s, signed, world z):
    ω_z of the torso body — how fast the waist twists.  Rotational
    chain drive shows up here directly even when the shoulder point's
    linear speed stays modest (short lever arm).

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
      left_hand_closing_speed    — -d(hand→opp_head)/dt, m/s, signed
      right_hand_closing_speed   — same for right hand
      left_hand_speed_to_opp     — v_hand · dir(hand→opp_head), m/s
      right_hand_speed_to_opp    — same for right hand
      left_shoulder_speed_to_opp  — v_shoulder · dir(shoulder→opp_head), m/s
      right_shoulder_speed_to_opp — same for right shoulder
      torso_yaw_rate             — torso ω_z, world frame, rad/s
    """

    def __init__(
        self,
        agent_id: str = "robot_a",
        action_dt: float = 0.05,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opp_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        # Action-step period for the finite-difference speed estimate:
        # phy_steps_per_action (25) × phy dt (0.002 s) = 0.05 s.
        self.action_dt = float(action_dt)

        self._left_elbow: float = 0.0
        self._right_elbow: float = 0.0
        self._left_hand_opp_head: float = 0.0
        self._right_hand_opp_head: float = 0.0
        self._left_hand_shoulder: float = 0.0
        self._right_hand_shoulder: float = 0.0
        self._opp_head_left_shoulder: float = 0.0
        self._opp_head_right_shoulder: float = 0.0
        self._prev_lh_dist: float | None = None
        self._prev_rh_dist: float | None = None
        self._left_closing_speed: float = 0.0
        self._right_closing_speed: float = 0.0
        self._prev_lh_pos: np.ndarray | None = None
        self._prev_rh_pos: np.ndarray | None = None
        self._left_speed_to_opp: float = 0.0
        self._right_speed_to_opp: float = 0.0
        self._prev_ls_pos: np.ndarray | None = None
        self._prev_rs_pos: np.ndarray | None = None
        self._left_shoulder_speed: float = 0.0
        self._right_shoulder_speed: float = 0.0
        self._torso_yaw_rate: float = 0.0

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
        self._prev_lh_dist = None
        self._prev_rh_dist = None
        self._left_closing_speed = 0.0
        self._right_closing_speed = 0.0
        self._prev_lh_pos = None
        self._prev_rh_pos = None
        self._left_speed_to_opp = 0.0
        self._right_speed_to_opp = 0.0
        self._prev_ls_pos = None
        self._prev_rs_pos = None
        self._left_shoulder_speed = 0.0
        self._right_shoulder_speed = 0.0
        self._torso_yaw_rate = 0.0

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
            self._left_closing_speed = 0.0
            self._right_closing_speed = 0.0
            self._prev_lh_dist = None
            self._prev_rh_dist = None
        else:
            hl = np.asarray(hand_left, dtype=np.float64)
            hr = np.asarray(hand_right, dtype=np.float64)
            oh = np.asarray(opp_head, dtype=np.float64)
            self._left_hand_opp_head = float(np.linalg.norm(hl - oh))
            self._right_hand_opp_head = float(np.linalg.norm(hr - oh))
            # Closing speed: -Δ(hand→opp_head dist)/Δt.  Equivalent to the
            # hand's world velocity projected onto the target direction —
            # the whole kinetic chain (swing + body lunge) counts.
            if self._prev_lh_dist is None:
                self._left_closing_speed = 0.0
                self._right_closing_speed = 0.0
            else:
                self._left_closing_speed = (
                    -(self._left_hand_opp_head - self._prev_lh_dist)
                    / self.action_dt
                )
                self._right_closing_speed = (
                    -(self._right_hand_opp_head - self._prev_rh_dist)
                    / self.action_dt
                )
            self._prev_lh_dist = self._left_hand_opp_head
            self._prev_rh_dist = self._right_hand_opp_head

            # Hand's own speed toward the target: v_hand · dir(hand→head).
            # Unlike closing_speed, the opponent's motion does not
            # contribute — only the fist's own movement pays.
            if self._prev_lh_pos is None:
                self._left_speed_to_opp = 0.0
                self._right_speed_to_opp = 0.0
            else:
                vl = (hl - self._prev_lh_pos) / self.action_dt
                vr = (hr - self._prev_rh_pos) / self.action_dt
                dir_l = oh - hl
                dir_r = oh - hr
                dir_l /= max(np.linalg.norm(dir_l), 1e-6)
                dir_r /= max(np.linalg.norm(dir_r), 1e-6)
                self._left_speed_to_opp = float(vl @ dir_l)
                self._right_speed_to_opp = float(vr @ dir_r)
            self._prev_lh_pos = hl
            self._prev_rh_pos = hr

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

        # Shoulder's own speed toward the target — proxy for proximal
        # kinetic-chain drive (hips/torso rotating the shoulder into
        # the punch).  Elbow-only flicks leave this near zero.
        if (
            opp_head is None
            or shoulder_left is None
            or shoulder_right is None
        ):
            self._left_shoulder_speed = 0.0
            self._right_shoulder_speed = 0.0
            self._prev_ls_pos = None
            self._prev_rs_pos = None
        else:
            oh = np.asarray(opp_head, dtype=np.float64)
            sl = np.asarray(shoulder_left, dtype=np.float64)
            sr = np.asarray(shoulder_right, dtype=np.float64)
            if self._prev_ls_pos is None:
                self._left_shoulder_speed = 0.0
                self._right_shoulder_speed = 0.0
            else:
                vsl = (sl - self._prev_ls_pos) / self.action_dt
                vsr = (sr - self._prev_rs_pos) / self.action_dt
                dir_sl = oh - sl
                dir_sr = oh - sr
                dir_sl /= max(np.linalg.norm(dir_sl), 1e-6)
                dir_sr /= max(np.linalg.norm(dir_sr), 1e-6)
                self._left_shoulder_speed = float(vsl @ dir_sl)
                self._right_shoulder_speed = float(vsr @ dir_sr)
            self._prev_ls_pos = sl
            self._prev_rs_pos = sr

        # Torso yaw rate (world-frame ω_z) — hip twist speed.
        self_av = self_view.get("body_angvel_world", {})
        torso_av = self_av.get(f"torso_{suffix}")
        if torso_av is None:
            self._torso_yaw_rate = 0.0
        else:
            self._torso_yaw_rate = float(
                np.asarray(torso_av, dtype=np.float64)[2]
            )

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
            "left_hand_closing_speed": self._left_closing_speed,
            "right_hand_closing_speed": self._right_closing_speed,
            "left_hand_speed_to_opp": self._left_speed_to_opp,
            "right_hand_speed_to_opp": self._right_speed_to_opp,
            "left_shoulder_speed_to_opp": self._left_shoulder_speed,
            "right_shoulder_speed_to_opp": self._right_shoulder_speed,
            "torso_yaw_rate": self._torso_yaw_rate,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "action_dt": self.action_dt}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ArmStateObserver":
        return cls(**config)
