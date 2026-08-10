"""3-stage standing-balance potential rewarder (dead-zone-free design).

Stage Definitions
-----------------
Stage 1 (rollover):
    从非 Stage 2/3 的状态 → 到俯身趴在地上的状态。
    Entry gate: none (always starts here at episode reset).
    Target signal: f_score (torso face-down orientation).
    The robot starts in a random fallen state (often face-up); it must
    roll over to a prone position.

Stage 2 (establish support):
    从俯身趴在地 → 到只有手和脚接触地面的支撑状态。
    Entry gate: f_score ≥ F_ENTER (rollover achieved).
    Target signal: contact_score (hand/foot proximity × no-extra-contact).
    The robot must bring hands/feet toward the ground AND lift other body
    parts (knees, torso, etc.) off the ground.  Only when extra_contact_count
    == 0 (only hands/feet touching) can it advance to Stage 3.

Stage 3 (close hand-foot distance):
    从任意的手脚支撑状态 → 到双手双脚靠近的手脚支撑状态。
    Entry gate: only hands/feet on ground (extra_contact_count == 0),
                regardless of torso orientation.
    Target signal: d_score (hand-midpoint to foot-midpoint distance).
    The robot must close the distance between hand-midpoint and foot-midpoint
    while maintaining only-hands-feet support.

Potential Bands
---------------
  Stage 1 [0, 0.33):    potential = 0.33 * f_score
  Stage 2 [0.33, 0.66): potential = 0.33 + 0.33 * contact_score
  Stage 3 [0.66, 1.0]:  potential = 0.66 + 0.34 * d_score

Stage is determined top-down each step: check Stage 3 first (only-hf contact),
then Stage 2 (f_score ≥ F_ENTER), else Stage 1.  No hysteresis needed since
Stage 3 does not depend on f_score.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

# --- Tunable constants ---
H_HAND_MAX = 0.3   # hand height (m) at which proximity saturates to 0
H_FOOT_MAX = 0.3   # foot height (m) at which proximity saturates to 0
D_MAX = 1.0        # hand-foot midpoint distance (m) at which d_score saturates to 0
D_MIN = 0.2        # hand-foot midpoint distance (m) at which d_score saturates to 1.0
OTHER_PENALTY_K = 0.5  # soft penalty coefficient for extra body contacts

# --- Stage gates ---
# Stage 3: only hands/feet on ground (no extra contacts), regardless of
#           torso orientation.  Crouching tilts the torso but that's fine.
# Stage 2: f_score ≥ F_ENTER (rollover achieved, but still has extra contacts).
# Stage 1: everything else (not yet rolled over).
F_ENTER = 0.8      # f_score needed to enter Stage 2 (f_down >= 0.6, clearly prone)


class StandingBalance3StageRewarder(BaseObserverPlugin):
    """Dead-zone-free 3-stage standing-balance potential function."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._stage: int = 1
        self._potential: float = 0.0
        self._f_score: float = 0.0
        self._contact_score: float = 0.0
        self._d_score: float = 0.0
        self._d_hf: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 1
        self._potential = 0.0
        self._f_score = 0.0
        self._contact_score = 0.0
        self._d_score = 0.0
        self._d_hf = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        static_data = ctx.accessor.get_static_data()[self.agent_id]

        # --- Signal 1: f_down (rollover orientation) ---
        torso_body_name = static_data["keypoint_body_names"]["torso"]
        body_xquat_dict = derived_state.get("body_xquat", {})
        q_torso = body_xquat_dict.get(
            torso_body_name, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        x_world_z = 2.0 * (x * z - w * y)
        f_down = -x_world_z
        f_score = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))

        # --- Signal 2: hand/foot proximity to ground + other-contact penalty ---
        body_xpos = derived_state.get("body_xpos", {})
        hand_l_name = static_data["keypoint_body_names"]["hand_left"]
        hand_r_name = static_data["keypoint_body_names"]["hand_right"]
        foot_l_name = static_data["keypoint_body_names"]["foot_left"]
        foot_r_name = static_data["keypoint_body_names"]["foot_right"]

        h_hand_l = float(body_xpos.get(hand_l_name, np.zeros(3, dtype=np.float32))[2])
        h_hand_r = float(body_xpos.get(hand_r_name, np.zeros(3, dtype=np.float32))[2])
        h_foot_l = float(body_xpos.get(foot_l_name, np.zeros(3, dtype=np.float32))[2])
        h_foot_r = float(body_xpos.get(foot_r_name, np.zeros(3, dtype=np.float32))[2])

        hand_proximity = float(np.clip(1.0 - (h_hand_l + h_hand_r) / (2.0 * H_HAND_MAX), 0.0, 1.0))
        foot_proximity = float(np.clip(1.0 - (h_foot_l + h_foot_r) / (2.0 * H_FOOT_MAX), 0.0, 1.0))
        support_score = (hand_proximity + foot_proximity) / 2.0

        contacts = self._get_detailed_contacts(ctx)
        extra_count = contacts["extra_contact_count"]
        other_penalty = 1.0 / (1.0 + OTHER_PENALTY_K * extra_count)

        contact_score = support_score * other_penalty

        # --- Signal 3: hand-foot midpoint distance ---
        hand_mid = (
            body_xpos.get(hand_l_name, np.zeros(3, dtype=np.float32))[:3]
            + body_xpos.get(hand_r_name, np.zeros(3, dtype=np.float32))[:3]
        ) / 2.0
        foot_mid = (
            body_xpos.get(foot_l_name, np.zeros(3, dtype=np.float32))[:3]
            + body_xpos.get(foot_r_name, np.zeros(3, dtype=np.float32))[:3]
        ) / 2.0
        d_hf = float(np.linalg.norm(hand_mid - foot_mid))
        d_score = float(np.clip((D_MAX - d_hf) / (D_MAX - D_MIN), 0.0, 1.0))

        # --- Stage determination (top-down, before potential) ---
        # Stage 3: only hands/feet touching ground, regardless of orientation.
        # Stage 2: rolled over (f_score ≥ F_ENTER) but still has extra contacts.
        # Stage 1: not yet rolled over.
        hf_contact = (
            contacts["hand_left"] or contacts["hand_right"]
            or contacts["foot_left"] or contacts["foot_right"]
        )
        only_hf_contact = (extra_count == 0 and hf_contact)

        if only_hf_contact:
            stage = 3
        elif f_score >= F_ENTER:
            stage = 2
        else:
            stage = 1

        # --- Stage-dependent potential ---
        # Each stage owns a [lo, hi] band and optimizes only its own target
        # signal.  Earlier stages are locked at their band ceiling, giving a
        # clean focused gradient with no interference between stages.
        if stage == 1:
            potential = 0.33 * f_score
        elif stage == 2:
            potential = 0.33 + 0.33 * contact_score
        else:
            potential = 0.66 + 0.34 * d_score

        self._stage = stage
        self._potential = float(potential)
        self._f_score = f_score
        self._contact_score = contact_score
        self._d_score = d_score
        self._d_hf = d_hf

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, Any]:
        """Parse ground contacts for feet, hands, and other body parts."""
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]

        contacts = {
            "foot_left": False,
            "foot_right": False,
            "hand_left": False,
            "hand_right": False,
            "has_other_contact": False,
            "extra_contact_count": 0,
            "extra_contact_bodies": set(),
        }

        if cv is None or cv['ncon'] == 0:
            return contacts

        foot_left_body = keypoint_names["foot_left"]
        foot_right_body = keypoint_names["foot_right"]
        hand_left_body = keypoint_names["hand_left"]
        hand_right_body = keypoint_names["hand_right"]

        static_all = ctx.accessor.get_static_data()
        body_id_to_name = static_all.get('body_id_to_name', {})
        geom_id_to_name = static_all.get('geom_id_to_name', {})

        robot_aff = 1 if self.agent_id == 'robot_a' else 2

        aff1 = cv['aff1']
        aff2 = cv['aff2']
        geom1 = cv['geom1']
        geom2 = cv['geom2']
        body1 = cv['body1']
        body2 = cv['body2']
        force_mag = cv['force_mag']

        for i in range(cv['ncon']):
            if aff1[i] == 0 and aff2[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom1[i]), '')
                body_robot = body_id_to_name.get(int(body2[i]), '')
            elif aff2[i] == 0 and aff1[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom2[i]), '')
                body_robot = body_id_to_name.get(int(body1[i]), '')
            else:
                continue

            if float(force_mag[i]) < 1.0:
                continue

            if geom_env == 'ground':
                if body_robot == foot_left_body:
                    contacts["foot_left"] = True
                elif body_robot == foot_right_body:
                    contacts["foot_right"] = True
                elif body_robot == hand_left_body:
                    contacts["hand_left"] = True
                elif body_robot == hand_right_body:
                    contacts["hand_right"] = True
                else:
                    contacts["has_other_contact"] = True
                    contacts["extra_contact_bodies"].add(body_robot)

        contacts["extra_contact_count"] = len(contacts["extra_contact_bodies"])

        return contacts

    def get_output(self) -> Dict[str, float]:
        return {
            "stage": float(self._stage),
            "potential": self._potential,
            "f_score": self._f_score,
            "contact_score": self._contact_score,
            "d_score": self._d_score,
            "d_hf": self._d_hf,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingBalance3StageRewarder":
        return cls(**config)
