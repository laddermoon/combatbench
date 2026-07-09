"""4-stage standup potential-based rewarder.

Design based on the natural human standup process:
  Stage 0: Not rolled over — guided by f_down orientation
  Stage 1: Rolled over (prone) — guided by torso height [0.12, 0.40] +
           pelvis height [0.12, 0.50], product shapes both-lift gradient
  Stage 2: Both feet on ground, hands optional, no other contact —
           guided by torso height + uprightness
  Stage 3: Foot on ground, no hands, no other, torso height >= 0.85m
           and uprightness >= cos(30°) ≈ 0.866
  Stage 4: No hands, no other, torso height >= H_STAND (1.1m) — fixed

Height metric: h_torso = root_pos[2] (MuJoCo root body = torso).
Pelvis height from body_xpos for Stage 1 hip-lift guidance.

Potential ranges (all stage boundaries continuous):
  Stage 0: [0.00, 0.15]   Stage 1: [0.15, 0.40]
  Stage 2: [0.40, 0.70]   Stage 3: [0.70, 1.00]   Stage 4: 1.00

Reference heights (from simulation):
  standing: torso=1.282, pelvis=0.857
  squat:    torso=0.596, pelvis=0.239
  prone:    torso=0.076, pelvis=0.071
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

# Thresholds (from original S8/S9 final working config)
F_PRONE = 0.5    # f_down threshold: 45° from downward = "significantly face-down"
H_STAND = 1.1       # torso height to count as "standing"
U_STAND = 0.70      # uprightness to count as "standing"
D_NARROW = 0.25     # foot distance for "narrow stand" (Stage 4 only)
V_STABLE = 2.0      # joint velocity threshold for "stable"


class Standup4StageRewarder(BaseObserverPlugin):
    """4-stage potential function for natural prone-to-stand training."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._stage: int = 0
        self._potential: float = 0.0
        self._foot_distance: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 0
        self._potential = 0.0
        self._foot_distance = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]

        h_torso = float(core_state["root_pos"][2])  # root body = torso in MuJoCo

        # Pelvis height from body_xpos
        static_data = ctx.accessor.get_static_data()[self.agent_id]
        body_xpos = derived_state.get("body_xpos", {})
        pelvis_body_name = static_data["keypoint_body_names"]["pelvis"]
        h_pelvis = float(body_xpos.get(pelvis_body_name, np.zeros(3, dtype=np.float32))[2])

        u_torso = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

        # Torso orientation: f_down = 1.0 means perfect prone (face down)
        torso_body_name = static_data["keypoint_body_names"]["torso"]
        body_xquat_dict = derived_state.get("body_xquat", {})
        q_torso = body_xquat_dict.get(
            torso_body_name, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        x_world_z = 2.0 * (x * z - w * y)
        f_down = -x_world_z

        # Joint velocity (stability metric)
        joint_vel = np.asarray(core_state["joint_vel_norm"], dtype=np.float32)
        mean_abs_joint_vel = float(np.mean(np.abs(joint_vel)))

        # Foot distance (horizontal plane)
        foot_l_name = static_data["keypoint_body_names"]["foot_left"]
        foot_r_name = static_data["keypoint_body_names"]["foot_right"]
        foot_l_pos = body_xpos.get(foot_l_name, np.zeros(3, dtype=np.float32))
        foot_r_pos = body_xpos.get(foot_r_name, np.zeros(3, dtype=np.float32))
        d_feet = float(np.linalg.norm(foot_l_pos[:2] - foot_r_pos[:2]))

        # Contacts
        contacts = self._get_detailed_contacts(ctx)
        foot_l = contacts["foot_left"]
        foot_r = contacts["foot_right"]
        hand_l = contacts["hand_left"]
        hand_r = contacts["hand_right"]
        other = contacts["has_other_contact"]

        has_hand = hand_l or hand_r
        has_foot = foot_l or foot_r

        # =================================================================
        # Stage determination (top-down priority, high stage first)
        # =================================================================

        stage = 0
        potential = 0.0

        # ---- Stage 4: Standing [1.00] ----
        # No hands, no other contact, torso height met. Fixed potential.
        if (not has_hand and not other
                and h_torso >= H_STAND):
            stage = 4
            potential = 1.0

        # ---- Stage 3: Standing, filtering to Stage 4 [0.70, 1.00] ----
        # Foot on ground, no hands, no other, torso height >= 0.85m and
        # uprightness >= cos(30°) ≈ 0.866 (gate conditions).
        # h_score: torso [0.85, H_STAND=1.10] → 0..1
        # potential = 0.70 + 0.30 * h_score
        # Transition from Stage 2: at h_torso=0.85, u_torso=0.866, Stage 2
        # potential = 0.70 = Stage 3 floor (continuous).
        # Transition to Stage 4: when h_torso >= H_STAND, h_score = 1.0,
        # potential = 1.00 = Stage 4 (continuous).
        elif (has_foot and not has_hand and not other
                and h_torso >= 0.85 and u_torso >= 0.866):
            stage = 3
            h_score = float(np.clip((h_torso - 0.85) / (H_STAND - 0.85), 0.0, 1.0))
            potential = 0.70 + 0.30 * h_score

        # ---- Stage 2: Both feet on ground, below stand threshold [0.40, 0.70] ----
        # Both feet on ground, no other contact. Hands optional (push-up allowed).
        # h_score: torso [0.15, 0.85] → 0..1 (saturates at Stage 3 height threshold)
        # u_score: uprightness [0.0, 0.866] → 0..1 (saturates at Stage 3 uprightness threshold)
        # potential = 0.40 + 0.30 * h_score * u_score
        # Transition to Stage 3: when h_torso >= 0.85 AND u_torso >= 0.866,
        # both scores = 1.0, potential = 0.70 = Stage 3 floor (continuous).
        elif foot_l and foot_r and not other:
            stage = 2
            h_score = float(np.clip((h_torso - 0.15) / 0.70, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.0) / 0.866, 0.0, 1.0))
            potential = 0.40 + 0.30 * h_score * u_score

        # ---- Stage 1: Rolled over — contact penalty [0.15, 0.40] ----
        # f_down above threshold but no both-feet support yet.
        # Start at 0.40, deduct 0.05 per non-hand/foot body part touching ground.
        # Floor at 0.15 (5 extra contacts → 0.15).
        # Rewards minimizing body contact with ground (clean posture).
        # Transition to Stage 2: when both feet touch ground (top-down priority),
        # switches to Stage 2 with floor 0.40.
        # Mutually exclusive: both-feet → Stage 2, else → Stage 1.
        elif f_down >= F_PRONE:
            stage = 1
            extra_count = contacts["extra_contact_count"]
            potential = max(0.15, 0.40 - 0.05 * extra_count)

        # ---- Stage 0: Arbitrary state (not rolled over) [0.00, 0.15] ----
        # Guide rolling over via f_down.
        else:
            stage = 0
            f_score = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))
            potential = 0.15 * f_score

        self._stage = stage
        self._potential = potential
        self._foot_distance = d_feet

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, bool]:
        """Parse ground contacts for feet, hands, and other."""
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

        suffix = ""
        if foot_left_body.endswith("_a"):
            suffix = "_a"
        elif foot_left_body.endswith("_b"):
            suffix = "_b"

        lower_arm_left_body = f"lower_arm_left{suffix}"
        lower_arm_right_body = f"lower_arm_right{suffix}"

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
                elif body_robot in (hand_left_body, lower_arm_left_body):
                    contacts["hand_left"] = True
                elif body_robot in (hand_right_body, lower_arm_right_body):
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
            "foot_distance": self._foot_distance,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "Standup4StageRewarder":
        return cls(**config)
