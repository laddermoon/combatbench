"""4-stage standup potential-based rewarder.

Design based on the natural human standup process:
  Stage 0: Roll over to prone (face-down)  — guided by f_down
  Stage 1: Prone, exploring hand/foot support — flat potential, exploration
  Stage 2: Hands+feet support (or feet only, height not enough) — guided by height+uprightness
  Stage 3: Both feet standing — guided by height+uprightness
  Stage 4: Feet narrowed stable stand — guided by stability + narrowness

Potential ranges (continuous at all stage boundaries):
  Stage 0: [0.00, 0.15]   Stage 1: 0.15 (flat)
  Stage 2: [0.15, 0.40]   Stage 3: [0.40, 0.70]
  Stage 4: Stage 3 base + [0.00, 0.30] narrow/stable bonus → [0.40, 1.00]

Thresholds follow the final working config of the original 9-stage experiment
(S8/S9): success_height=0.60, success_uprightness=0.70.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

# Thresholds (from original S8/S9 final working config)
F_PRONE = 0.5       # f_down threshold: "significantly face-down"
H_STAND = 0.60      # pelvis height to count as "standing"
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

        h_pelvis = float(core_state["root_pos"][2])

        u_torso = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

        # Torso orientation: f_down = 1.0 means perfect prone (face down)
        static_data = ctx.accessor.get_static_data()[self.agent_id]
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
        body_xpos = derived_state.get("body_xpos", {})
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

        # ---- Stage 4: Narrow stable stand [0.40+bonus, 1.00] ----
        # Both feet, no hands, no other contact, standing height+uprightness,
        # feet narrowed, low velocity.
        # Potential = Stage 3 base + narrow/stable bonus (continuous with Stage 3).
        if (foot_l and foot_r and not has_hand and not other
                and h_pelvis >= H_STAND and u_torso >= U_STAND
                and d_feet < D_NARROW and mean_abs_joint_vel < V_STABLE):
            stage = 4
            h_score = float(np.clip((h_pelvis - H_STAND) / 0.20, 0.0, 1.0))
            u_score = float(np.clip((u_torso - U_STAND) / 0.20, 0.0, 1.0))
            v_score = float(np.exp(-mean_abs_joint_vel))
            narrow_score = float(np.clip((D_NARROW - d_feet) / D_NARROW, 0.0, 1.0))
            base = 0.40 + 0.30 * h_score * u_score
            potential = base + 0.30 * h_score * u_score * v_score * narrow_score

        # ---- Stage 3: Both feet standing [0.40, 0.70] ----
        # Both feet, no hands, no other, standing height+uprightness.
        elif (foot_l and foot_r and not has_hand and not other
                and h_pelvis >= H_STAND and u_torso >= U_STAND):
            stage = 3
            h_score = float(np.clip((h_pelvis - H_STAND) / 0.20, 0.0, 1.0))
            u_score = float(np.clip((u_torso - U_STAND) / 0.20, 0.0, 1.0))
            potential = 0.40 + 0.30 * h_score * u_score

        # ---- Stage 2: Hands+feet support OR feet-only below stand threshold [0.15, 0.40] ----
        # Case A: hands and feet both on ground (push-up position)
        # Case B: feet on ground, no hands, no other, but height/uprightness below Stage 3
        #         (height not enough → still Stage 2, per user requirement)
        # Lenient on knee/other contact for Case A (matching original V1 Stage 2).
        elif (has_hand and has_foot) or (
            has_foot and not has_hand and not other
            and (h_pelvis < H_STAND or u_torso < U_STAND)
        ):
            stage = 2
            h_score = float(np.clip((h_pelvis - 0.20) / 0.40, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.0) / 0.70, 0.0, 1.0))
            potential = 0.15 + 0.25 * h_score * u_score

        # ---- Stage 1: Prone (rolled over), no support yet — flat 0.15 ----
        # f_down above threshold but no hand+foot support.
        # Exploration: no gradient, robot finds its own way to push up.
        elif f_down >= F_PRONE:
            stage = 1
            potential = 0.15

        # ---- Stage 0: Not rolled over [0.00, 0.15] ----
        # Guide rolling to prone via f_down.
        else:
            stage = 0
            f_score = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))
            potential = 0.15 * f_score

        self._stage = stage
        self._potential = potential
        self._foot_distance = d_feet

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, bool]:
        """Parse ground contacts for feet, hands, shins, other."""
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]

        contacts = {
            "foot_left": False,
            "foot_right": False,
            "hand_left": False,
            "hand_right": False,
            "shin_left": False,
            "shin_right": False,
            "has_other_contact": False,
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

        shin_left_body = f"shin_left{suffix}"
        shin_right_body = f"shin_right{suffix}"
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
                elif body_robot == shin_left_body:
                    contacts["shin_left"] = True
                elif body_robot == shin_right_body:
                    contacts["shin_right"] = True
                else:
                    contacts["has_other_contact"] = True

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
