"""Standup v2 potential-based rewarder with transition gaps.

Key change from original StandupPotentialRewarder:
- Creates potential gaps at stage boundaries to reward risky transitions
- Stage 3 (single foot): [0.40, 0.55]  (was [0.45, 0.60])
- Stage 4 (double feet): [0.65, 0.85]  (was [0.60, 0.75])
- Stage 5 (perfect stand): [0.85, 1.00] (was [0.75, 1.00])
- Gap of 0.10 at 3→4 boundary provides explicit reward for placing second foot
- Relaxes Stage 4 to allow brief shin contact (transition tolerance)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class StandupPotentialRewarderV2(BaseObserverPlugin):
    """V2 potential function with transition gaps for standup training."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._stage: int = 0
        self._potential: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 0
        self._potential = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]

        h_pelvis = float(core_state["root_pos"][2])
        u_torso = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        torso_body_name = static_data["keypoint_body_names"]["torso"]

        body_xquat_dict = derived_state.get("body_xquat", {})
        q_torso = body_xquat_dict.get(
            torso_body_name, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        x_world_z = 2.0 * (x * z - w * y)
        f_down = -x_world_z

        joint_vel = np.asarray(core_state["joint_vel_norm"], dtype=np.float32)
        mean_abs_joint_vel = float(np.mean(np.abs(joint_vel)))

        contacts = self._get_detailed_contacts(ctx)

        stage = 0
        potential = 0.0

        foot_l = contacts["foot_left"]
        foot_r = contacts["foot_right"]
        knee_l = contacts["shin_left"]
        knee_r = contacts["shin_right"]
        hand_l = contacts["hand_left"]
        hand_r = contacts["hand_right"]
        other = contacts["has_other_contact"]

        has_hand = hand_l or hand_r
        has_foot = foot_l or foot_r
        has_knee = knee_l or knee_r

        # Stage 5: Perfect stand — both feet, upright, tall
        if foot_l and foot_r and not has_hand and u_torso > 0.70 and h_pelvis > 0.60 and not other:
            stage = 5
            h_score = float(np.clip((h_pelvis - 0.60) / 0.20, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.70) / 0.20, 0.0, 1.0))
            v_score = float(np.exp(-mean_abs_joint_vel))
            potential = 0.90 + 0.10 * h_score * u_score * v_score

        # Stage 4: Double feet standing (squat or low stand)
        # Relaxed: allow brief knee/shin contact during transition
        elif foot_l and foot_r and not has_hand and not other:
            stage = 4
            h_score = float(np.clip((h_pelvis - 0.35) / 0.40, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.20) / 0.65, 0.0, 1.0))
            potential = 0.65 + 0.20 * h_score * u_score

        # Stage 3.5: Double feet + brief knee contact (transition helper)
        elif foot_l and foot_r and has_knee and not has_hand and not other:
            stage = 4  # Count as stage 4 but lower potential
            h_score = float(np.clip((h_pelvis - 0.25) / 0.50, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.10) / 0.75, 0.0, 1.0))
            potential = 0.55 + 0.10 * h_score * u_score

        # Stage 3: Single foot stand, hands off
        elif has_foot and not has_hand and not other:
            stage = 3
            h_score = float(np.clip((h_pelvis - 0.35) / 0.40, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.20) / 0.65, 0.0, 1.0))
            potential = 0.40 + 0.15 * h_score * u_score

        # Stage 2: Hands + feet support
        elif has_hand and has_foot:
            stage = 2
            h_score = float(np.clip((h_pelvis - 0.20) / 0.40, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.0) / 0.80, 0.0, 1.0))
            potential = 0.25 + 0.15 * h_score * u_score

        # Stage 1: Hands only (push-up)
        elif has_hand:
            stage = 1
            h_score = float(np.clip((h_pelvis - 0.15) / 0.30, 0.0, 1.0))
            u_score = float(np.clip((u_torso - 0.0) / 0.80, 0.0, 1.0))
            potential = 0.15 + 0.10 * h_score * u_score

        # Stage 0: Rollover / belly-down
        else:
            stage = 0
            f_score = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))
            potential = 0.15 * f_score

        self._stage = stage
        self._potential = potential

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, bool]:
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]

        ground_geom = 'ground'

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

            if geom_env != ground_geom:
                continue
            if float(force_mag[i]) < 1.0:
                continue

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
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupPotentialRewarderV2":
        return cls(**config)
