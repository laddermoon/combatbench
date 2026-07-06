"""Standup v3 dual potential rewarder.

Computes both a smooth (gapless) and a gapped potential in one pass.
The experiment chooses which potential to use for PBRS at reward extraction time,
enabling phase-based training without environment restart.

Smooth potential (Phase A — early training):
  No gaps between stages. Continuous gradient from 0 to 1.
  Wall-aware: wall-assisted poses get capped low.
  Lower Stage 5 thresholds (h>0.60, u>0.70) — achievable from zero.

  Stage 5: [0.75, 1.00] — both feet, no hands, no wall, h>0.60, u>0.70
  Stage 4: [0.60, 0.75] — both feet, no hands, no wall
  Stage 3: [0.45, 0.60] — single foot, no hands, no wall
  Stage 2: [0.30, 0.45] — hands + feet (wall OK)
  Stage 1: [0.20, 0.30] — hands only (wall OK)
  Stage 0: [0.00, 0.20] — rollover

Gapped potential (Phase B — transition training):
  0.10 gap at Stage 3→4 boundary rewards risky second-foot placement.
  Velocity gate on Stage 5 prevents jump-up exploit.
  Wall-assisted standing capped below free Stage 3.

  Stage 5:   [0.90, 1.00] — both feet, no wall, low vel, h>0.60, u>0.70
  Stage 4.5: [0.80, 0.85] — high-vel standing (transition)
  Stage 4:   [0.65, 0.85] — both feet, no wall
  Stage 3.5: [0.55, 0.65] — both feet + knee (transition tolerance)
  Stage 3:   [0.40, 0.55] — single foot, no wall
  Stage 2:   [0.25, 0.40] — hands + feet
  Stage 1:   [0.15, 0.25] — hands only
  Stage 0:   [0.00, 0.15] — rollover
  Wall-assisted: [0.30, 0.48] — capped below free Stage 3
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class StandupPotentialRewarderV3(BaseObserverPlugin):
    """Dual-mode potential rewarder for phased standup training."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._stage: int = 0
        self._potential_smooth: float = 0.0
        self._potential_gapped: float = 0.0
        self._has_wall: bool = False

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 0
        self._potential_smooth = 0.0
        self._potential_gapped = 0.0
        self._has_wall = False

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

        foot_l = contacts["foot_left"]
        foot_r = contacts["foot_right"]
        knee_l = contacts["shin_left"]
        knee_r = contacts["shin_right"]
        hand_l = contacts["hand_left"]
        hand_r = contacts["hand_right"]
        other = contacts["has_other_contact"]
        has_wall = contacts["has_wall_contact"]

        has_hand = hand_l or hand_r
        has_foot = foot_l or foot_r
        has_knee = knee_l or knee_r

        # =================================================================
        # Smooth potential (V1-style, no gaps, wall-aware)
        # =================================================================
        ps = 0.0
        stage_s = 0

        if foot_l and foot_r and not has_hand and not other and not has_wall and u_torso > 0.70 and h_pelvis > 0.60:
            stage_s = 5
            h_s = float(np.clip((h_pelvis - 0.60) / 0.20, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.70) / 0.20, 0.0, 1.0))
            v_s = float(np.exp(-mean_abs_joint_vel))
            ps = 0.75 + 0.25 * h_s * u_s * v_s
        elif foot_l and foot_r and not has_hand and not other and not has_wall:
            stage_s = 4
            h_s = float(np.clip((h_pelvis - 0.40) / 0.35, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.30) / 0.55, 0.0, 1.0))
            ps = 0.60 + 0.15 * h_s * u_s
        elif foot_l and foot_r and not has_hand and not other and has_wall:
            stage_s = 3
            h_s = float(np.clip((h_pelvis - 0.40) / 0.35, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.30) / 0.55, 0.0, 1.0))
            ps = 0.45 + 0.10 * h_s * u_s
        elif has_foot and not has_hand and not other and not has_wall:
            stage_s = 3
            h_s = float(np.clip((h_pelvis - 0.40) / 0.35, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.30) / 0.55, 0.0, 1.0))
            ps = 0.45 + 0.15 * h_s * u_s
        elif has_hand and has_foot:
            stage_s = 2
            h_s = float(np.clip((h_pelvis - 0.20) / 0.40, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.0) / 0.80, 0.0, 1.0))
            ps = 0.30 + 0.15 * h_s * u_s
        elif has_hand:
            stage_s = 1
            h_s = float(np.clip((h_pelvis - 0.15) / 0.30, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.0) / 0.80, 0.0, 1.0))
            ps = 0.20 + 0.10 * h_s * u_s
        else:
            stage_s = 0
            f_s = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))
            ps = 0.20 * f_s

        # =================================================================
        # Gapped potential (V2-style, with gaps, velocity gate, wall-aware)
        # =================================================================
        pg = 0.0
        stage_g = 0

        if (foot_l and foot_r and not has_hand and u_torso > 0.70
                and h_pelvis > 0.60 and not other and not has_wall
                and mean_abs_joint_vel < 2.0):
            stage_g = 5
            h_s = float(np.clip((h_pelvis - 0.60) / 0.20, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.70) / 0.20, 0.0, 1.0))
            v_s = float(np.exp(-mean_abs_joint_vel))
            pg = 0.90 + 0.10 * h_s * u_s * (v_s ** 3)
        elif (foot_l and foot_r and not has_hand and u_torso > 0.70
                and h_pelvis > 0.60 and not other and has_wall
                and mean_abs_joint_vel < 2.0):
            stage_g = 3
            h_s = float(np.clip((h_pelvis - 0.60) / 0.20, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.70) / 0.20, 0.0, 1.0))
            v_s = float(np.exp(-mean_abs_joint_vel))
            pg = 0.40 + 0.08 * h_s * u_s * v_s
        elif (foot_l and foot_r and not has_hand and u_torso > 0.70
                and h_pelvis > 0.60 and not other and not has_wall
                and mean_abs_joint_vel >= 2.0):
            stage_g = 4
            h_s = float(np.clip((h_pelvis - 0.60) / 0.20, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.70) / 0.20, 0.0, 1.0))
            v_s = float(np.exp(-mean_abs_joint_vel))
            pg = 0.80 + 0.05 * h_s * u_s * v_s
        elif foot_l and foot_r and not has_hand and not other and not has_wall:
            stage_g = 4
            h_s = float(np.clip((h_pelvis - 0.35) / 0.40, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.20) / 0.65, 0.0, 1.0))
            pg = 0.65 + 0.20 * h_s * u_s
        elif foot_l and foot_r and not has_hand and not other and has_wall:
            stage_g = 2
            h_s = float(np.clip((h_pelvis - 0.25) / 0.50, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.10) / 0.75, 0.0, 1.0))
            pg = 0.30 + 0.08 * h_s * u_s
        elif foot_l and foot_r and has_knee and not has_hand and not other and not has_wall:
            stage_g = 4
            h_s = float(np.clip((h_pelvis - 0.25) / 0.50, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.10) / 0.75, 0.0, 1.0))
            pg = 0.55 + 0.10 * h_s * u_s
        elif has_foot and not has_hand and not other and not has_wall:
            stage_g = 3
            h_s = float(np.clip((h_pelvis - 0.35) / 0.40, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.20) / 0.65, 0.0, 1.0))
            pg = 0.40 + 0.15 * h_s * u_s
        elif has_hand and has_foot:
            stage_g = 2
            h_s = float(np.clip((h_pelvis - 0.20) / 0.40, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.0) / 0.80, 0.0, 1.0))
            pg = 0.25 + 0.15 * h_s * u_s
        elif has_hand:
            stage_g = 1
            h_s = float(np.clip((h_pelvis - 0.15) / 0.30, 0.0, 1.0))
            u_s = float(np.clip((u_torso - 0.0) / 0.80, 0.0, 1.0))
            pg = 0.15 + 0.10 * h_s * u_s
        else:
            stage_g = 0
            f_s = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))
            pg = 0.15 * f_s

        self._stage = max(stage_s, stage_g)
        self._potential_smooth = ps
        self._potential_gapped = pg
        self._has_wall = has_wall

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
            "has_wall_contact": False,
            "wall_hand_contact": False,
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

            if geom_env == ground_geom:
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
            else:
                contacts["has_wall_contact"] = True
                if body_robot in (hand_left_body, lower_arm_left_body,
                                  hand_right_body, lower_arm_right_body):
                    contacts["wall_hand_contact"] = True

        return contacts

    def get_output(self) -> Dict[str, float]:
        return {
            "stage": float(self._stage),
            "potential_smooth": self._potential_smooth,
            "potential_gapped": self._potential_gapped,
            "has_wall_contact": 1.0 if self._has_wall else 0.0,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupPotentialRewarderV3":
        return cls(**config)
