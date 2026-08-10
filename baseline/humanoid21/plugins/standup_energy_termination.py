"""Termination plugin for energy-based standup training.

Terminates episode early with "success" when the robot maintains balance
on two feet (no hands, no other body contact) for ``balance_steps``
consecutive action steps.  Also terminates with "stagnation" if the robot
stays low for too long.

Action step = 25 phy steps × 0.002s = 0.05s, so 40 steps ≈ 2 seconds.

Balance detection is self-contained (does not depend on observer plugin
outputs) to avoid plugin execution-order coupling.
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BasePlugin, SimContext


class StandupEnergyTerminationPlugin(BasePlugin):
    """Early termination for energy-based standup-from-fall training."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        balance_steps: int = 40,
        stagnation_height: float = 0.25,
        stagnation_steps: int = 150,
    ) -> None:
        self.agent_id = str(agent_id)
        self.balance_steps = max(1, int(balance_steps))
        self.stagnation_height = float(stagnation_height)
        self.stagnation_steps = max(1, int(stagnation_steps))
        self._balance_streak = 0
        self._stagnation_streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standup_energy_termination"

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "balance_steps": self.balance_steps,
            "stagnation_height": self.stagnation_height,
            "stagnation_steps": self.stagnation_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupEnergyTerminationPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._balance_streak = 0
        self._stagnation_streak = 0

    def _check_clean(self, ctx: SimContext) -> bool:
        """Check no hands or other body parts touch the ground.
        Feet touching or not is irrelevant — only non-foot contacts disqualify."""
        cv = ctx.accessor.get_derived_state(['contacts']).get('contacts')
        if cv is None or cv['ncon'] == 0:
            return True

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]
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

        for i in range(cv['ncon']):
            if cv['aff1'][i] == 0 and cv['aff2'][i] == robot_aff:
                geom_env = geom_id_to_name.get(int(cv['geom1'][i]), '')
                body_robot = body_id_to_name.get(int(cv['body2'][i]), '')
            elif cv['aff2'][i] == 0 and cv['aff1'][i] == robot_aff:
                geom_env = geom_id_to_name.get(int(cv['geom2'][i]), '')
                body_robot = body_id_to_name.get(int(cv['body1'][i]), '')
            else:
                continue

            if float(cv['force_mag'][i]) < 1.0:
                continue

            if geom_env == 'ground':
                if body_robot in (foot_left_body, foot_right_body):
                    continue
                if body_robot in (hand_left_body, lower_arm_left_body,
                                  hand_right_body, lower_arm_right_body):
                    return False
                return False

        return True

    def on_post_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        height = float(core_state["root_pos"][2])

        if self._check_clean(ctx):
            self._balance_streak += 1
            if self._balance_streak >= self.balance_steps:
                ctx.request_termination("standup_success")
        else:
            self._balance_streak = 0

        if height < self.stagnation_height:
            self._stagnation_streak += 1
            if self._stagnation_streak >= self.stagnation_steps:
                ctx.request_termination("standup_stagnation")
        else:
            self._stagnation_streak = 0
