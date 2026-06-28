"""Termination plugin for standup training.

Terminates episode early when:
- **Success**: robot height and uprightness exceed thresholds for ``success_hold_steps``
  AND robot is NOT touching any wall (must maintain free balance).
- **Give-up**: robot stays low for ``stagnation_steps`` consecutive steps
  (avoids wasting rollout budget on hopeless states).

Reads physical state directly from ctx.accessor (no dependency on observer plugins).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BasePlugin, SimContext


class StandupTerminationPlugin(BasePlugin):
    """Early termination for standup-from-fall training."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        success_height: float = 0.60,
        success_uprightness: float = 0.70,
        success_hold_steps: int = 50,
        stagnation_height: float = 0.25,
        stagnation_steps: int = 150,
    ) -> None:
        self.agent_id = str(agent_id)
        self.success_height = float(success_height)
        self.success_uprightness = float(success_uprightness)
        self.success_hold_steps = max(1, int(success_hold_steps))
        self.stagnation_height = float(stagnation_height)
        self.stagnation_steps = max(1, int(stagnation_steps))
        self._success_streak = 0
        self._stagnation_streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standup_termination"

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "success_height": self.success_height,
            "success_uprightness": self.success_uprightness,
            "success_hold_steps": self.success_hold_steps,
            "stagnation_height": self.stagnation_height,
            "stagnation_steps": self.stagnation_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupTerminationPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._success_streak = 0
        self._stagnation_streak = 0

    def _check_wall_contact(self, ctx: SimContext) -> bool:
        """Check if robot is touching any non-ground environment geometry (wall, post, etc.)."""
        derived_contacts = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_contacts.get('contacts')
        if cv is None or cv['ncon'] == 0:
            return False

        static_all = ctx.accessor.get_static_data()
        geom_id_to_name = static_all.get('geom_id_to_name', {})

        robot_aff = 1 if self.agent_id == 'robot_a' else 2
        aff1 = cv['aff1']
        aff2 = cv['aff2']
        geom1 = cv['geom1']
        geom2 = cv['geom2']
        force_mag = cv['force_mag']

        for i in range(cv['ncon']):
            if aff1[i] == 0 and aff2[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom1[i]), '')
            elif aff2[i] == 0 and aff1[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom2[i]), '')
            else:
                continue

            if float(force_mag[i]) < 1.0:
                continue
            if geom_env != 'ground':
                return True
        return False

    def on_post_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

        wall_contact = self._check_wall_contact(ctx)

        # Success: standing tall and upright AND not touching wall
        if (height >= self.success_height
                and uprightness >= self.success_uprightness
                and not wall_contact):
            self._success_streak += 1
            self._stagnation_streak = 0
            if self._success_streak >= self.success_hold_steps:
                ctx.request_termination("standup_success")
        else:
            self._success_streak = 0

        # Stagnation: stuck on the ground
        if height < self.stagnation_height:
            self._stagnation_streak += 1
            if self._stagnation_streak >= self.stagnation_steps:
                ctx.request_termination("standup_stagnation")
        else:
            self._stagnation_streak = 0
