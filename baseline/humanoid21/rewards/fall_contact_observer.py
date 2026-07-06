"""Per-step non-foot-grounded contact observer for humanoid21.

Records a boolean per action step indicating whether any non-foot body part
of the target robot is in contact with the ground. This data is consumed
post-rollout by the experiment's ``extract_rewards`` to implement a dense
fall/recovery reward via an offline state machine.

The contact detection logic mirrors ``ImbalanceTerminationPlugin2`` but
without any tolerance counter or termination — it simply records the raw
per-step contact state.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class FallContactObserver(BaseObserverPlugin):
    """Records per-step ``is_non_foot_grounded`` boolean for offline fall/reward analysis."""

    FOOT_BODY_NAMES = {'foot_left', 'foot_right'}

    def __init__(self, agent_id: str = "robot_a", force_threshold: float = 1.0):
        self.agent_id = str(agent_id)
        self.force_threshold = float(force_threshold)
        self._ground_geom_name: Optional[str] = None
        self._is_non_foot_grounded: bool = False

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "force_threshold": self.force_threshold,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "FallContactObserver":
        return cls(**config)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')
        self._is_non_foot_grounded = False

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._is_non_foot_grounded = self._check_non_foot_grounded(ctx)

    def get_output(self) -> Dict[str, float]:
        return {
            "is_non_foot_grounded": 1.0 if self._is_non_foot_grounded else 0.0,
        }

    def _check_non_foot_grounded(self, ctx: ReadOnlySimContext) -> bool:
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')
        if cv is None or cv['ncon'] == 0:
            return False

        static_data = ctx.accessor.get_static_data()
        body_id_to_name = static_data.get('body_id_to_name', {})
        geom_id_to_name = static_data.get('geom_id_to_name', {})
        ground_geom = self._ground_geom_name or 'ground'

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
            if float(force_mag[i]) < self.force_threshold:
                continue
            if not any(foot in body_robot for foot in self.FOOT_BODY_NAMES):
                return True

        return False
