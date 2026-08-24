"""Observer plugin that outputs per-step foot heights and contact states.

Outputs four values per step:
  - h_left_foot:  left foot midpoint height above standing foot height
  - h_right_foot: right foot midpoint height above standing foot height
  - left_foot_contact:  whether left foot is in contact with ground
  - right_foot_contact: whether right foot is in contact with ground

Foot height is defined as the world z-coordinate of the foot body's
inertial frame origin (``body_xipos``), which coincides with the
midpoint of the two foot box geoms.  This is preferred over the ankle
joint anchor because it reflects the actual foot center position and
naturally rises when the foot tilts while still touching the ground
(one edge lifts, pulling the midpoint up).

Standing foot height (STANDING_FOOT_Z = 0.027 m) was obtained by
measuring ``body_xipos[foot_left_a][2]`` and ``body_xipos[foot_right_a][2]``
with the robot in the default standing pose::

    sim = Humanoid21Simulator(initial_distance=2.0)
    sim.reset(seed=42, options={'initial_pose_a': 'standing',
                                 'initial_pose_b': 'standing'})
    ds = sim.get_derived_state(['robot_a'])
    ds['robot_a']['body_xipos']['foot_left_a'][2]   # → 0.027
    ds['robot_a']['body_xipos']['foot_right_a'][2]  # → 0.027

Both robot_a and robot_b yield the same value.  The value is a
constant of the robot model (foot geom layout + default standing pose)
and does not depend on the simulator seed.

Contact detection reuses the same logic as
``CrossSupportBalanceRewarder._get_foot_contact_state``: a foot is
considered in contact when any contact entry pairs the foot body with
the ground geom.  No force threshold is applied.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# Standing foot midpoint height (m).
#
# This is the world z-coordinate of the foot body's inertial frame
# origin (``body_xipos``) when the robot stands in its default pose.
# The inertial frame origin coincides with the midpoint of the two
# foot box geoms (verified: local geom positions are [0.035,-0.02,0]
# and [0.035,0.02,0], midpoint [0.035,0,0] == body_ipos).
#
# h = body_xipos_z - STANDING_FOOT_Z
#   = 0  when standing
#   > 0  when foot is lifted or tilted (midpoint rises)
#   < 0  when squatting
STANDING_FOOT_Z: float = 0.027


class FootStateObserver(BaseObserverPlugin):
    """Per-agent foot state observer.

    Outputs foot midpoint heights (relative to standing) and ground
    contact booleans for both feet of the specified agent.
    """

    def __init__(self, agent_id: str = "robot_a", standing_foot_z: float = STANDING_FOOT_Z):
        self.agent_id = str(agent_id)
        self.standing_foot_z = float(standing_foot_z)

        self._h_left: float = 0.0
        self._h_right: float = 0.0
        self._left_contact: bool = False
        self._right_contact: bool = False
        self._ground_geom_name: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        static_data = ctx.accessor.get_static_data()
        if 'ground_geom_name' not in static_data:
            raise KeyError(
                f"on_pre_episode: 'ground_geom_name' not in static_data "
                f"(available={list(static_data.keys())})"
            )
        self._ground_geom_name = static_data['ground_geom_name']

        self._h_left = 0.0
        self._h_right = 0.0
        self._left_contact = False
        self._right_contact = False

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        derived_state = ctx.accessor.get_derived_state([self.agent_id, 'contacts'])
        robot_state = derived_state.get(self.agent_id)
        if robot_state is None:
            raise KeyError(
                f"on_post_action_step: '{self.agent_id}' not in derived_state "
                f"(available={list(derived_state.keys())})"
            )

        body_xipos = robot_state.get('body_xipos')
        if body_xipos is None:
            raise KeyError(
                f"on_post_action_step: 'body_xipos' not in "
                f"derived_state['{self.agent_id}'] "
                f"(available={list(robot_state.keys())})"
            )

        suffix = self.agent_id[-1]  # 'a' or 'b'
        left_key = f"foot_left_{suffix}"
        right_key = f"foot_right_{suffix}"

        if left_key not in body_xipos:
            raise KeyError(
                f"on_post_action_step: '{left_key}' not in body_xipos "
                f"(available={list(body_xipos.keys())})"
            )
        if right_key not in body_xipos:
            raise KeyError(
                f"on_post_action_step: '{right_key}' not in body_xipos "
                f"(available={list(body_xipos.keys())})"
            )

        left_z = float(body_xipos[left_key][2])
        right_z = float(body_xipos[right_key][2])

        self._h_left = left_z - self.standing_foot_z
        self._h_right = right_z - self.standing_foot_z

        # --- Contact detection ---
        cv = derived_state.get('contacts')
        self._left_contact, self._right_contact = self._detect_contact(
            cv, ctx, suffix,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_output(self) -> Dict[str, Any]:
        return {
            "h_left_foot": self._h_left,
            "h_right_foot": self._h_right,
            "left_foot_contact": self._left_contact,
            "right_foot_contact": self._right_contact,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "standing_foot_z": self.standing_foot_z,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "FootStateObserver":
        return cls(**config)

    # ------------------------------------------------------------------
    # Contact detection
    # ------------------------------------------------------------------

    def _detect_contact(
        self, cv: Any, ctx: ReadOnlySimContext, suffix: str,
    ) -> Tuple[bool, bool]:
        """Detect ground contact for both feet.

        Same logic as CrossSupportBalanceRewarder._get_foot_contact_state:
        any contact entry pairing a foot body with the ground geom counts
        as contact. No force threshold.
        """
        left_foot_body = f"foot_left_{suffix}"
        right_foot_body = f"foot_right_{suffix}"
        ground_geom = self._ground_geom_name

        left_contact = False
        right_contact = False

        if cv is None or cv['ncon'] <= 0:
            return left_contact, right_contact

        static_data = ctx.accessor.get_static_data()
        if 'body_id_to_name' not in static_data:
            raise KeyError(
                f"_detect_contact: 'body_id_to_name' not in static_data "
                f"(available={list(static_data.keys())})"
            )
        if 'geom_id_to_name' not in static_data:
            raise KeyError(
                f"_detect_contact: 'geom_id_to_name' not in static_data "
                f"(available={list(static_data.keys())})"
            )
        body_id_to_name = static_data['body_id_to_name']
        geom_id_to_name = static_data['geom_id_to_name']

        robot_aff = 1 if self.agent_id == 'robot_a' else 2

        aff1 = cv['aff1']
        aff2 = cv['aff2']
        geom1 = cv['geom1']
        geom2 = cv['geom2']
        body1 = cv['body1']
        body2 = cv['body2']

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
            if body_robot == left_foot_body:
                left_contact = True
            elif body_robot == right_foot_body:
                right_contact = True

        return left_contact, right_contact
