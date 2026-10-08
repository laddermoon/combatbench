"""Observer plugin that reports non-foot robot body contacts with arena walls.

Outputs per step:
  - wall_contact: whether any non-foot body geom of the agent touches a
    wall geom (env geom whose name ends in ``wall``), at or above
    ``force_threshold`` newtons
  - wall_body: name of the robot body in wall contact ('' when none)

Motivation: the balance curriculum's random torso pushes can be gamed
by leaning the back against an arena wall — the wall absorbs backward
forces and anchors the backward lean that counters forward pulls,
letting the policy score recovery without learning dynamic balance.
This observer exposes the exploit per frame so the experiment can
penalize it, and the same detection marks wall-leans as falls in
``StandingTriggeredForcePlugin``.

Contact machinery mirrors ``FootStateObserver._detect_contact`` /
``StandingTriggeredForcePlugin._is_non_foot_grounded``.
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


_FOOT_BODY_NAMES = ("foot_left", "foot_right")


def sustained_wall_mask(contact, window: int = 40, count_thresh: int = 10):
    """Per-frame mask of sustained wall reliance (touch-and-go safe).

    The exploit evolved past consecutive-run detection: the robot taps
    the wall and bounces off, repeatedly — each contact stays under the
    run-length grace but the *pattern* still harvests wall support.
    A legitimate recovery brace uses the wall once or twice per
    incident; reliance shows up as contact DENSITY.  Frames are
    penalized (contact or not) while the trailing ``window``-frame
    contact count exceeds ``count_thresh`` — a single <10-frame brace
    in a 40-frame window stays free, camping or tap-cycling pays.

    ``contact``: bool array (T,) → bool array (T,), True on penalized
    frames.
    """
    import numpy as np
    c = np.asarray(contact, dtype=bool)
    pref = np.concatenate([[0], np.cumsum(c.astype(np.int64))])
    idx = np.arange(len(c)) + 1
    cnt = pref[idx] - pref[np.maximum(idx - window, 0)]
    return cnt > count_thresh


class WallContactObserver(BaseObserverPlugin):
    """Per-agent wall-contact detector."""

    def __init__(self, agent_id: str = "robot_a",
                 force_threshold: float = 1.0):
        self.agent_id = str(agent_id)
        self.force_threshold = float(force_threshold)
        self._wall_contact: bool = False
        self._wall_body: str = ""

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._wall_contact = False
        self._wall_body = ""

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')
        self._wall_contact = False
        self._wall_body = ""
        if cv is None or cv['ncon'] <= 0:
            return

        static_data = ctx.accessor.get_static_data()
        body_id_to_name = static_data.get('body_id_to_name', {})
        geom_id_to_name = static_data.get('geom_id_to_name', {})
        robot_aff = 1 if self.agent_id == 'robot_a' else 2

        aff1, aff2 = cv['aff1'], cv['aff2']
        geom1, geom2 = cv['geom1'], cv['geom2']
        body1, body2 = cv['body1'], cv['body2']
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
            if not geom_env.startswith('wall'):
                continue
            if float(force_mag[i]) < self.force_threshold:
                continue
            if any(foot in body_robot for foot in _FOOT_BODY_NAMES):
                continue
            self._wall_contact = True
            self._wall_body = body_robot
            return

    def get_output(self) -> Dict[str, Any]:
        return {
            "wall_contact": self._wall_contact,
            "wall_body": self._wall_body,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "force_threshold": self.force_threshold,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "WallContactObserver":
        return cls(**config)
