"""CoreStateObserver — logs per-step flattened core state for world-model fitting.

Emits ``{agent_id: (55,) float32}`` per step via ``get_output()``:
``[root_pos(3), root_rot quat wxyz(4), root_vel_local(3),
  root_angular_vel_local(3), joint_pos_norm(21), joint_vel_norm(21)]``.

Injected into an env blueprint at the dict level by dynabptt experiments::

    d = env_bp.to_dict()
    d["observer_plugins"]["dynabptt_cs"] = {
        "cls": "baseline.framework.dynabptt.corestate:CoreStateObserver",
        "config": {},
    }
    env_bp = EnvBlueprint.from_dict(d)

Frame semantics: the recorder captures this observer's post-action-step
output, so frame[i] carries s_{i+1}; pairs ``(cs[i-1], action[i]) -> cs[i]``
are the world-model transitions (i >= 1; the s_0 transition is dropped).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework.observer_plugin import BaseObserverPlugin
from envs.framework.context import ReadOnlySimContext

STATE_KEYS = (
    "root_pos",
    "root_rot",
    "root_vel_local",
    "root_angular_vel_local",
    "joint_pos_norm",
    "joint_vel_norm",
)
STATE_DIM = 55


def flatten_core_state(core_state: Dict[str, Any], agent_id: str) -> np.ndarray:
    parts = [np.asarray(core_state[agent_id][k], dtype=np.float32) for k in STATE_KEYS]
    return np.concatenate(parts)


class CoreStateObserver(BaseObserverPlugin):
    def __init__(self) -> None:
        self._out: Dict[str, np.ndarray] = {}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        cs = ctx.accessor.get_core_state()
        self._out = {aid: flatten_core_state(cs, aid) for aid in cs}

    def get_output(self) -> Dict[str, np.ndarray]:
        return self._out

    def to_blueprint(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "CoreStateObserver":
        return cls()
