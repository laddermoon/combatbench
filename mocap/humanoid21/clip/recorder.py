"""Record a CoreClip from a live Humanoid21 simulator / EnvRuntime.

Per action step, store the current CoreState of one agent plus the
action that was issued at that step (causal convention:
``ref_action[k]`` was applied at state k and aims to produce the
state observed at k+1).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .core_clip import CORE_FIELDS, CoreClip, SPEC_VERSION


class CoreClipRecorder:
    """Append-only recorder.  Call ``record`` once per action step."""

    def __init__(self, agent_id: str = "robot_a",
                 clip_id: str = "rec",
                 meta_extra: Optional[Dict[str, Any]] = None) -> None:
        self.agent_id = str(agent_id)
        self.clip_id = str(clip_id)
        self._meta_extra = dict(meta_extra or {})
        self._states: List[Dict[str, np.ndarray]] = []
        self._actions: List[np.ndarray] = []

    def record(self, core_state: Dict[str, np.ndarray],
               action: Optional[np.ndarray] = None) -> None:
        """Store the agent's CoreState (post-step) and, on the first
        call pattern the caller chooses, the action issued that step.

        ``core_state`` is ``sim.get_core_state()[agent_id]`` (already
        robot-scoped).  ``action`` is the normalized (21,) action the
        policy emitted for that step; pass ``None`` when not tracked.
        """
        self._states.append({f: np.asarray(core_state[f]).copy()
                             for f in CORE_FIELDS})
        if action is not None:
            a = np.asarray(action, dtype=np.float32)
            if a.shape != (21,):
                raise ValueError(f"action shape {a.shape}, expected (21,)")
            self._actions.append(a)

    def record_sim(self, sim, action: Optional[np.ndarray] = None) -> None:
        """Convenience wrapper reading from a Humanoid21Simulator."""
        self.record(sim.get_core_state()[self.agent_id], action)

    def finish(self, with_actions: bool = True,
               meta_extra: Optional[Dict[str, Any]] = None) -> CoreClip:
        meta: Dict[str, Any] = {
            "spec_version": SPEC_VERSION,
            "clip_id": self.clip_id,
            "robot": "humanoid21",
            "joint_space": "humanoid21",
            "source": {"type": "policy_rollout",
                       "agent_id": self.agent_id},
        }
        meta.update(self._meta_extra)
        if meta_extra:
            meta.update(meta_extra)
        ra = None
        if with_actions and self._actions:
            if len(self._actions) != len(self._states):
                raise ValueError(
                    f"{len(self._actions)} actions for "
                    f"{len(self._states)} states")
            ra = np.stack(self._actions)
            meta["ref_action_semantics"] = "pd_target_norm"
        return CoreClip.from_states(self._states, meta, ra)

    def __len__(self) -> int:
        return len(self._states)
