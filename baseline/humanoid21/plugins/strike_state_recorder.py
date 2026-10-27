"""Strike-start state set recorder (PostActionRecorder).

Collects full core states for later RSI-style initialisation of strike
training.  Each action step, if BOTH robots' torso body z >= ``stand_h``,
the complete core_state dict of both robots is kept:

    root_pos, root_rot, root_vel_local, root_angular_vel_local,
    joint_pos_norm, joint_vel_norm

— exactly the fields ``mutator.set_core_state`` writes, so saved states
can be re-injected verbatim by a reset plugin.

Per-episode npz: per-robot arrays (S,...) of every standing frame plus
``steps`` and a ``meta_json`` provenance blob.  A merge step downstream
samples the final state set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from envs.framework.recorder import PostActionRecorder
from envs.framework.context import ReadOnlySimContext

_FIELDS = (
    "root_pos", "root_rot", "root_vel_local",
    "root_angular_vel_local", "joint_pos_norm", "joint_vel_norm",
)
_AGENTS = ("robot_a", "robot_b")


class StrikeStartStateRecorder(PostActionRecorder):
    """Stores both-robots standing core states per episode."""

    def __init__(self, output_dir: str,
                 stand_h: float = 1.10) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._stand_h = float(stand_h)
        self._ep_rows: List[Dict[str, Dict[str, np.ndarray]]] = []
        self._ep_steps: List[int] = []
        self._ep = 0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._ep_rows = []
        self._ep_steps = []
        sd = ctx.accessor.get_static_data()
        self._torso_names = {
            aid: sd.get(aid, {}).get(
                "keypoint_body_names", {}).get("torso")
            for aid in _AGENTS
        }

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        ds = ctx.accessor.get_derived_state(list(_AGENTS))
        for aid in _AGENTS:
            xpos = ds.get(aid, {}).get("body_xpos", {})
            torso = xpos.get(self._torso_names.get(aid) or "")
            if torso is None or float(np.asarray(torso)[2]) < self._stand_h:
                return
        cs = ctx.accessor.get_core_state()
        self._ep_rows.append({
            aid: {f: np.asarray(cs[aid][f]).copy() for f in _FIELDS}
            for aid in _AGENTS
        })
        self._ep_steps.append(int(ctx.episode_step))

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._ep += 1
        if not self._ep_rows:
            return
        payload: Dict[str, np.ndarray] = {
            "steps": np.asarray(self._ep_steps, dtype=np.int32),
        }
        for aid in _AGENTS:
            for f in _FIELDS:
                payload[f"{aid}/{f}"] = np.stack(
                    [r[aid][f] for r in self._ep_rows])
        meta = {
            "episode_index": self._ep,
            "base_seed": ctx.base_seed,
            "n_states": len(self._ep_rows),
            "stand_h": self._stand_h,
        }
        payload["meta_json"] = np.asarray(json.dumps(meta))
        np.savez(self._out / f"ep_{self._ep:04d}_s{ctx.base_seed}.npz",
                 **payload)
