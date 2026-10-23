"""Record a CoreClip from a live Humanoid21 simulator / EnvRuntime.

Per action step, store the current CoreState of one agent plus the
action that was issued at that step (causal convention:
``ref_action[k]`` was applied at state k and aims to produce the
state observed at k+1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from envs.framework.context import ReadOnlySimContext
from envs.framework.recorder import PostActionRecorder

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
        from .core_clip import record_lineage
        meta = record_lineage(meta, "record", parent=self.agent_id)
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


class CoreClipPostRecorder(PostActionRecorder):
    """``PostActionRecorder`` adapter: one CoreClip npz per agent on
    episode end.

    Temporal pairing follows the causal CoreClip convention: each call
    sees the post-step state ``s_{t+1}`` and the action ``a_t`` that was
    applied at ``s_t``, so the recorder stores ``(s_t, a_t)`` pairs —
    the pre-episode state seeds ``s_0``, the terminal post-state is
    implicit in ``a_{T-1}``.

    CLI usage (``envs.framework.round_runner --recorder``)::

        mocap.humanoid21.clip.recorder:CoreClipPostRecorder?\
            output_dir=<dir>&clip_id=<id>&agents=robot_a,robot_b
    """

    def __init__(self, output_dir: str,
                 clip_id: str = "rec",
                 agents: str = "robot_a,robot_b",
                 meta_extra: Optional[Mapping[str, Any]] = None) -> None:
        self.output_dir = Path(output_dir)
        self.clip_id = str(clip_id)
        self.agents = tuple(s for s in str(agents).split(",") if s)
        self.meta_extra = dict(meta_extra or {})
        self._recs: Dict[str, CoreClipRecorder] = {}
        self._prev: Dict[str, Dict[str, np.ndarray]] = {}

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        cs = ctx.accessor.get_core_state()
        seed = ctx.base_seed
        for rid in self.agents:
            self._prev[rid] = {
                f: np.asarray(cs[rid][f]).copy() for f in CORE_FIELDS}
            extra = dict(self.meta_extra)
            extra.update({"episode_seed": seed})
            self._recs[rid] = CoreClipRecorder(
                agent_id=rid,
                clip_id=f"{self.clip_id}_{seed}_{rid}",
                meta_extra=extra)

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        cs = ctx.accessor.get_core_state()
        for rid in self.agents:
            a = action.get(rid)
            self._recs[rid].record(
                self._prev[rid],
                None if a is None else np.asarray(a, dtype=np.float32))
            self._prev[rid] = {
                f: np.asarray(cs[rid][f]).copy() for f in CORE_FIELDS}

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed = ctx.base_seed
        for rid, rec in self._recs.items():
            cc = rec.finish(meta_extra={
                "termination_reasons":
                    ctx.agent_termination_proposals.get(rid, [])})
            cc.save(self.output_dir / f"{cc.clip_id}.npz")
        self._recs = {}
        self._prev = {}
