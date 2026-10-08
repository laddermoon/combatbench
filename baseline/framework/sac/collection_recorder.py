"""SAC episode recorder.

Copied and adapted from the neutral episode recorder so SAC collection can
record pre-action facts and collection provenance without importing the
PPO-oriented rollout package.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.framework.context import AGENT_IDS, ReadOnlySimContext
from envs.framework.recorder import PostActionRecorder

from .collected_episode import CollectedEpisode


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_snapshot(element) for element in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(element) for element in value)
    return value


class SACEpisodeRecorder(PostActionRecorder):
    """Buffer one episode and emit a SAC :class:`CollectedEpisode`."""

    def __init__(
        self,
        *,
        blueprint_hash: str,
        provenance: Mapping[str, Any],
        observer_names_to_keep: Optional[Sequence[str]] = None,
        snapshot_arrays: bool = True,
        require_action_extras: bool = False,
        require_pre_action_facts: bool = False,
    ) -> None:
        self._blueprint_hash = str(blueprint_hash)
        self._provenance = dict(provenance)
        self._observer_whitelist = (
            list(observer_names_to_keep)
            if observer_names_to_keep is not None
            else None
        )
        self._snapshot_arrays = bool(snapshot_arrays)
        self._require_action_extras = bool(require_action_extras)
        self._require_pre_action_facts = bool(require_pre_action_facts)

        self._frames: List[Dict[str, Any]] = []
        self._pending_pre_action_facts: Dict[str, Dict[str, Any]] = {}
        self._episode_index = -1
        self._base_seed: Optional[int] = None
        self._episode_options: Dict[str, Any] = {}
        self._agent_termination_proposal_records: Dict[str, List[Tuple[str, int]]] = {
            agent_id: [] for agent_id in AGENT_IDS
        }
        self._seen_reasons: Dict[str, set] = {agent_id: set() for agent_id in AGENT_IDS}
        self._final_observation: Dict[str, np.ndarray] = {}
        self._last_episode: Optional[CollectedEpisode] = None

    def set_provenance(self, provenance: Mapping[str, Any]) -> None:
        self._provenance = dict(provenance)

    def get_last_episode(self) -> CollectedEpisode:
        if self._last_episode is None:
            raise RuntimeError(
                "No completed SAC episode available; call this only after "
                "EnvRuntime has finished at least one episode."
            )
        return self._last_episode

    def has_completed_episode(self) -> bool:
        return self._last_episode is not None

    def set_pending_pre_action_facts(
        self,
        facts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Attach facts captured immediately before the next runtime.step()."""
        if self._pending_pre_action_facts:
            raise RuntimeError(
                "pre-action facts were not consumed by a completed step"
            )
        self._pending_pre_action_facts = {
            str(agent_id): dict(values) for agent_id, values in facts.items()
        }

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._frames = []
        self._pending_pre_action_facts = {}
        self._episode_index += 1
        self._base_seed = ctx.base_seed
        self._episode_options = dict(ctx.episode_options)
        self._agent_termination_proposal_records = {aid: [] for aid in AGENT_IDS}
        self._seen_reasons = {aid: set() for aid in AGENT_IDS}
        self._final_observation = {}

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        snap = _snapshot if self._snapshot_arrays else (lambda v: v)
        outputs = dict(observer_outputs)
        if self._observer_whitelist is not None:
            outputs = {
                key: outputs[key]
                for key in self._observer_whitelist
                if key in outputs
            }

        pre_action_facts = self._pending_pre_action_facts
        self._pending_pre_action_facts = {}
        if self._require_pre_action_facts and not pre_action_facts:
            raise RuntimeError("required pre-action facts missing for frame")

        self._frames.append(
            {
                "episode_step": int(ctx.episode_step),
                "physics_step": int(ctx.physics_step),
                "observation": snap(dict(observation)),
                "action": snap(dict(action)),
                "observer_outputs": snap(outputs),
                "action_extras": (
                    snap({agent: extras for agent, extras in action_extras.items()})
                    if action_extras is not None
                    else None
                ),
                "pre_action_facts": snap(pre_action_facts),
            }
        )
        for agent_id in AGENT_IDS:
            for reason in ctx.agent_termination_proposals.get(agent_id, ()):
                if reason not in self._seen_reasons[agent_id]:
                    self._seen_reasons[agent_id].add(reason)
                    self._agent_termination_proposal_records[agent_id].append(
                        (reason, int(ctx.episode_step))
                    )

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        final_obs = ctx.accessor.get_observation()
        self._final_observation = {
            str(agent): np.asarray(value) for agent, value in final_obs.items()
        }

        for agent_id in AGENT_IDS:
            if not self._agent_termination_proposal_records[agent_id]:
                raise RuntimeError(
                    f"SACEpisodeRecorder.on_post_episode: agent {agent_id!r} "
                    "has no termination proposal records"
                )

        if self._base_seed is None:
            raise RuntimeError(
                "SACEpisodeRecorder.on_post_episode: ctx.base_seed was None"
            )

        frozen_records: Dict[str, Tuple[Tuple[str, int], ...]] = {
            agent_id: tuple(self._agent_termination_proposal_records[agent_id])
            for agent_id in AGENT_IDS
        }
        self._last_episode = CollectedEpisode.from_buffer_frames(
            frames=self._frames,
            final_observation=self._final_observation,
            base_seed=int(self._base_seed),
            episode_index=int(self._episode_index),
            blueprint_hash=self._blueprint_hash,
            agent_termination_proposal_records=frozen_records,
            episode_options=self._episode_options,
            observer_names_to_keep=self._observer_whitelist,
            episode_metrics=_snapshot(dict(ctx.metrics)),
            provenance=self._provenance,
            require_action_extras=self._require_action_extras,
            require_pre_action_facts=self._require_pre_action_facts,
        )


__all__ = ["SACEpisodeRecorder"]
