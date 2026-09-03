"""``EpisodeRecorder``: post-action recorder producing one :class:`Episode`.

See ``baseline/common/rollout/DESIGN.md`` §3 for scope and rationale.
Mirrors :class:`envs.framework.recorder.EpisodeBufferRecorder` for the
per-step buffering, but adds:

* Captures ``final_observation`` (``obs_{T+1}``) on ``on_post_episode``
  by reading ``ctx.accessor.get_observation()`` so RL bootstrap targets
  are persisted.
* Returns a strongly-typed :class:`Episode` via :meth:`get_last_episode`.
* Carries the owning blueprint's hash so the produced episodes can be
  validated against an :class:`EpisodeCollection`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.framework.context import AGENT_IDS, ReadOnlySimContext
from envs.framework.recorder import PostActionRecorder

from .episode import Episode


def _snapshot(value: Any) -> Any:
    """Allocation-light deep-ish copy preserving ndarrays.

    Same semantics as :func:`envs.framework.recorder._snapshot`, copied
    locally to avoid importing a private helper.
    """
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_snapshot(element) for element in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(element) for element in value)
    return value


class EpisodeRecorder(PostActionRecorder):
    """Buffer one episode and emit a :class:`Episode` after it ends.

    Parameters
    ----------
    blueprint_hash:
        Stable hash of the :class:`EnvBlueprint` driving this episode.
        Embedded into the produced :class:`Episode` so collections can
        verify uniformity.
    observer_names_to_keep:
        Optional whitelist; only these top-level keys of
        ``observer_outputs`` are buffered. ``None`` keeps all.
    snapshot_arrays:
        If ``True`` (default), copy ndarrays per frame so subsequent
        observer mutations cannot retroactively change buffered data.
    """

    def __init__(
        self,
        blueprint_hash: str,
        observer_names_to_keep: Optional[Sequence[str]] = None,
        snapshot_arrays: bool = True,
    ) -> None:
        self._blueprint_hash = str(blueprint_hash)
        self._observer_whitelist: Optional[List[str]] = (
            list(observer_names_to_keep) if observer_names_to_keep is not None else None
        )
        self._snapshot_arrays = bool(snapshot_arrays)

        # Per-episode buffers (cleared on every on_pre_episode).
        self._frames: List[Dict[str, Any]] = []
        self._episode_index: int = -1
        self._base_seed: Optional[int] = None
        self._episode_options: Dict[str, Any] = {}
        self._agent_termination_proposal_records: Dict[str, List[Tuple[str, int]]] = {
            aid: [] for aid in AGENT_IDS
        }
        self._seen_reasons: Dict[str, set] = {aid: set() for aid in AGENT_IDS}
        self._final_observation: Dict[str, np.ndarray] = {}
        self._last_episode: Optional[Episode] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_last_episode(self) -> Episode:
        """Return the most recently completed episode.

        Raises ``RuntimeError`` if called before any episode has finished.
        """
        if self._last_episode is None:
            raise RuntimeError(
                "No completed episode available; call this only after "
                "EnvRuntime has finished at least one episode."
            )
        return self._last_episode

    def has_completed_episode(self) -> bool:
        return self._last_episode is not None

    # ------------------------------------------------------------------
    # Recorder hooks
    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._frames = []
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
        outputs: Dict[str, Any] = dict(observer_outputs)
        if self._observer_whitelist is not None:
            outputs = {k: outputs[k] for k in self._observer_whitelist if k in outputs}

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
            }
        )
        # Track per-agent termination proposals: for each agent, detect
        # new reasons not yet seen and record (reason, episode_step).
        # Same reason is only recorded once (first occurrence).
        for aid in AGENT_IDS:
            for reason in ctx.agent_termination_proposals.get(aid, ()):
                if reason not in self._seen_reasons[aid]:
                    self._seen_reasons[aid].add(reason)
                    self._agent_termination_proposal_records[aid].append(
                        (reason, int(ctx.episode_step))
                    )

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        # Capture obs_{T+1} for RL bootstrap. Defensive: if the accessor
        # cannot produce an observation here we record an empty mapping
        # so trainers can detect & skip rather than crash deep inside
        # episode assembly.
        try:
            final_obs = ctx.accessor.get_observation()
            self._final_observation = {
                str(agent): np.asarray(value) for agent, value in final_obs.items()
            }
        except Exception:
            self._final_observation = {}

        # Validate: every agent must have at least one termination proposal
        # (episode ends only when all_agents_terminated is True).
        for aid in AGENT_IDS:
            if not self._agent_termination_proposal_records[aid]:
                raise RuntimeError(
                    f"EpisodeRecorder.on_post_episode: agent {aid!r} has no "
                    f"termination proposal records — episode ended without "
                    f"all agents terminated, indicating a bug in termination logic"
                )

        if self._base_seed is None:
            raise RuntimeError(
                "EpisodeRecorder.on_post_episode: ctx.base_seed was None at "
                "episode start. EpisodeRunner is supposed to resolve the "
                "base seed before reset; this recorder needs it for the "
                "produced Episode."
            )

        # Freeze records into tuples.
        frozen_records: Dict[str, Tuple[Tuple[str, int], ...]] = {
            aid: tuple(self._agent_termination_proposal_records[aid])
            for aid in AGENT_IDS
        }

        self._last_episode = Episode.from_buffer_frames(
            frames=self._frames,
            final_observation=self._final_observation,
            base_seed=int(self._base_seed),
            episode_index=int(self._episode_index),
            blueprint_hash=self._blueprint_hash,
            agent_termination_proposal_records=frozen_records,
            episode_options=self._episode_options,
            observer_names_to_keep=self._observer_whitelist,
            episode_metrics=_snapshot(dict(ctx.metrics)),
        )


__all__ = ["EpisodeRecorder"]
