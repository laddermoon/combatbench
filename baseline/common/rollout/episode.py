"""Strongly-typed single-episode data object.

See ``baseline/common/rollout/DESIGN.md`` §2.1 for the full schema and
the rationale for each field. This module is intentionally
framework-light: it depends only on :class:`EnvBlueprint` for hashing
and on numpy. It does NOT import any simulator / runtime code so it
can be used anywhere (training, eval, offline analysis).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.framework.blueprint import EnvBlueprint


# ---------------------------------------------------------------------------
# Blueprint hash
# ---------------------------------------------------------------------------
def blueprint_hash(blueprint: EnvBlueprint) -> str:
    """Stable content hash of an :class:`EnvBlueprint`.

    Uses SHA-256 over a sorted JSON serialization. Two blueprints with
    the same plugins / observers / simulator config produce the same
    hash regardless of dict ordering.
    """
    payload = json.dumps(blueprint.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _flatten_keys(node: Any, prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Any]]:
    """Walk a nested dict and yield (path, leaf) pairs.

    A "leaf" is anything that is not a plain ``dict``. Lists / tuples /
    ndarrays are leaves (we do not flatten through them).
    """
    if isinstance(node, dict):
        out: List[Tuple[Tuple[str, ...], Any]] = []
        for key, value in node.items():
            out.extend(_flatten_keys(value, prefix + (str(key),)))
        return out
    return [(prefix, node)]


def _try_stack(values: List[Any]) -> Any:
    """Stack ``values`` along a new axis 0 if every entry is an ndarray
    of identical shape & dtype; otherwise return a plain ``list``.

    Implements the §2.1.1 stacking policy.
    """
    if not values:
        return np.empty((0,))
    first = values[0]
    if not isinstance(first, np.ndarray):
        return list(values)
    ref_shape = first.shape
    ref_dtype = first.dtype
    for value in values[1:]:
        if not isinstance(value, np.ndarray):
            return list(values)
        if value.shape != ref_shape or value.dtype != ref_dtype:
            return list(values)
    return np.stack(values, axis=0)


def _stack_observer_outputs(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Stack per-frame ``observer_outputs`` into a dict of stacked leaves.

    Output structure mirrors the input dict tree, with leaf values
    replaced by either ``np.ndarray`` (T, *) or ``list[T]``.

    Empty input yields ``{}``.
    """
    if not frames:
        return {}
    # Collect the union of paths across all frames. Paths that are
    # missing in some frames are filled with ``None`` for that frame
    # and will fall back to a plain list (because list[ndarray|None]
    # is not a stackable ndarray sequence).
    all_paths: Dict[Tuple[str, ...], List[Any]] = {}
    num_frames = len(frames)
    for index, frame in enumerate(frames):
        for path, leaf in _flatten_keys(dict(frame)):
            if path not in all_paths:
                all_paths[path] = [None] * num_frames
            all_paths[path][index] = leaf

    # Reassemble back into nested dict, stacking each leaf list.
    out: Dict[str, Any] = {}
    for path, values in all_paths.items():
        cursor = out
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = _try_stack(values)
    return out


def _stack_agent_field(
    frames: Sequence[Mapping[str, Any]],
    field_name: str,
) -> Dict[str, np.ndarray]:
    """Stack ``frame[field_name][agent_id]`` across frames per agent.

    Returns ``{agent_id: (T, *) ndarray}``. ``field_name`` is e.g.
    ``"observation"`` or ``"action"``. Frames missing the field
    contribute ``None`` and force the agent to be dropped (raises).

    Behaviour for varying agent sets:

    * If an agent appears in some frames but not others: ``ValueError``.
    * If an agent's array shape / dtype varies across frames:
      ``ValueError``.
    """
    if not frames:
        return {}
    # Discover the agent set from the first non-None frame.
    agent_ids: Optional[Sequence[str]] = None
    for frame in frames:
        value = frame.get(field_name)
        if value is None:
            continue
        agent_ids = list(value.keys())
        break
    if agent_ids is None:
        return {}

    out: Dict[str, np.ndarray] = {}
    for agent_id in agent_ids:
        per_frame: List[np.ndarray] = []
        for frame in frames:
            value = frame.get(field_name)
            if value is None or agent_id not in value:
                raise ValueError(
                    f"frame is missing {field_name}[{agent_id!r}] — cannot "
                    f"stack episode (frames must have a consistent agent set)"
                )
            arr = np.asarray(value[agent_id])
            per_frame.append(arr)
        try:
            out[agent_id] = np.stack(per_frame, axis=0)
        except ValueError as exc:
            raise ValueError(
                f"cannot stack {field_name}[{agent_id!r}]: per-frame shape / "
                f"dtype mismatch"
            ) from exc
    return out


def _stack_action_extras(
    frames: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Stack ``action_extras`` into ``{agent_id: {key: (T,*) ndarray}}``.

    Per DESIGN §2.1: if any frame has ``action_extras=None`` or that
    agent's extras is ``None`` we drop the agent silently (i.e. omit
    from the output). Trainers must check membership rather than rely
    on NaN sentinels.
    """
    if not frames:
        return {}
    # Determine which agents have extras in EVERY frame.
    candidate_agents: Optional[set[str]] = None
    for frame in frames:
        ae = frame.get("action_extras")
        if not ae:
            return {}  # any frame without extras at all → drop everything
        present = {a for a, v in ae.items() if v is not None}
        if candidate_agents is None:
            candidate_agents = present
        else:
            candidate_agents &= present
    if not candidate_agents:
        return {}

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for agent_id in candidate_agents:
        # Discover the per-key shape from the first frame.
        first_extras = frames[0]["action_extras"][agent_id]
        if not isinstance(first_extras, Mapping):
            continue
        keys = list(first_extras.keys())
        per_key: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
        for frame in frames:
            extras = frame["action_extras"][agent_id]
            for key in keys:
                if key not in extras:
                    raise ValueError(
                        f"action_extras[{agent_id!r}][{key!r}] missing in "
                        f"some frame; extras schema must be stable per agent"
                    )
                value = extras[key]
                per_key[key].append(np.asarray(value))
        stacked: Dict[str, np.ndarray] = {}
        for key, values in per_key.items():
            try:
                stacked[key] = np.stack(values, axis=0)
            except ValueError as exc:
                raise ValueError(
                    f"cannot stack action_extras[{agent_id!r}][{key!r}]: "
                    f"per-frame shape / dtype mismatch"
                ) from exc
        out[agent_id] = stacked
    return out


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    """One episode of trajectory data, fully stacked.

    See ``baseline/common/rollout/DESIGN.md`` §2.1 for the schema rules.

    Convention: ``T = num_frames`` is the number of action steps. The
    ``final_observation`` field carries ``obs_{T+1}`` (the post-final-action
    observation) so RL trainers can bootstrap.
    """

    base_seed: int
    episode_index: int
    blueprint_hash: str
    num_frames: int
    episode_options: Mapping[str, Any]

    agent_termination_proposal_records: Mapping[str, Tuple[Tuple[str, int], ...]]
    """Per-agent termination proposal history.

    Maps ``agent_id`` → tuple of ``(reason, episode_step)`` pairs.
    ``reason`` is the termination reason string (e.g. ``"ko"``,
    ``"timeout"``, ``"imbalance_robot_a"``).
    ``episode_step`` is the value of ``ctx.episode_step`` when the
    proposal was **first proposed** — i.e. the action-step number
    (starting from 1) at which ``on_post_action_step`` detected the
    new reason. The same reason is only recorded once (first
    occurrence).

    Data slicing usage::

        # EpisodeRecorder.on_post_episode guarantees records are non-empty.
        records = episode.agent_termination_proposal_records["robot_a"]
        first_reason, term_step = records[0]
        obs_a = episode.observations["robot_a"][:term_step]
        act_a = episode.actions["robot_a"][:term_step]

        # True termination (non-timeout → bootstrap=0):
        is_true_terminated = first_reason != "timeout"
    """

    observations: Mapping[str, np.ndarray]
    actions: Mapping[str, np.ndarray]
    action_extras: Mapping[str, Mapping[str, np.ndarray]]
    observer_outputs: Mapping[str, Any]

    final_observation: Mapping[str, np.ndarray]

    episode_metrics: Mapping[str, Any] = field(default_factory=dict)
    """Per-episode metrics snapshot from ``ctx.metrics`` at episode end.

    Captured by ``EpisodeRecorder.on_post_episode``.  Contains any
    metrics written by world plugins (e.g. push_count, fall_count).
    """

    @property
    def agent_termination_reason(self) -> Mapping[str, str]:
        """Per-agent termination reason (first record for each agent).

        Returns ``agent_id`` → single reason string.
        Empty string if that agent has no termination records.
        """
        return {
            aid: records[0][0] if records else ""
            for aid, records in self.agent_termination_proposal_records.items()
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_buffer_frames(
        cls,
        *,
        frames: Sequence[Mapping[str, Any]],
        final_observation: Mapping[str, np.ndarray],
        base_seed: int,
        episode_index: int,
        blueprint_hash: str,
        agent_termination_proposal_records: Mapping[str, Sequence[Tuple[str, int]]],
        episode_options: Optional[Mapping[str, Any]] = None,
        observer_names_to_keep: Optional[Sequence[str]] = None,
        episode_metrics: Optional[Mapping[str, Any]] = None,
    ) -> "Episode":
        """Build an :class:`Episode` from raw recorder frames.

        ``frames`` follows the :class:`EpisodeBufferRecorder` shape
        (one dict per ``on_post_action_step`` call): each must have
        ``observation``, ``action``, ``observer_outputs``, optionally
        ``action_extras``.

        ``agent_termination_proposal_records``: per-agent tuple of
        ``(reason, episode_step)`` pairs. See the field docstring for
        semantics.

        ``observer_names_to_keep``: optional whitelist on the top-level
        ``observer_outputs`` keys. ``None`` keeps all.
        """
        observations = _stack_agent_field(frames, "observation")
        actions = _stack_agent_field(frames, "action")
        extras = _stack_action_extras(frames)

        observer_frames: List[Mapping[str, Any]] = []
        for frame in frames:
            outputs = dict(frame.get("observer_outputs") or {})
            if observer_names_to_keep is not None:
                outputs = {k: outputs[k] for k in observer_names_to_keep if k in outputs}
            observer_frames.append(outputs)
        observer_outputs = _stack_observer_outputs(observer_frames)

        final_obs_dict = {
            agent_id: np.asarray(value) for agent_id, value in final_observation.items()
        }

        frozen_records: Dict[str, Tuple[Tuple[str, int], ...]] = {
            aid: tuple(records) for aid, records in agent_termination_proposal_records.items()
        }

        return cls(
            base_seed=int(base_seed),
            episode_index=int(episode_index),
            blueprint_hash=str(blueprint_hash),
            num_frames=int(len(frames)),
            agent_termination_proposal_records=frozen_records,
            episode_options=dict(episode_options or {}),
            observations=observations,
            actions=actions,
            action_extras=extras,
            observer_outputs=observer_outputs,
            final_observation=final_obs_dict,
            episode_metrics=dict(episode_metrics or {}),
        )

    # ------------------------------------------------------------------
    # On-disk serialization (single-episode pair: <stem>.npz + <stem>.json)
    # ------------------------------------------------------------------
    EPISODE_FORMAT_VERSION = 2

    def save(self, stem_path: Any) -> None:
        """Serialize to ``<stem_path>.npz`` (arrays) + ``<stem_path>.json``
        (metadata + non-array observer outputs).

        ``stem_path`` may be a ``str`` or ``Path``; the suffix is stripped.
        """
        from pathlib import Path
        stem = Path(stem_path)
        if stem.suffix in (".npz", ".json"):
            stem = stem.with_suffix("")

        arrays: Dict[str, np.ndarray] = {}
        for agent_id, value in self.observations.items():
            arrays[f"obs__{agent_id}"] = np.asarray(value)
        for agent_id, value in self.actions.items():
            arrays[f"act__{agent_id}"] = np.asarray(value)
        for agent_id, extras in self.action_extras.items():
            for key, value in extras.items():
                arrays[f"extras__{agent_id}__{key}"] = np.asarray(value)
        for agent_id, value in self.final_observation.items():
            arrays[f"final_obs__{agent_id}"] = np.asarray(value)

        # Observer outputs: split into array-leaves (npz) and list-leaves (json).
        observer_arrays: Dict[str, np.ndarray] = {}
        observer_lists: Dict[str, List[Any]] = {}
        for path, leaf in _flatten_keys(dict(self.observer_outputs)):
            joined = "/".join(path)
            if isinstance(leaf, np.ndarray):
                observer_arrays[f"obs_outputs__{joined}"] = leaf
            else:
                observer_lists[joined] = list(leaf) if isinstance(leaf, list) else [leaf]
        arrays.update(observer_arrays)

        # Save arrays.
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(stem.with_suffix(".npz")), **arrays)

        # Save metadata + non-array observer leaves.
        meta = {
            "format_version": self.EPISODE_FORMAT_VERSION,
            "base_seed": int(self.base_seed),
            "episode_index": int(self.episode_index),
            "blueprint_hash": self.blueprint_hash,
            "num_frames": int(self.num_frames),
            "agent_termination_proposal_records": {
                aid: [[reason, step] for reason, step in records]
                for aid, records in self.agent_termination_proposal_records.items()
            },
            "episode_options": _to_jsonable(dict(self.episode_options)),
            "observer_outputs_lists": {
                key: _to_jsonable(value) for key, value in observer_lists.items()
            },
            "agent_ids": sorted(self.observations.keys()),
            "extras_layout": {
                agent_id: sorted(extras.keys())
                for agent_id, extras in self.action_extras.items()
            },
        }
        stem.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, stem_path: Any) -> "Episode":
        """Inverse of :meth:`save`."""
        from pathlib import Path
        stem = Path(stem_path)
        if stem.suffix in (".npz", ".json"):
            stem = stem.with_suffix("")

        meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        fmt_ver = int(meta.get("format_version", 0))
        if fmt_ver != cls.EPISODE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported episode format_version "
                f"{meta.get('format_version')}; expected "
                f"{cls.EPISODE_FORMAT_VERSION}"
            )

        with np.load(str(stem.with_suffix(".npz"))) as npz:
            keys = list(npz.keys())
            observations: Dict[str, np.ndarray] = {}
            actions: Dict[str, np.ndarray] = {}
            action_extras: Dict[str, Dict[str, np.ndarray]] = {}
            final_observation: Dict[str, np.ndarray] = {}
            observer_arrays: Dict[Tuple[str, ...], np.ndarray] = {}
            for key in keys:
                value = np.array(npz[key])
                if key.startswith("obs__"):
                    observations[key[len("obs__"):]] = value
                elif key.startswith("act__"):
                    actions[key[len("act__"):]] = value
                elif key.startswith("extras__"):
                    rest = key[len("extras__"):]
                    agent_id, _, sub = rest.partition("__")
                    action_extras.setdefault(agent_id, {})[sub] = value
                elif key.startswith("final_obs__"):
                    final_observation[key[len("final_obs__"):]] = value
                elif key.startswith("obs_outputs__"):
                    path = tuple(key[len("obs_outputs__"):].split("/"))
                    observer_arrays[path] = value

        # Rebuild observer_outputs dict tree: arrays first, then lists.
        observer_outputs: Dict[str, Any] = {}
        for path, value in observer_arrays.items():
            cursor = observer_outputs
            for sub in path[:-1]:
                cursor = cursor.setdefault(sub, {})
            cursor[path[-1]] = value
        for joined, values in (meta.get("observer_outputs_lists") or {}).items():
            path = tuple(joined.split("/"))
            cursor = observer_outputs
            for sub in path[:-1]:
                cursor = cursor.setdefault(sub, {})
            cursor[path[-1]] = list(values)

        # Load agent_termination_proposal_records.
        records_raw = meta.get("agent_termination_proposal_records", {})
        agent_termination_proposal_records: Dict[str, Tuple[Tuple[str, int], ...]] = {
            aid: tuple((r, int(s)) for r, s in pairs)
            for aid, pairs in records_raw.items()
        }

        return cls(
            base_seed=int(meta["base_seed"]),
            episode_index=int(meta["episode_index"]),
            blueprint_hash=str(meta["blueprint_hash"]),
            num_frames=int(meta["num_frames"]),
            agent_termination_proposal_records=agent_termination_proposal_records,
            episode_options=dict(meta.get("episode_options") or {}),
            observations=observations,
            actions=actions,
            action_extras=action_extras,
            observer_outputs=observer_outputs,
            final_observation=final_observation,
        )


# ---------------------------------------------------------------------------
# JSON-safe coercion (small, local; avoids cross-import with framework)
# ---------------------------------------------------------------------------
def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


__all__ = ["Episode", "blueprint_hash"]
