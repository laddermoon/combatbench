"""SAC-owned collected episode schema.

This is a copy-adapted, SAC-local equivalent of the rollout episode object.
It deliberately contains no PPO sampling context, ratio, or GAE fields.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.framework.blueprint import EnvBlueprint


COLLECTION_CONTRACT_VERSION = "sac_collection_v1"


def blueprint_hash(blueprint: EnvBlueprint) -> str:
    payload = json.dumps(
        blueprint.to_dict(), sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mapping_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten_keys(
    node: Any,
    prefix: Tuple[str, ...] = (),
) -> List[Tuple[Tuple[str, ...], Any]]:
    if isinstance(node, dict):
        out: List[Tuple[Tuple[str, ...], Any]] = []
        for key, value in node.items():
            out.extend(_flatten_keys(value, prefix + (str(key),)))
        return out
    return [(prefix, node)]


def _try_stack(values: List[Any]) -> Any:
    if not values:
        return np.empty((0,))
    first = values[0]
    if not isinstance(first, np.ndarray):
        return list(values)
    ref_shape = first.shape
    ref_dtype = first.dtype
    for value in values[1:]:
        if (
            not isinstance(value, np.ndarray)
            or value.shape != ref_shape
            or value.dtype != ref_dtype
        ):
            return list(values)
    return np.stack(values, axis=0)


def _stack_mapping_leaves(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not frames:
        return {}
    all_paths: Dict[Tuple[str, ...], List[Any]] = {}
    num_frames = len(frames)
    for index, frame in enumerate(frames):
        for path, leaf in _flatten_keys(dict(frame)):
            if path not in all_paths:
                all_paths[path] = [None] * num_frames
            all_paths[path][index] = leaf

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
    if not frames:
        return {}
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
                    "stack collected episode"
                )
            per_frame.append(np.asarray(value[agent_id]))
        try:
            out[agent_id] = np.stack(per_frame, axis=0)
        except ValueError as exc:
            raise ValueError(
                f"cannot stack {field_name}[{agent_id!r}]: per-frame shape / "
                "dtype mismatch"
            ) from exc
    return out


def _stack_nested_agent_field(
    frames: Sequence[Mapping[str, Any]],
    field_name: str,
    *,
    required: bool,
) -> Dict[str, Dict[str, Any]]:
    if not frames:
        return {}
    candidate_agents: Optional[set[str]] = None
    for frame in frames:
        value = frame.get(field_name)
        if value is None:
            if required:
                raise ValueError(f"frame is missing required {field_name}")
            return {}
        present = {agent for agent, fields in value.items() if fields is not None}
        if candidate_agents is None:
            candidate_agents = present
        else:
            candidate_agents &= present
    if not candidate_agents:
        if required:
            raise ValueError(f"no agents carry required {field_name}")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for agent_id in sorted(candidate_agents):
        per_agent_frames = []
        for frame in frames:
            value = frame[field_name]
            if agent_id not in value or value[agent_id] is None:
                raise ValueError(
                    f"{field_name}[{agent_id!r}] missing in some frame"
                )
            per_agent_frames.append(dict(value[agent_id]))
        out[agent_id] = _stack_mapping_leaves(per_agent_frames)
    return out


@dataclass(frozen=True)
class CollectedEpisode:
    """One SAC collection episode plus immutable collection provenance."""

    base_seed: int
    episode_index: int
    blueprint_hash: str
    num_frames: int
    episode_options: Mapping[str, Any]

    agent_termination_proposal_records: Mapping[str, Tuple[Tuple[str, int], ...]]
    observations: Mapping[str, np.ndarray]
    actions: Mapping[str, np.ndarray]
    action_extras: Mapping[str, Mapping[str, Any]]
    observer_outputs: Mapping[str, Any]
    pre_action_facts: Mapping[str, Mapping[str, Any]]
    final_observation: Mapping[str, np.ndarray]
    episode_metrics: Mapping[str, Any] = field(default_factory=dict)
    physics_steps: Optional[np.ndarray] = None

    run_id: str = ""
    collection_round: int = 0
    job_index: int = 0
    episode_seed: int = 0
    job_key: str = ""
    policy_fingerprints: Mapping[str, str] = field(default_factory=dict)
    behavior: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    worker_id: int = 0
    wall_time_s: float = 0.0
    contract_version: str = COLLECTION_CONTRACT_VERSION

    @property
    def agent_frame_boundary(self) -> Dict[str, int]:
        n = int(self.num_frames)
        last_physical = n
        if self.physics_steps is not None and n:
            ps = np.asarray(self.physics_steps, dtype=np.int64)
            deltas = np.diff(ps, prepend=np.int64(0))
            physical = np.nonzero(deltas > 0)[0]
            last_physical = int(physical[-1]) + 1 if physical.size else 0
        boundary: Dict[str, int] = {}
        agent_ids = set(self.observations) | set(self.agent_termination_proposal_records)
        for agent_id in agent_ids:
            records = self.agent_termination_proposal_records.get(agent_id, ())
            first = records[0][1] if records else n
            boundary[agent_id] = int(min(first, last_physical))
        return boundary

    @property
    def agent_termination_reason(self) -> Mapping[str, str]:
        return {
            agent_id: records[0][0] if records else ""
            for agent_id, records in self.agent_termination_proposal_records.items()
        }

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
        provenance: Optional[Mapping[str, Any]] = None,
        require_action_extras: bool = False,
        require_pre_action_facts: bool = False,
    ) -> "CollectedEpisode":
        observations = _stack_agent_field(frames, "observation")
        actions = _stack_agent_field(frames, "action")
        action_extras = _stack_nested_agent_field(
            frames, "action_extras", required=require_action_extras,
        )
        pre_action_facts = _stack_nested_agent_field(
            frames, "pre_action_facts", required=require_pre_action_facts,
        )
        physics_steps = (
            np.asarray([int(f["physics_step"]) for f in frames], dtype=np.int64)
            if frames and all("physics_step" in f for f in frames)
            else None
        )

        observer_frames: List[Mapping[str, Any]] = []
        for frame in frames:
            outputs = dict(frame.get("observer_outputs") or {})
            if observer_names_to_keep is not None:
                outputs = {
                    key: outputs[key]
                    for key in observer_names_to_keep
                    if key in outputs
                }
            observer_frames.append(outputs)
        observer_outputs = _stack_mapping_leaves(observer_frames)

        final_obs_dict = {
            agent_id: np.asarray(value)
            for agent_id, value in final_observation.items()
        }
        frozen_records = {
            agent_id: tuple(records)
            for agent_id, records in agent_termination_proposal_records.items()
        }
        prov = dict(provenance or {})

        return cls(
            base_seed=int(base_seed),
            episode_index=int(episode_index),
            blueprint_hash=str(blueprint_hash),
            num_frames=int(len(frames)),
            agent_termination_proposal_records=frozen_records,
            episode_options=dict(episode_options or {}),
            observations=observations,
            actions=actions,
            action_extras=action_extras,
            observer_outputs=observer_outputs,
            pre_action_facts=pre_action_facts,
            final_observation=final_obs_dict,
            episode_metrics=dict(episode_metrics or {}),
            physics_steps=physics_steps,
            run_id=str(prov.get("run_id", "")),
            collection_round=int(prov.get("collection_round", 0)),
            job_index=int(prov.get("job_index", 0)),
            episode_seed=int(prov.get("episode_seed", base_seed)),
            job_key=str(prov.get("job_key", "")),
            policy_fingerprints=dict(prov.get("policy_fingerprints", {})),
            behavior=dict(prov.get("behavior", {})),
            worker_id=int(prov.get("worker_id", 0)),
            wall_time_s=float(prov.get("wall_time_s", 0.0)),
        )


__all__ = [
    "COLLECTION_CONTRACT_VERSION",
    "CollectedEpisode",
    "blueprint_hash",
    "mapping_hash",
]
