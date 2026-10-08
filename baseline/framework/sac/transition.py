"""SAC transition/slice schema and strict validation.

``sac_transition_v2`` is the replay admission unit.  It is intentionally
independent from PPO trajectory structures: no advantages, ratios, GAE, or
sampling-context fields are part of this contract.  Compared with v1 it makes
the next-state actor gate explicit so target construction does not infer it
from task-specific features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .collected_episode import CollectedEpisode


SAC_TRANSITION_SCHEMA = "sac_transition_v2"
ACTION_BOUNDS_ATOL = 1e-5


@dataclass(frozen=True)
class SACTransitionSlice:
    """Contiguous per-agent transition slice.

    Metadata is deliberately per-slice; ``source_keys`` and semantic fields are
    per transition. ``sample_id`` is intentionally absent: replay assigns it on
    admission.
    """

    schema_version: str
    slice_id: str
    obs: np.ndarray
    actions: np.ndarray
    next_obs: np.ndarray
    rewards: np.ndarray
    channel_valid: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap: np.ndarray
    termination_reason: Tuple[str, ...]
    physics_delta: np.ndarray
    actor_gate: np.ndarray
    actor_weight: np.ndarray
    actor_gate_next: np.ndarray
    actor_weight_next: np.ndarray
    sample_weight: np.ndarray
    task_facts: Mapping[str, np.ndarray] = field(default_factory=dict)
    reward_features: Mapping[str, np.ndarray] = field(default_factory=dict)
    policy_action: Optional[np.ndarray] = None
    source_keys: Tuple[str, ...] = ()
    behavior: Mapping[str, Any] = field(default_factory=dict)
    collection: Mapping[str, Any] = field(default_factory=dict)
    versions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_transitions(self) -> int:
        return int(self.obs.shape[0])


def _as_float_array(name: str, value: Any, *, ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_bool_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype != np.bool_:
        raise TypeError(f"{name} must be bool, got dtype {arr.dtype}")
    return arr


def _check_meta_dict(name: str, value: Mapping[str, Any], required: Sequence[str]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{name} missing required keys {missing}")


def validate_transition_slice(
    sl: SACTransitionSlice,
    *,
    required_task_facts: Sequence[str] = (),
) -> SACTransitionSlice:
    """Validate one ``sac_transition_v2`` slice, failing loudly."""

    if not isinstance(sl, SACTransitionSlice):
        raise TypeError(
            f"expected SACTransitionSlice, got {type(sl).__name__}"
        )
    if sl.schema_version != SAC_TRANSITION_SCHEMA:
        raise ValueError(
            f"unsupported transition schema {sl.schema_version!r}; expected "
            f"{SAC_TRANSITION_SCHEMA!r}"
        )
    if not sl.slice_id:
        raise ValueError("slice_id must be non-empty")

    obs = _as_float_array("obs", sl.obs, ndim=2)
    actions = _as_float_array("actions", sl.actions, ndim=2)
    next_obs = _as_float_array("next_obs", sl.next_obs, ndim=2)
    T, obs_dim = obs.shape
    if T <= 0:
        raise ValueError("transition slice must contain at least one transition")
    if next_obs.shape != obs.shape:
        raise ValueError(
            f"next_obs shape {next_obs.shape} does not match obs {obs.shape}"
        )
    if actions.shape[0] != T or actions.shape[1] <= 0:
        raise ValueError(f"invalid actions shape {actions.shape} for T={T}")
    if np.any(actions < -1.0 - ACTION_BOUNDS_ATOL) or np.any(
        actions > 1.0 + ACTION_BOUNDS_ATOL
    ):
        raise ValueError("actions contain values outside [-1, 1]")

    rewards = _as_float_array("rewards", sl.rewards, ndim=2)
    channel_valid = np.asarray(sl.channel_valid)
    actor_gate = _as_float_array("actor_gate", sl.actor_gate, ndim=2)
    actor_weight = _as_float_array("actor_weight", sl.actor_weight, ndim=2)
    actor_gate_next = _as_float_array("actor_gate_next", sl.actor_gate_next, ndim=2)
    actor_weight_next = _as_float_array(
        "actor_weight_next", sl.actor_weight_next, ndim=2,
    )
    if rewards.shape[0] != T or rewards.shape[1] <= 0:
        raise ValueError(f"invalid rewards shape {rewards.shape} for T={T}")
    C = rewards.shape[1]
    if channel_valid.shape != (T, C) or channel_valid.dtype != np.bool_:
        raise ValueError(
            f"channel_valid must be bool with shape {(T, C)}, got "
            f"{channel_valid.shape}/{channel_valid.dtype}"
        )
    if actor_gate.shape != (T, C):
        raise ValueError(
            f"actor_gate shape {actor_gate.shape} != {(T, C)}"
        )
    if actor_weight.shape != (T, C):
        raise ValueError(
            f"actor_weight shape {actor_weight.shape} != {(T, C)}"
        )
    if actor_gate_next.shape != (T, C):
        raise ValueError(
            f"actor_gate_next shape {actor_gate_next.shape} != {(T, C)}"
        )
    if actor_weight_next.shape != (T, C):
        raise ValueError(
            f"actor_weight_next shape {actor_weight_next.shape} != {(T, C)}"
        )
    for name, gate, weight in (
        ("actor_gate", actor_gate, actor_weight),
        ("actor_gate_next", actor_gate_next, actor_weight_next),
    ):
        if np.any(gate < 0.0):
            raise ValueError(f"{name} must be non-negative")
        gate_sum = gate.sum(axis=1)
        if np.any(gate_sum <= 0.0):
            raise ValueError(f"{name} row sums must be positive")
        expected_weight = gate / gate_sum[:, None]
        if not np.allclose(weight, expected_weight, atol=1e-5, rtol=1e-5):
            raise ValueError(
                f"{name.replace('_gate', '_weight')} must equal normalized {name}"
            )

    terminated = _as_bool_array("terminated", sl.terminated)
    truncated = _as_bool_array("truncated", sl.truncated)
    bootstrap = _as_float_array("bootstrap", sl.bootstrap, ndim=1)
    physics_delta = _as_float_array("physics_delta", sl.physics_delta, ndim=1)
    sample_weight = _as_float_array("sample_weight", sl.sample_weight, ndim=1)
    for name, arr in (
        ("terminated", terminated),
        ("truncated", truncated),
        ("bootstrap", bootstrap),
        ("physics_delta", physics_delta),
        ("sample_weight", sample_weight),
    ):
        if arr.shape[0] != T:
            raise ValueError(f"{name} length {arr.shape[0]} != T={T}")

    if np.any(terminated & truncated):
        raise ValueError("terminated and truncated cannot both be true")
    if np.any(terminated[:-1]) or np.any(truncated[:-1]):
        raise ValueError("episode-boundary flags may only appear at the last transition")
    if np.any(physics_delta <= 0):
        raise ValueError("physics_delta must be > 0 for every transition")
    if np.any(sample_weight <= 0):
        raise ValueError("sample_weight must be positive")
    if np.any(~np.isin(bootstrap, np.asarray([0.0, 1.0], dtype=np.float32))):
        raise ValueError("bootstrap must contain only 0 or 1")
    if np.any(terminated & (bootstrap != 0.0)):
        raise ValueError("terminated transitions must have bootstrap=0")
    if np.any(truncated & (bootstrap != 1.0)):
        raise ValueError("truncated transitions must have bootstrap=1")
    if np.any((~terminated) & (~truncated) & (bootstrap != 1.0)):
        raise ValueError("non-terminal transitions must have bootstrap=1")

    reasons = tuple(str(reason) for reason in sl.termination_reason)
    if len(reasons) != T:
        raise ValueError(f"termination_reason length {len(reasons)} != T={T}")
    if terminated[-1] or truncated[-1]:
        if not reasons[-1]:
            raise ValueError("terminal transition requires a termination reason")
    elif any(reasons):
        raise ValueError("termination reason present without terminal flag")
    if reasons[-1] == "timeout" and not truncated[-1]:
        raise ValueError("timeout reason must produce truncated=True")
    if reasons[-1] and reasons[-1] != "timeout" and not terminated[-1]:
        raise ValueError("non-timeout terminal reason must produce terminated=True")

    if len(sl.source_keys) != T:
        raise ValueError(f"source_keys length {len(sl.source_keys)} != T={T}")
    if any(not str(key) for key in sl.source_keys):
        raise ValueError("source_keys must be non-empty")
    if len(set(sl.source_keys)) != T:
        raise ValueError("source_keys must be unique within a slice")

    if sl.policy_action is not None:
        policy_action = _as_float_array("policy_action", sl.policy_action, ndim=2)
        if policy_action.shape != actions.shape:
            raise ValueError(
                f"policy_action shape {policy_action.shape} != actions {actions.shape}"
            )

    for mapping_name, mapping in (
        ("task_facts", sl.task_facts),
        ("reward_features", sl.reward_features),
    ):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{mapping_name} must be a mapping")
        for key, value in mapping.items():
            arr = _as_float_array(f"{mapping_name}[{key!r}]", value, ndim=1)
            if arr.shape[0] != T:
                raise ValueError(
                    f"{mapping_name}[{key!r}] length {arr.shape[0]} != T={T}"
                )
    missing_facts = [
        name for name in required_task_facts if name not in sl.task_facts
    ]
    if missing_facts:
        raise ValueError(f"task_facts missing required fields {missing_facts}")

    _check_meta_dict(
        "behavior", sl.behavior,
        ("mode", "explore_factor", "parameters"),
    )
    _check_meta_dict(
        "collection", sl.collection,
        (
            "run_id", "collection_round", "job_index", "agent_id",
            "episode_seed", "job_key", "env_blueprint_hash",
            "policy_fingerprint", "behavior_policy_version",
        ),
    )
    _check_meta_dict(
        "versions", sl.versions,
        ("schema", "reward_semantics", "objective_mode", "regularizer_mode", "policy_arch"),
    )
    return sl


def _slice_agent_array(
    episode: CollectedEpisode,
    agent_id: str,
    field_name: str,
    T: int,
) -> np.ndarray:
    value = getattr(episode, field_name).get(agent_id)
    if value is None:
        raise KeyError(f"episode.{field_name}[{agent_id!r}] missing")
    arr = np.asarray(value)
    if arr.shape[0] < T:
        raise ValueError(
            f"episode.{field_name}[{agent_id!r}] has length {arr.shape[0]} < T={T}"
        )
    return arr


def _stack_agent_extras(
    episode: CollectedEpisode,
    agent_id: str,
    name: str,
    T: int,
) -> Optional[np.ndarray]:
    extras = episode.action_extras.get(agent_id)
    if extras is None or name not in extras:
        return None
    arr = np.asarray(extras[name])
    if arr.shape[0] < T:
        raise ValueError(
            f"action_extras[{agent_id!r}][{name!r}] length {arr.shape[0]} < T={T}"
        )
    return arr


def build_agent_transition_slice(
    episode: CollectedEpisode,
    agent_id: str,
    *,
    channel_names: Sequence[str],
    rewards: Mapping[str, np.ndarray],
    actor_gate: Mapping[str, np.ndarray],
    actor_gate_next: Mapping[str, np.ndarray],
    channel_valid: Optional[Mapping[str, np.ndarray]] = None,
    task_facts: Optional[Mapping[str, np.ndarray]] = None,
    reward_features: Optional[Mapping[str, np.ndarray]] = None,
    sample_weight: float = 1.0,
    versions: Optional[Mapping[str, Any]] = None,
    slice_index: int = 0,
) -> Optional[SACTransitionSlice]:
    """Build one contiguous per-agent ``sac_transition_v2`` slice."""

    T_full = int(episode.num_frames)
    if T_full <= 0:
        return None
    T = int(episode.agent_frame_boundary.get(agent_id, T_full))
    if T <= 0:
        return None

    obs_all = _slice_agent_array(episode, agent_id, "observations", T)
    act_all = _slice_agent_array(episode, agent_id, "actions", T)
    final_obs = episode.final_observation.get(agent_id)
    if final_obs is None:
        raise KeyError(f"episode.final_observation[{agent_id!r}] missing")
    if episode.physics_steps is None:
        raise KeyError("episode.physics_steps missing; cannot detect degenerate frames")
    physics_steps = np.asarray(episode.physics_steps, dtype=np.int64)
    if physics_steps.shape[0] < T:
        raise ValueError(
            f"physics_steps length {physics_steps.shape[0]} < T={T}"
        )
    physics_delta = np.diff(physics_steps[:T], prepend=np.int64(0))
    if np.any(physics_delta <= 0):
        raise ValueError(
            f"non-positive physics_delta inside admitted slice for {agent_id}"
        )

    next_obs = np.empty_like(obs_all[:T])
    next_obs[:-1] = obs_all[1:T]
    if T < obs_all.shape[0]:
        next_obs[-1] = obs_all[T]
    else:
        next_obs[-1] = np.asarray(final_obs)

    terminated = np.zeros(T, dtype=np.bool_)
    truncated = np.zeros(T, dtype=np.bool_)
    bootstrap = np.ones(T, dtype=np.float32)
    reasons = [""] * T
    reason = episode.agent_termination_reason.get(agent_id, "")
    if not reason:
        raise ValueError(
            f"agent {agent_id!r} has no termination reason in collected episode"
        )
    reasons[-1] = str(reason)
    if reason == "timeout":
        truncated[-1] = True
    else:
        terminated[-1] = True
        bootstrap[-1] = 0.0

    def _series(name: str, source: Mapping[str, np.ndarray], key: str) -> np.ndarray:
        if key not in source:
            raise KeyError(f"{name}[{key!r}] missing")
        arr = np.asarray(source[key])
        if arr.ndim != 1 or arr.shape[0] < T:
            raise ValueError(
                f"{name}[{key!r}] must be 1-D with length >= {T}, "
                f"got {arr.shape}"
            )
        return arr[:T].astype(np.float32, copy=False)

    channels = tuple(str(name) for name in channel_names)
    rewards_mat = np.stack(
        [_series("rewards", rewards, name) for name in channels], axis=1,
    )
    gate_mat = np.stack(
        [_series("actor_gate", actor_gate, name) for name in channels], axis=1,
    )
    gate_next_mat = np.stack(
        [
            _series("actor_gate_next", actor_gate_next, name)
            for name in channels
        ],
        axis=1,
    )
    if channel_valid is None:
        valid_mat = np.ones((T, len(channels)), dtype=np.bool_)
    else:
        valid_mat = np.stack(
            [
                np.asarray(_series("channel_valid", channel_valid, name), dtype=np.bool_)
                for name in channels
            ],
            axis=1,
        )

    def _mapping_series(
        label: str,
        source: Optional[Mapping[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, value in (source or {}).items():
            arr = np.asarray(value)
            if arr.ndim != 1 or arr.shape[0] < T:
                raise ValueError(
                    f"{label}[{key!r}] must be 1-D with length >= {T}, "
                    f"got {arr.shape}"
                )
            out[str(key)] = arr[:T]
        return out

    gate_sum = gate_mat.sum(axis=1, keepdims=True)
    if np.any(gate_sum <= 0):
        raise ValueError("actor_gate row sums must be positive")
    weight_mat = gate_mat / gate_sum
    gate_next_sum = gate_next_mat.sum(axis=1, keepdims=True)
    if np.any(gate_next_sum <= 0):
        raise ValueError("actor_gate_next row sums must be positive")
    weight_next_mat = gate_next_mat / gate_next_sum

    policy_action = _stack_agent_extras(
        episode, agent_id, "policy_action", T,
    )
    if (
        policy_action is not None
        and policy_action.shape[1:] != act_all[:T].shape[1:]
    ):
        raise ValueError(
            f"policy_action shape {policy_action.shape} != action "
            f"{act_all[:T].shape}"
        )

    env_hash = str(episode.blueprint_hash)
    behavior = dict(episode.behavior.get(agent_id) or {})
    fingerprint = str(episode.policy_fingerprints.get(agent_id, ""))
    if not fingerprint:
        raise ValueError(f"policy fingerprint missing for {agent_id!r}")
    collection = {
        "run_id": str(episode.run_id),
        "collection_round": int(episode.collection_round),
        "job_index": int(episode.job_index),
        "agent_id": str(agent_id),
        "episode_seed": int(episode.episode_seed),
        "job_key": str(episode.job_key),
        "env_blueprint_hash": env_hash,
        "policy_fingerprint": fingerprint,
        "behavior_policy_version": int(episode.collection_round),
        "worker_id": int(episode.worker_id),
    }
    version_dict = {
        "schema": SAC_TRANSITION_SCHEMA,
        "reward_semantics": "",
        "objective_mode": "",
        "regularizer_mode": "",
        "policy_arch": "",
    }
    version_dict.update(dict(versions or {}))

    identity_prefix = (
        f"{collection['run_id']}/r{int(episode.collection_round):06d}/"
        f"j{int(episode.job_index):06d}/{agent_id}/e{int(episode.episode_seed)}"
    )
    slice_id = f"{identity_prefix}/slice{int(slice_index):04d}"
    source_keys = tuple(
        f"{identity_prefix}/f{index:06d}/h{env_hash}/v{sac_transition_version}"
        for index, sac_transition_version in enumerate(
            [SAC_TRANSITION_SCHEMA] * T
        )
    )

    sl = SACTransitionSlice(
        schema_version=SAC_TRANSITION_SCHEMA,
        slice_id=slice_id,
        obs=np.asarray(obs_all[:T], dtype=np.float32),
        actions=np.asarray(act_all[:T], dtype=np.float32),
        next_obs=np.asarray(next_obs, dtype=np.float32),
        rewards=np.asarray(rewards_mat, dtype=np.float32),
        channel_valid=np.asarray(valid_mat, dtype=np.bool_),
        terminated=terminated,
        truncated=truncated,
        bootstrap=bootstrap,
        termination_reason=tuple(reasons),
        physics_delta=np.asarray(physics_delta, dtype=np.float32),
        actor_gate=np.asarray(gate_mat, dtype=np.float32),
        actor_weight=np.asarray(weight_mat, dtype=np.float32),
        actor_gate_next=np.asarray(gate_next_mat, dtype=np.float32),
        actor_weight_next=np.asarray(weight_next_mat, dtype=np.float32),
        sample_weight=np.full(T, float(sample_weight), dtype=np.float32),
        task_facts=_mapping_series("task_facts", task_facts),
        reward_features=_mapping_series("reward_features", reward_features),
        policy_action=(
            np.asarray(policy_action[:T], dtype=np.float32)
            if policy_action is not None else None
        ),
        source_keys=source_keys,
        behavior=behavior,
        collection=collection,
        versions=version_dict,
    )
    return validate_transition_slice(sl)


__all__ = [
    "SAC_TRANSITION_SCHEMA",
    "SACTransitionSlice",
    "build_agent_transition_slice",
    "validate_transition_slice",
]
