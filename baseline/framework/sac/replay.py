"""SAC replay buffer implementing ``sac_transition_v1`` admission.

The first-version contract is deliberately strict: 1-step transitions,
FIFO retention, and independent uniform sampling.  Slot index is not an
identity; each admitted row gets a monotone ``sample_id`` and a persistent
``source_key``.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .experiment import ReplayPlan
from .transition import SACTransitionSlice, validate_transition_slice


SAC_REPLAY_SCHEMA = "sac_replay_v1"


class SACReplayError(RuntimeError):
    """Raised for replay admission, sampling, or persistence failures."""


class SACReplayBuffer:
    """FIFO uniform replay over validated SAC transition slices."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        channel_names: Sequence[str],
        *,
        task_fact_names: Sequence[str] = (),
        reward_feature_names: Sequence[str] = (),
        rng_seed: int = 0,
        replay_plan: Optional[ReplayPlan] = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"replay capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.channel_names = tuple(str(name) for name in channel_names)
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be non-empty and unique")
        if replay_plan is not None:
            self._validate_replay_plan(replay_plan)

        self.size = 0
        self.ptr = 0
        self.total_inserted = 0
        self.overwritten = 0
        self._next_sample_id = 0
        self._lock = threading.RLock()
        self.rng = np.random.default_rng(int(rng_seed))

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, len(self.channel_names)), dtype=np.float32)
        self.channel_valid = np.zeros(
            (self.capacity, len(self.channel_names)), dtype=np.bool_
        )
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self.bootstrap = np.ones(self.capacity, dtype=np.float32)
        self.physics_delta = np.zeros(self.capacity, dtype=np.float32)
        self.actor_gate = np.zeros(
            (self.capacity, len(self.channel_names)), dtype=np.float32
        )
        self.actor_weight = np.zeros(
            (self.capacity, len(self.channel_names)), dtype=np.float32
        )
        self.sample_weight = np.ones(self.capacity, dtype=np.float32)
        self.policy_action = np.zeros(
            (self.capacity, self.action_dim), dtype=np.float32
        )

        self.sample_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.source_keys: List[Optional[str]] = [None] * self.capacity
        self.slice_ids: List[Optional[str]] = [None] * self.capacity
        self.frame_indices = np.full(self.capacity, -1, dtype=np.int64)
        self.metadata: List[Optional[Dict[str, Any]]] = [None] * self.capacity

        self._task_fact_names = tuple(task_fact_names)
        self._reward_feature_names = tuple(reward_feature_names)
        self.task_facts = {
            name: np.zeros(self.capacity, dtype=np.float32)
            for name in self._task_fact_names
        }
        self.reward_features = {
            name: np.zeros(self.capacity, dtype=np.float32)
            for name in self._reward_feature_names
        }

        self._seen_source_keys: set[str] = set()
        self._sample_by_source: Dict[str, int] = {}
        self._source_by_sample: Dict[int, str] = {}

    @staticmethod
    def _validate_replay_plan(plan: ReplayPlan) -> None:
        unsupported = []
        if plan.stratify_by is not None:
            unsupported.append(f"stratify_by={plan.stratify_by!r}")
        if float(plan.freshness_weight) != 0.0:
            unsupported.append(f"freshness_weight={plan.freshness_weight}")
        if unsupported:
            raise SACReplayError(
                "sac_replay_v1 does not support "
                + ", ".join(unsupported)
            )

    def _ensure_dynamic_fields(self, sl: SACTransitionSlice) -> None:
        if not self._task_fact_names and not self.task_facts:
            self._task_fact_names = tuple(sorted(sl.task_facts))
            self.task_facts = {
                name: np.zeros(self.capacity, dtype=np.float32)
                for name in self._task_fact_names
            }
        if not self._reward_feature_names and not self.reward_features:
            self._reward_feature_names = tuple(sorted(sl.reward_features))
            self.reward_features = {
                name: np.zeros(self.capacity, dtype=np.float32)
                for name in self._reward_feature_names
            }
        if set(sl.task_facts) != set(self._task_fact_names):
            raise SACReplayError(
                f"task_fact names changed: expected {self._task_fact_names}, "
                f"got {tuple(sorted(sl.task_facts))}"
            )
        if set(sl.reward_features) != set(self._reward_feature_names):
            raise SACReplayError(
                f"reward_feature names changed: expected "
                f"{self._reward_feature_names}, got "
                f"{tuple(sorted(sl.reward_features))}"
            )

    def add_slices(self, slices: Iterable[SACTransitionSlice]) -> int:
        """Atomically validate and insert all supplied slices."""
        validated = [validate_transition_slice(sl) for sl in slices if sl is not None]
        if not validated:
            return 0

        new_keys: List[str] = []
        for sl in validated:
            new_keys.extend(sl.source_keys)
        if len(new_keys) != len(set(new_keys)):
            raise SACReplayError("duplicate source_key inside incoming slices")
        conflicts = sorted(set(new_keys) & self._seen_source_keys)
        if conflicts:
            raise SACReplayError(
                f"source_key conflicts with previously admitted samples: "
                f"{conflicts[:4]}"
            )
        for sl in validated:
            self._ensure_dynamic_fields(sl)

        added = 0
        with self._lock:
            for sl in validated:
                for t in range(sl.num_transitions):
                    self._insert_one(sl, t)
                    added += 1
        return added

    def _insert_one(self, sl: SACTransitionSlice, t: int) -> None:
        p = self.ptr
        if self.size == self.capacity:
            old_key = self.source_keys[p]
            old_sample = int(self.sample_ids[p])
            if old_key is not None:
                self._sample_by_source.pop(old_key, None)
            if old_sample >= 0:
                self._source_by_sample.pop(old_sample, None)
            self.overwritten += 1
        else:
            self.size += 1

        sample_id = int(self._next_sample_id)
        self._next_sample_id += 1
        source_key = str(sl.source_keys[t])

        self.obs[p] = sl.obs[t]
        self.actions[p] = sl.actions[t]
        self.next_obs[p] = sl.next_obs[t]
        self.rewards[p] = sl.rewards[t]
        self.channel_valid[p] = sl.channel_valid[t]
        self.terminated[p] = sl.terminated[t]
        self.truncated[p] = sl.truncated[t]
        self.bootstrap[p] = sl.bootstrap[t]
        self.physics_delta[p] = sl.physics_delta[t]
        self.actor_gate[p] = sl.actor_gate[t]
        self.actor_weight[p] = sl.actor_weight[t]
        self.sample_weight[p] = sl.sample_weight[t]
        self.policy_action[p] = (
            sl.actions[t] if sl.policy_action is None else sl.policy_action[t]
        )

        for name, arr in self.task_facts.items():
            arr[p] = float(sl.task_facts[name][t])
        for name, arr in self.reward_features.items():
            arr[p] = float(sl.reward_features[name][t])

        self.sample_ids[p] = sample_id
        self.source_keys[p] = source_key
        self.slice_ids[p] = str(sl.slice_id)
        self.frame_indices[p] = int(t)
        self.metadata[p] = {
            "sample_id": sample_id,
            "source_key": source_key,
            "slice_id": str(sl.slice_id),
            "frame_index": int(t),
            "termination_reason": str(sl.termination_reason[t]),
            "behavior": dict(sl.behavior),
            "collection": dict(sl.collection),
            "versions": dict(sl.versions),
        }
        self._seen_source_keys.add(source_key)
        self._sample_by_source[source_key] = sample_id
        self._source_by_sample[sample_id] = source_key
        self.ptr = (self.ptr + 1) % self.capacity
        self.total_inserted += 1

    def _live_slots(self) -> np.ndarray:
        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.arange(self.size, dtype=np.int64)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        channel_names: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """Sample unique live transitions uniformly with the replay RNG."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        with self._lock:
            if batch_size > self.size:
                raise SACReplayError(
                    f"cannot sample {batch_size} unique transitions from "
                    f"size={self.size}"
                )
            slots = self.rng.choice(
                self._live_slots(), size=batch_size, replace=False,
            )
            sample_ids = np.asarray(self.sample_ids[slots], dtype=np.int64)
            if len(np.unique(sample_ids)) != batch_size:
                raise SACReplayError("sampled batch contains duplicate sample_id")

            batch: Dict[str, Any] = {
                "obs": torch.as_tensor(self.obs[slots], device=device),
                "actions": torch.as_tensor(self.actions[slots], device=device),
                "next_obs": torch.as_tensor(self.next_obs[slots], device=device),
                "rewards": torch.as_tensor(self.rewards[slots], device=device),
                "channel_valid": torch.as_tensor(self.channel_valid[slots], device=device),
                "terminated": torch.as_tensor(self.terminated[slots], device=device),
                "truncated": torch.as_tensor(self.truncated[slots], device=device),
                "bootstrap": torch.as_tensor(self.bootstrap[slots], device=device),
                "physics_delta": torch.as_tensor(self.physics_delta[slots], device=device),
                "actor_gate": torch.as_tensor(self.actor_gate[slots], device=device),
                "actor_weight": torch.as_tensor(self.actor_weight[slots], device=device),
                "sample_weight": torch.as_tensor(self.sample_weight[slots], device=device),
                "policy_action": torch.as_tensor(self.policy_action[slots], device=device),
                "sample_ids": torch.as_tensor(sample_ids, device=device),
                "source_keys": [self.source_keys[int(slot)] for slot in slots],
                "metadata": [self.metadata[int(slot)] for slot in slots],
                "indices": np.asarray(slots, dtype=np.int64),
                "task_facts": {
                    name: torch.as_tensor(arr[slots], device=device)
                    for name, arr in self.task_facts.items()
                },
                "reward_features": {
                    name: torch.as_tensor(arr[slots], device=device)
                    for name, arr in self.reward_features.items()
                },
            }

            selected = channel_names or self.channel_names
            for ch in selected:
                if ch not in self.channel_names:
                    raise KeyError(f"unknown replay channel {ch!r}")
                c = self.channel_names.index(ch)
                batch[f"rewards_{ch}"] = batch["rewards"][:, c]
                batch[f"dones_{ch}"] = batch["terminated"].float()
                batch[f"actor_weights_{ch}"] = batch["actor_weight"][:, c]
            return batch

    def get_by_sample_ids(self, sample_ids: Sequence[int]) -> Dict[str, Any]:
        wanted = {int(s) for s in sample_ids}
        slots = [
            i for i in range(self.size)
            if int(self.sample_ids[i]) in wanted
        ]
        found = {int(self.sample_ids[i]) for i in slots}
        missing = wanted - found
        if missing:
            raise KeyError(f"sample_ids not live in replay: {sorted(missing)}")
        idx = np.asarray(slots, dtype=np.int64)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "next_obs": self.next_obs[idx],
            "rewards": self.rewards[idx],
            "terminated": self.terminated[idx],
            "truncated": self.truncated[idx],
            "bootstrap": self.bootstrap[idx],
            "actor_gate": self.actor_gate[idx],
            "actor_weight": self.actor_weight[idx],
            "sample_weight": self.sample_weight[idx],
            "policy_action": self.policy_action[idx],
            "sample_ids": self.sample_ids[idx],
            "source_keys": [self.source_keys[i] for i in slots],
            "metadata": [self.metadata[i] for i in slots],
            "task_facts": {name: arr[idx] for name, arr in self.task_facts.items()},
            "reward_features": {
                name: arr[idx] for name, arr in self.reward_features.items()
            },
        }

    def sample_nstep(self, *args: Any, **kwargs: Any) -> None:
        raise SACReplayError(
            "sac_replay_v1 supports only 1-step transitions; sample_nstep "
            "is explicitly disabled"
        )

    def relabel(self, *args: Any, **kwargs: Any) -> None:
        raise SACReplayError(
            "sac_replay_v1 does not support relabeling"
        )

    def buffer_stats(self) -> Dict[str, Any]:
        per_channel: Dict[str, Dict[str, float]] = {}
        for c, name in enumerate(self.channel_names):
            rewards = self.rewards[: self.size, c]
            gates = self.actor_gate[: self.size, c]
            weights = self.actor_weight[: self.size, c]
            per_channel[name] = {
                "reward_mean": float(rewards.mean()) if self.size else 0.0,
                "reward_std": float(rewards.std()) if self.size else 0.0,
                "gate_mean": float(gates.mean()) if self.size else 0.0,
                "weight_mean": float(weights.mean()) if self.size else 0.0,
            }
        sample_ids = self.sample_ids[: self.size]
        ages = (
            self._next_sample_id - sample_ids
            if self.size else np.empty(0, dtype=np.int64)
        )
        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
            "total_inserted": self.total_inserted,
            "overwritten": self.overwritten,
            "next_sample_id": self._next_sample_id,
            "sample_id_min": int(sample_ids.min()) if self.size else -1,
            "sample_id_max": int(sample_ids.max()) if self.size else -1,
            "sample_age_mean": float(ages.mean()) if self.size else 0.0,
            "per_channel": per_channel,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": SAC_REPLAY_SCHEMA,
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "channel_names": self.channel_names,
            "task_fact_names": self._task_fact_names,
            "reward_feature_names": self._reward_feature_names,
            "size": self.size,
            "ptr": self.ptr,
            "total_inserted": self.total_inserted,
            "overwritten": self.overwritten,
            "next_sample_id": self._next_sample_id,
            "obs": self.obs,
            "actions": self.actions,
            "next_obs": self.next_obs,
            "rewards": self.rewards,
            "channel_valid": self.channel_valid,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "bootstrap": self.bootstrap,
            "physics_delta": self.physics_delta,
            "actor_gate": self.actor_gate,
            "actor_weight": self.actor_weight,
            "sample_weight": self.sample_weight,
            "policy_action": self.policy_action,
            "sample_ids": self.sample_ids,
            "source_keys": self.source_keys,
            "slice_ids": self.slice_ids,
            "frame_indices": self.frame_indices,
            "metadata": self.metadata,
            "task_facts": self.task_facts,
            "reward_features": self.reward_features,
            "seen_source_keys": self._seen_source_keys,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != SAC_REPLAY_SCHEMA:
            raise SACReplayError(
                f"unsupported replay schema {state.get('schema')!r}"
            )
        if (
            int(state["capacity"]) != self.capacity
            or int(state["obs_dim"]) != self.obs_dim
            or int(state["action_dim"]) != self.action_dim
            or tuple(state["channel_names"]) != self.channel_names
        ):
            raise SACReplayError("replay state is incompatible with buffer shape")

        with self._lock:
            self.size = int(state["size"])
            self.ptr = int(state["ptr"])
            self.total_inserted = int(state["total_inserted"])
            self.overwritten = int(state["overwritten"])
            self._next_sample_id = int(state["next_sample_id"])
            for name in (
                "obs", "actions", "next_obs", "rewards", "channel_valid",
                "terminated", "truncated", "bootstrap", "physics_delta",
                "actor_gate", "actor_weight", "sample_weight", "policy_action",
                "sample_ids", "frame_indices",
            ):
                getattr(self, name)[:] = np.asarray(state[name])
            self.source_keys = list(state["source_keys"])
            self.slice_ids = list(state["slice_ids"])
            self.metadata = list(state["metadata"])
            self._task_fact_names = tuple(state["task_fact_names"])
            self._reward_feature_names = tuple(state["reward_feature_names"])
            self.task_facts = dict(state["task_facts"])
            self.reward_features = dict(state["reward_features"])
            self._seen_source_keys = set(state["seen_source_keys"])
            self._sample_by_source = {
                str(key): int(sample_id)
                for key, sample_id in zip(self.source_keys, self.sample_ids)
                if key is not None and int(sample_id) >= 0
            }
            self._source_by_sample = {
                sample_id: key
                for key, sample_id in self._sample_by_source.items()
            }
            self.rng.bit_generator.state = dict(state["rng_state"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(self.state_dict(), tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "SACReplayBuffer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        replay = cls(
            capacity=int(state["capacity"]),
            obs_dim=int(state["obs_dim"]),
            action_dim=int(state["action_dim"]),
            channel_names=tuple(state["channel_names"]),
            task_fact_names=tuple(state["task_fact_names"]),
            reward_feature_names=tuple(state["reward_feature_names"]),
        )
        replay.load_state_dict(state)
        return replay

    def __len__(self) -> int:
        return self.size


__all__ = [
    "SAC_REPLAY_SCHEMA",
    "SACReplayBuffer",
    "SACReplayError",
]
