"""TaggedReplay — trajectory-continuous replay buffer for SAC V2.

Stores transitions flat but tracks trajectory boundaries for n-step
return computation. Per-channel rewards, dones, and actor_weights are
stored independently. Tags and reward_features support stratification
and relabeling.

Key design decisions (see DECISIONS.md):
- In-memory only, no disk persistence.
- Trajectory-continuous: (traj_id, traj_step) pairs enable n-step.
- Per-channel done: terminated → no bootstrap, truncated → bootstrap.
- Thread-safe write interface (for future async rollout).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .experiment import TrajectorySlice


# ---------------------------------------------------------------------------
# TaggedReplay buffer
# ---------------------------------------------------------------------------

class TaggedReplay:
    """Fixed-capacity circular replay buffer with trajectory tracking.

    Stores transitions as flat arrays but maintains (traj_id, traj_step)
    pairs so n-step returns can be computed by walking forward within
    the same trajectory.

    Per-channel data:
    - ``rewards[channel]``: (capacity,) float32 — per-step reward.
    - ``dones[channel]``: (capacity,) bool — per-step termination.
    - ``actor_weights[channel]``: (capacity,) float32 — per-step weight.

    Tags: ``tags[tag_name]``: (capacity,) float32 — per-step tags.
    Reward features: ``reward_features[feat_name]``: (capacity,) float32.

    Trajectory tracking:
    - ``traj_ids``: (capacity,) int32 — which trajectory this transition
      belongs to.
    - ``traj_steps``: (capacity,) int32 — position within trajectory.
    - ``traj_lengths``: Dict[traj_id, int] — length of each trajectory.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        channel_names: Tuple[str, ...],
        tag_names: Tuple[str, ...] = (),
        reward_feature_names: Tuple[str, ...] = (),
        store_core_state: bool = False,
        core_state_dim: int = 0,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.channel_names = tuple(channel_names)
        self.tag_names = tuple(tag_names)
        self.reward_feature_names = tuple(reward_feature_names)
        self.store_core_state = bool(store_core_state)
        self.core_state_dim = int(core_state_dim)

        self.size = 0
        self.ptr = 0
        self._lock = threading.Lock()

        # Core transition data
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)

        # Trajectory tracking
        self.traj_ids = np.full(capacity, -1, dtype=np.int32)
        self.traj_steps = np.zeros(capacity, dtype=np.int32)

        # Per-channel data
        self.rewards: Dict[str, np.ndarray] = {
            ch: np.zeros(capacity, dtype=np.float32) for ch in channel_names
        }
        self.dones: Dict[str, np.ndarray] = {
            ch: np.zeros(capacity, dtype=bool) for ch in channel_names
        }
        self.actor_weights: Dict[str, np.ndarray] = {
            ch: np.zeros(capacity, dtype=np.float32) for ch in channel_names
        }

        # Tags
        self.tags: Dict[str, np.ndarray] = {
            tag: np.zeros(capacity, dtype=np.float32) for tag in tag_names
        }

        # Reward features (for relabeling)
        self.reward_features: Dict[str, np.ndarray] = {
            feat: np.zeros(capacity, dtype=np.float32)
            for feat in reward_feature_names
        }

        # Core state (for buffer-based reset, Phase 2)
        if self.store_core_state and self.core_state_dim > 0:
            self.core_states = np.zeros(
                (capacity, self.core_state_dim), dtype=np.float32,
            )
        else:
            self.core_states = None

        # Sample weights (importance)
        self.sample_weights = np.ones(capacity, dtype=np.float32)

        # Trajectory length tracking
        self._next_traj_id = 0
        self._traj_lengths: Dict[int, int] = {}

        # Relabel version tracking
        self._relabel_version = 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_slices(self, slices: List[TrajectorySlice]) -> int:
        """Add a list of trajectory slices to the buffer.

        Each slice is a contiguous segment. Transitions within a slice
        get the same traj_id, enabling n-step return computation.

        Returns the number of transitions added.
        """
        n_added = 0
        with self._lock:
            for slc in slices:
                n = self._add_one_slice(slc)
                n_added += n
        return n_added

    def _add_one_slice(self, slc: TrajectorySlice) -> int:
        """Add a single trajectory slice. Caller holds the lock."""
        T = len(slc.obs)
        if T == 0:
            return 0

        traj_id = self._next_traj_id
        self._next_traj_id += 1
        self._traj_lengths[traj_id] = T

        obs = np.asarray(slc.obs, dtype=np.float32)
        acts = np.asarray(slc.actions, dtype=np.float32)
        last_obs = np.asarray(slc.last_obs, dtype=np.float32)

        for t in range(T):
            p = self.ptr

            self.obs[p] = obs[t]
            self.actions[p] = acts[t]

            # next_obs: obs[t+1] for t < T-1, last_obs for t = T-1
            if t < T - 1:
                self.next_obs[p] = obs[t + 1]
            else:
                self.next_obs[p] = last_obs

            self.traj_ids[p] = traj_id
            self.traj_steps[p] = t

            # Per-channel data
            for ch in self.channel_names:
                if ch in slc.rewards:
                    self.rewards[ch][p] = float(slc.rewards[ch][t])
                if ch in slc.dones:
                    self.dones[ch][p] = bool(slc.dones[ch][t])
                if ch in slc.actor_weights:
                    self.actor_weights[ch][p] = float(slc.actor_weights[ch][t])

            # Tags
            for tag in self.tag_names:
                if tag in slc.tags:
                    self.tags[tag][p] = float(slc.tags[tag][t])

            # Reward features
            for feat in self.reward_feature_names:
                if feat in slc.reward_features:
                    self.reward_features[feat][p] = float(
                        slc.reward_features[feat][t]
                    )

            # Core state
            if self.core_states is not None and slc.core_states is not None:
                self.core_states[p] = slc.core_states[t]

            self.sample_weights[p] = float(slc.importance)

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        return T

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        channel_names: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """Sample a random minibatch and return as GPU tensors.

        Returns dict with:
        - ``obs``, ``actions``, ``next_obs``: (B, *) tensors.
        - ``rewards_<ch>``: (B,) per channel.
        - ``dones_<ch>``: (B,) per channel.
        - ``actor_weights_<ch>``: (B,) per channel.
        - ``sample_weights``: (B,).
        - ``indices``: (B,) numpy int array — for priority updates
          (Phase 2).
        - ``n_step_info``: Dict per channel with n-step transition
          indices and discount factors (see ``sample_nstep``).

        For n-step returns, use ``sample_nstep`` instead.
        """
        ch_names = channel_names or self.channel_names
        idx = np.random.randint(0, self.size, size=batch_size)

        batch: Dict[str, Any] = {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "sample_weights": torch.as_tensor(
                self.sample_weights[idx], dtype=torch.float32, device=device,
            ),
            "indices": idx,
        }

        for ch in ch_names:
            batch[f"rewards_{ch}"] = torch.as_tensor(
                self.rewards[ch][idx], dtype=torch.float32, device=device,
            )
            batch[f"dones_{ch}"] = torch.as_tensor(
                self.dones[ch][idx], dtype=torch.float32, device=device,
            )
            batch[f"actor_weights_{ch}"] = torch.as_tensor(
                self.actor_weights[ch][idx], dtype=torch.float32, device=device,
            )

        return batch

    def sample_nstep(
        self,
        batch_size: int,
        device: torch.device,
        n_steps: Dict[str, int],
    ) -> Dict[str, Any]:
        """Sample a minibatch with n-step return computation info.

        For each channel, ``n_steps[channel]`` specifies the n-step
        horizon. The returned batch contains, per channel:
        - ``rewards_<ch>``: (B, n) — rewards for n steps (padded with 0).
        - ``dones_<ch>``: (B, n) — done flags for n steps.
        - ``next_obs_<ch>``: (B, obs_dim) — the obs at step t+n_step
          (or last valid obs if trajectory ended earlier).
        - ``valid_steps_<ch>``: (B,) — actual number of valid steps
          (min of n_step and remaining trajectory length).
        - ``discount_<ch>``: (B,) — gamma^n_valid for bootstrap.

        Also contains the standard ``obs``, ``actions``,
        ``actor_weights_<ch>``, ``sample_weights``.
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        batch: Dict[str, Any] = {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "sample_weights": torch.as_tensor(
                self.sample_weights[idx], dtype=torch.float32, device=device,
            ),
            "indices": idx,
        }

        for ch in self.channel_names:
            n = n_steps.get(ch, 1)
            aw = self.actor_weights[ch][idx]
            batch[f"actor_weights_{ch}"] = torch.as_tensor(
                aw, dtype=torch.float32, device=device,
            )

            # Build n-step reward, done, next_obs arrays
            rewards_n = np.zeros((batch_size, n), dtype=np.float32)
            dones_n = np.zeros((batch_size, n), dtype=np.float32)
            next_obs_n = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
            valid_steps = np.zeros(batch_size, dtype=np.int32)

            for i, start_idx in enumerate(idx):
                traj_id = int(self.traj_ids[start_idx])
                traj_step = int(self.traj_steps[start_idx])
                traj_len = self._traj_lengths.get(traj_id, 0)

                # How many steps can we take within this trajectory?
                remaining = traj_len - traj_step
                n_valid = min(n, remaining)
                valid_steps[i] = n_valid

                if n_valid == 0:
                    # Shouldn't happen (start_idx is always valid),
                    # but guard anyway.
                    next_obs_n[i] = self.next_obs[start_idx]
                    continue

                for k in range(n_valid):
                    p = (start_idx + k) % self.capacity
                    rewards_n[i, k] = self.rewards[ch][p]
                    dones_n[i, k] = float(self.dones[ch][p])

                # next_obs is the obs after the last valid step
                last_p = (start_idx + n_valid - 1) % self.capacity
                next_obs_n[i] = self.next_obs[last_p]

                # If any intermediate step was done, truncate
                done_mask = dones_n[i, :n_valid]
                first_done = np.argmax(done_mask > 0.5) if np.any(done_mask > 0.5) else n_valid
                if first_done < n_valid:
                    # Episode ended at first_done (0-indexed within the
                    # n-step window). The next_obs should be the obs
                    # after that step, and rewards beyond are 0.
                    done_p = (start_idx + first_done) % self.capacity
                    next_obs_n[i] = self.next_obs[done_p]
                    rewards_n[i, first_done + 1:] = 0.0
                    dones_n[i, first_done + 1:] = 0.0
                    valid_steps[i] = first_done + 1

            batch[f"rewards_{ch}"] = torch.as_tensor(
                rewards_n, dtype=torch.float32, device=device,
            )
            batch[f"dones_{ch}"] = torch.as_tensor(
                dones_n, dtype=torch.float32, device=device,
            )
            batch[f"next_obs_{ch}"] = torch.as_tensor(
                next_obs_n, dtype=torch.float32, device=device,
            )
            batch[f"valid_steps_{ch}"] = torch.as_tensor(
                valid_steps, dtype=torch.float32, device=device,
            )

        return batch

    # ------------------------------------------------------------------
    # Relabel
    # ------------------------------------------------------------------

    def relabel(
        self,
        relabel_fn,
        ctx: Dict[str, Any],
        batch_size: int = 100_000,
    ) -> int:
        """Recompute rewards and actor_weights for all transitions.

        Scans the entire buffer in batches, calling ``relabel_fn`` on
        each batch. ``relabel_fn`` receives (reward_features_dict,
        tags_dict, ctx) and returns (rewards_dict, actor_weights_dict).

        Returns the number of transitions relabeled.
        """
        if not self.reward_feature_names:
            return 0

        n_relabeled = 0
        with self._lock:
            for start in range(0, self.size, batch_size):
                end = min(start + batch_size, self.size)
                n = end - start

                feat_batch = {
                    feat: self.reward_features[feat][start:end]
                    for feat in self.reward_feature_names
                }
                tag_batch = {
                    tag: self.tags[tag][start:end]
                    for tag in self.tag_names
                }

                result = relabel_fn(feat_batch, tag_batch, ctx)
                if result is None:
                    continue

                new_rewards, new_aw = result
                for ch in self.channel_names:
                    if ch in new_rewards:
                        self.rewards[ch][start:end] = np.asarray(
                            new_rewards[ch][:n], dtype=np.float32,
                        )
                    if ch in new_aw:
                        self.actor_weights[ch][start:end] = np.asarray(
                            new_aw[ch][:n], dtype=np.float32,
                        )

                n_relabeled += n

        self._relabel_version += 1
        return n_relabeled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def buffer_stats(self) -> Dict[str, Any]:
        """Compute buffer composition statistics for logging."""
        if self.size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "n_trajectories": 0,
                "utilization": 0.0,
                "per_channel": {},
                "tag_stats": {},
            }

        per_channel: Dict[str, Dict[str, float]] = {}
        for ch in self.channel_names:
            r = self.rewards[ch][:self.size]
            aw = self.actor_weights[ch][:self.size]
            dn = self.dones[ch][:self.size]
            per_channel[ch] = {
                "reward_mean": float(r.mean()),
                "reward_std": float(r.std()),
                "reward_min": float(r.min()),
                "reward_max": float(r.max()),
                "aw_mean": float(aw.mean()),
                "aw_min": float(aw.min()),
                "aw_max": float(aw.max()),
                "done_rate": float(dn.mean()),
                "active_rate": float((aw > 0).mean()),
            }

        tag_stats: Dict[str, Dict[str, float]] = {}
        for tag in self.tag_names:
            t = self.tags[tag][:self.size]
            tag_stats[tag] = {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "min": float(t.min()),
                "max": float(t.max()),
            }

        return {
            "size": self.size,
            "capacity": self.capacity,
            "n_trajectories": len(self._traj_lengths),
            "utilization": self.size / self.capacity,
            "per_channel": per_channel,
            "tag_stats": tag_stats,
            "relabel_version": self._relabel_version,
        }

    def __len__(self) -> int:
        return self.size
