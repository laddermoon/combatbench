"""Golden test for PPOBuffer + GAE computation.

Captures the exact numerical output of the current (v1) PPO buffer and GAE
pipeline across all branch combinations, so that the v2 rewrite can be
verified bit-for-bit.

Branches covered:
  - Single segment, episode terminated (fall)
  - Single segment, episode truncated (timeout)
  - Multi-segment (struggle → stability), mid-episode boundary
  - Multi-segment with key_weights (per-key critic control)
  - Multi-segment with key_termination (per-key termination override)
  - Segment with mode (actor routing)
  - Segment weight != 1.0
  - Episode skipped (empty segments)
  - v1 API (prepare_training_segments) fallback

Run:
  PYTHONPATH=/data1/mono/things/combatbench python3 -m pytest \
    baseline/framework/test_ppo_buffer_golden.py -q
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.common.algos import compute_gae
from baseline.common.rollout.episode import Episode
from baseline.framework.experiment import (
    CommonParams,
    Experiment,
    Segment,
    TrainablePolicy,
)
from baseline.framework.ppo_trainer import PPOBuffer


# ---------------------------------------------------------------------------
# Mock experiment
# ---------------------------------------------------------------------------

class MockExperiment(Experiment):
    """Minimal experiment for golden testing.

    Supports both v1 (prepare_training_segments) and v2 (prepare_segments) APIs.
    """

    def __init__(
        self,
        reward_keys: Tuple[str, ...],
        gammas: Dict[str, float],
        seg_mode: str = "v2",  # "v1" or "v2"
        segments_override: Optional[Dict[int, List[Segment]]] = None,
        v1_segments_override: Optional[Dict[int, List[Tuple]]] = None,
        rewards_override: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    ):
        self._reward_keys = reward_keys
        self._gammas = gammas
        self._seg_mode = seg_mode
        self._segments_override = segments_override or {}
        self._v1_segments_override = v1_segments_override or {}
        self._rewards_override = rewards_override or {}
        self._name = "mock_exp"
        self._obs_dim = 4
        self._action_dim = 2

    def common_params(self) -> CommonParams:
        return CommonParams(
            name=self._name,
            reward_keys=self._reward_keys,
            gammas=self._gammas,
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            learning_rate=1e-4,
            critic_learning_rate=3e-4,
            grad_clip_norm=1.0,
            episodes_per_update=8,
            max_updates=100,
            eval_interval=10,
            eval_episodes=4,
            video_eval_interval=10,
            rollout_workers=1,
            eval_workers=1,
            seed=42,
        )

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        raise NotImplementedError

    def initial_weights(self) -> Tuple[float, ...]:
        return tuple(1.0 for _ in self._reward_keys)

    def next_weights(
        self, eval_metrics: Dict[str, float], current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        return current_weights

    def extract_rewards(self, episode: Episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        ep_idx = episode.episode_index
        if ep_idx in self._rewards_override:
            return self._rewards_override[ep_idx]
        # Default: deterministic per-step rewards
        return {
            key: (np.arange(T, dtype=np.float32) * 0.01 + ord(key[0]) * 0.001)
            for key in self._reward_keys
        }

    def compute_episode_metrics(self, episode: Episode) -> Dict[str, float]:
        return {"length": float(episode.num_frames)}

    def scheduler_info(self) -> Dict[str, Any]:
        return {}

    def compare_eval(self, esum: Dict[str, float], best_esum: Dict[str, float]) -> bool:
        return esum.get("length", 0) > best_esum.get("length", 0)

    def build_rollout_jobs(self, policy_bp, base_seed):
        return []

    def build_eval_jobs(self, policy_bp, base_seed):
        return []

    def video_env_blueprint(self):
        raise NotImplementedError

    def ppo_params(self):
        from baseline.framework.experiment import PPOParams
        return PPOParams(
            log_std_min=-4.0, log_std_max=0.0, gae_lambda=0.95,
            clip_eps=0.2, entropy_coef=1e-3, target_kl=0.05,
            update_epochs=4, minibatch_size=4096,
        )

    def build_v_critic(self, reward_key: str, device: torch.device) -> torch.nn.Module:
        raise NotImplementedError

    def prepare_training_segments(self, episode: Episode):
        T = episode.num_frames
        ep_idx = episode.episode_index
        if ep_idx in self._v1_segments_override:
            return self._v1_segments_override[ep_idx]
        return [(0, T, 1.0)]

    def prepare_segments(self, episode: Episode) -> Optional[List[Segment]]:
        if self._seg_mode == "v1":
            return None  # fall back to prepare_training_segments
        ep_idx = episode.episode_index
        if ep_idx in self._segments_override:
            return self._segments_override[ep_idx]
        return None  # fall back to prepare_training_segments


# ---------------------------------------------------------------------------
# Synthetic episode builder
# ---------------------------------------------------------------------------

def make_episode(
    ep_idx: int = 0,
    T: int = 10,
    obs_dim: int = 4,
    action_dim: int = 2,
    is_terminated: bool = True,
    termination_proposals: Tuple[str, ...] = ("imbalance",),
    agent_id: str = "robot_a",
    seed: int = 42,
) -> Episode:
    rng = np.random.RandomState(seed + ep_idx)
    obs = rng.randn(T, obs_dim).astype(np.float32)
    acts = rng.randn(T, action_dim).astype(np.float32)
    final_obs = rng.randn(obs_dim).astype(np.float32)
    if is_terminated:
        records = {aid: tuple((r, 0) for r in termination_proposals) for aid in (agent_id, "robot_b")}
    else:
        records = {aid: (("timeout", 0),) for aid in (agent_id, "robot_b")}
    return Episode(
        base_seed=seed,
        episode_index=ep_idx,
        blueprint_hash="test_hash",
        num_frames=T,
        agent_termination_proposal_records=records,
        episode_options={"agent_id": agent_id},
        observations={agent_id: obs},
        actions={agent_id: acts},
        action_extras={agent_id: {}},
        observer_outputs={},
        final_observation={agent_id: final_obs},
    )


# ---------------------------------------------------------------------------
# Mock critic — deterministic linear V(s) = w·s + b
# ---------------------------------------------------------------------------

class MockCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        w = rng.randn(obs_dim, 1).astype(np.float32) * 0.1
        b = rng.randn(1).astype(np.float32) * 0.01
        self.register_parameter(
            "w", torch.nn.Parameter(torch.as_tensor(w))
        )
        self.register_parameter(
            "b", torch.nn.Parameter(torch.as_tensor(b))
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs @ self.w + self.b


# ---------------------------------------------------------------------------
# Golden capture
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden_data"


def capture_buffer_state(buf: PPOBuffer, reward_keys: Tuple[str, ...]) -> Dict[str, Any]:
    """Capture all internal buffer state for golden comparison."""
    state: Dict[str, Any] = {
        "ep_lengths": list(buf.ep_lengths),
        "episode_lengths": list(buf.episode_lengths),
        "sample_weights": buf.sample_weights.tolist() if buf.sample_weights.size else [],
        "frame_modes": buf.frame_modes.tolist() if buf.frame_modes is not None else None,
        "obs_shape": list(buf.obs.shape),
        "actions_shape": list(buf.actions.shape),
        "log_probs_shape": list(buf.log_probs.shape),
    }
    # final_obs list
    state["final_obs"] = [
        fo.tolist() if fo is not None else None
        for fo in buf.final_obs
    ]
    # Per-key state
    for key in reward_keys:
        state[f"key_seg_active__{key}"] = list(buf.key_seg_active[key])
        state[f"key_seg_terminated__{key}"] = list(buf.key_seg_terminated[key])
        state[f"key_seg_actor_weight__{key}"] = list(buf.key_seg_actor_weight[key])
        # reward_data: list of arrays per segment
        rd = buf.reward_data[key]
        state[f"reward_data__{key}"] = [arr.tolist() for arr in rd]

    return state


def capture_gae_state(
    buf: PPOBuffer,
    reward_keys: Tuple[str, ...],
    gammas: Dict[str, float],
    gae_lambdas: Dict[str, float],
    critics: Dict[str, torch.nn.Module],
    device: torch.device,
    stage_weights: Tuple[float, ...],
) -> Dict[str, Any]:
    """Replicate the GAE computation from ppo_update and capture all intermediates."""
    obs_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)

    # Compute values for each critic
    values_all: Dict[str, np.ndarray] = {}
    for key, critic in critics.items():
        with torch.no_grad():
            values_all[key] = critic(obs_t).reshape(-1).cpu().numpy().astype(np.float32)

    # Bootstrap collection (mirrors ppo_trainer.py logic)
    bootstrap_indices: List[int] = []
    bootstrap_obs: List[np.ndarray] = []
    for i, T in enumerate(buf.ep_lengths):
        needs_boot = any(
            buf.key_seg_active[key][i] and not buf.key_seg_terminated[key][i]
            for key in reward_keys
        )
        if needs_boot and buf.final_obs[i] is not None:
            bootstrap_indices.append(i)
            bootstrap_obs.append(np.asarray(buf.final_obs[i], dtype=np.float32))

    bootstrap_values: Dict[str, np.ndarray] = {}
    bootstrap_pos: Dict[int, int] = {}
    if bootstrap_obs:
        boot_t = torch.as_tensor(np.stack(bootstrap_obs), dtype=torch.float32, device=device)
        for key, critic in critics.items():
            with torch.no_grad():
                bootstrap_values[key] = critic(boot_t).reshape(-1).cpu().numpy().astype(np.float32)
        bootstrap_pos = {ep_idx: pos for pos, ep_idx in enumerate(bootstrap_indices)}

    # Segment offsets
    seg_offsets: List[int] = []
    _off = 0
    for T in buf.ep_lengths:
        seg_offsets.append(_off)
        _off += T

    # Key frame mask
    n = sum(buf.ep_lengths)
    key_frame_mask: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        mask = np.zeros(n, dtype=bool)
        for i, is_active in enumerate(buf.key_seg_active[key]):
            if is_active:
                s = seg_offsets[i]
                e = s + buf.ep_lengths[i]
                mask[s:e] = True
        key_frame_mask[key] = mask

    # GAE per key
    advs_all: Dict[str, np.ndarray] = {}
    rets_all: Dict[str, np.ndarray] = {}
    per_key_last_values: Dict[str, List[float]] = {}

    for key in reward_keys:
        advs_list = []
        rets_list = []
        last_values = []
        for i, T in enumerate(buf.ep_lengths):
            s = seg_offsets[i]
            values = values_all[key][s: s + T]

            if not buf.key_seg_active[key][i]:
                advs_list.append(np.zeros(T, dtype=np.float32))
                rets_list.append(np.zeros(T, dtype=np.float32))
                last_values.append(0.0)
                continue

            last_value = 0.0
            key_terminated = buf.key_seg_terminated[key][i]
            if not key_terminated and buf.final_obs[i] is not None and i in bootstrap_pos:
                last_value = float(bootstrap_values[key][bootstrap_pos[i]])
            last_values.append(last_value)

            rewards = buf.reward_data[key][i]
            adv, ret = compute_gae(
                rewards=rewards, values=values, last_value=last_value,
                gamma=gammas[key], lam=gae_lambdas[key],
            )
            advs_list.append(adv)
            rets_list.append(ret)

        advs_all[key] = np.concatenate(advs_list)
        rets_all[key] = np.concatenate(rets_list)
        per_key_last_values[key] = last_values

    # Explained variance
    explained_variances: Dict[str, float] = {}
    for key in reward_keys:
        mask = key_frame_mask[key]
        y_true = rets_all[key][mask]
        y_pred = values_all[key][mask]
        var_y = np.var(y_true) if y_true.size > 0 else 0.0
        if var_y < 1e-8:
            ev = 0.0
        else:
            ev = float(1.0 - np.var(y_true - y_pred) / var_y)
        explained_variances[f"ev_{key}"] = ev

    # Confidence
    confidences: Dict[str, float] = {}
    for key in reward_keys:
        ev = explained_variances.get(f"ev_{key}", 0.0)
        confidences[key] = float(np.clip(ev, 0.0, 1.0) ** 0.5)

    # Combined advantage (default path, no experiment override)
    def _normalize_adv(adv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        active = adv[mask]
        if active.size == 0:
            return np.zeros_like(adv, dtype=np.float32)
        mean = float(active.mean())
        std = float(active.std())
        if std < 1e-8:
            return np.zeros_like(adv, dtype=np.float32)
        result = np.zeros_like(adv, dtype=np.float32)
        result[mask] = ((active - mean) / std).astype(np.float32)
        return result

    # Build per-key per-frame actor_weight from buffer
    key_actor_weight_frame: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        aw_frame = np.zeros(n, dtype=np.float32)
        for i, is_active in enumerate(buf.key_seg_active[key]):
            if is_active:
                s = seg_offsets[i]
                e = s + buf.ep_lengths[i]
                aw_frame[s:e] = buf.key_seg_actor_weight[key][i]
        key_actor_weight_frame[key] = aw_frame

    combined_adv = np.zeros(n, dtype=np.float32)
    for key in reward_keys:
        aw_frame = key_actor_weight_frame[key]
        if not np.any(aw_frame > 0.0):
            continue
        conf = confidences[key]
        combined_adv = combined_adv + aw_frame * conf * _normalize_adv(
            advs_all[key], key_frame_mask[key],
        )

    # Capture
    state: Dict[str, Any] = {
        "bootstrap_indices": bootstrap_indices,
        "bootstrap_pos": {str(k): v for k, v in bootstrap_pos.items()},
        "n_total": n,
        "seg_offsets": seg_offsets,
    }
    for key in reward_keys:
        state[f"values__{key}"] = values_all[key].tolist()
        state[f"advs__{key}"] = advs_all[key].tolist()
        state[f"rets__{key}"] = rets_all[key].tolist()
        state[f"key_frame_mask__{key}"] = key_frame_mask[key].tolist()
        state[f"last_values__{key}"] = per_key_last_values[key]
        state[f"ev__{key}"] = explained_variances[f"ev_{key}"]
        state[f"confidence__{key}"] = confidences[key]
        if key in bootstrap_values:
            state[f"bootstrap_values__{key}"] = bootstrap_values[key].tolist()

    state["combined_adv"] = combined_adv.tolist()

    return state


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

REWARD_KEYS = ("r_a", "r_b")
GAMMAS = {"r_a": 0.99, "r_b": 0.95}
GAE_LAMBDA = 0.95
GAE_LAMBDAS = {"r_a": 0.95, "r_b": 0.95}
OBS_DIM = 4
ACTION_DIM = 2


def _build_buffer_and_capture(
    episodes: List[Episode],
    experiment: MockExperiment,
    stage_weights: Tuple[float, ...],
    label: str,
) -> Dict[str, Any]:
    """Build PPOBuffer, run GAE, capture everything."""
    cp = experiment.common_params()
    device = torch.device("cpu")

    # Convert episodes to trajectories via legacy converter
    from baseline.framework.trajectory import legacy_to_trajectories
    all_trajs = []
    ep_metrics = []
    ep_lengths = []
    for ep in episodes:
        ep_metrics.append(experiment.compute_episode_metrics(ep))
        ep_lengths.append(ep.num_frames)
        trajs = legacy_to_trajectories(
            experiment, ep, REWARD_KEYS, GAMMAS, stage_weights,
        )
        all_trajs.extend(trajs)

    buf = PPOBuffer(
        trajectories=all_trajs,
        actor=MockActor(),
        device=device,
        reward_keys=REWARD_KEYS,
        episode_metrics=ep_metrics,
        episode_lengths=ep_lengths,
    )

    # Build mock critics
    critics: Dict[str, torch.nn.Module] = {}
    for i, key in enumerate(REWARD_KEYS):
        critics[key] = MockCritic(OBS_DIM, seed=100 + i)

    buffer_state = capture_buffer_state(buf, REWARD_KEYS)
    gae_state = capture_gae_state(
        buf, REWARD_KEYS, GAMMAS, GAE_LAMBDAS, critics, device, stage_weights,
    )

    return {
        "label": label,
        "buffer": buffer_state,
        "gae": gae_state,
    }


# We need to handle the PPOBuffer constructor: it calls actor.evaluate_actions
# in Phase 2. We need a mock actor.

class MockActor:
    """Mock actor that returns deterministic log_probs."""
    log_std = torch.tensor(0.0)
    log_std_min = -4.0
    log_std_max = 0.0

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
        *, frame_modes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = obs.shape[0]
        lp = torch.zeros(T, dtype=torch.float32)
        ent = torch.zeros(T, dtype=torch.float32)
        return lp, ent

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = obs.shape[0]
        return torch.zeros(T, ACTION_DIM), torch.zeros(T)

    def to_blueprint(self, dest_path: str, *, stochastic: bool = False):
        raise NotImplementedError

    def parameters(self):
        return iter([])

    def train(self, mode: bool = True):
        return self

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# Scenario builders — return (result, structural_assert_fn)
# ---------------------------------------------------------------------------

def _build_s1() -> Dict[str, Any]:
    ep = make_episode(ep_idx=0, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1")
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s1_single_terminated")

def _build_s2() -> Dict[str, Any]:
    ep = make_episode(ep_idx=1, T=12, is_terminated=False,
                      termination_proposals=())
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1")
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s2_single_truncated")

def _build_s3() -> Dict[str, Any]:
    ep = make_episode(ep_idx=2, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    segs = [Segment(start=0, end=5, weight=1.0), Segment(start=5, end=10, weight=1.0)]
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={2: segs})
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s3_multi_segment_boundary")

def _build_s4() -> Dict[str, Any]:
    ep = make_episode(ep_idx=3, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    segs = [
        Segment(start=0, end=5, weight=1.0, key_weights={"r_a": 1.0}),
        Segment(start=5, end=10, weight=1.0, key_weights={"r_b": 1.0}),
    ]
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={3: segs})
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s4_key_weights")

def _build_s5() -> Dict[str, Any]:
    ep = make_episode(ep_idx=4, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    segs = [
        Segment(start=0, end=5, weight=1.0,
                key_weights={"r_a": 1.0, "r_b": 1.0},
                termination="terminated",
                key_termination={"r_a": "terminated", "r_b": "truncated"}),
        Segment(start=5, end=10, weight=1.0,
                key_weights={"r_a": 1.0, "r_b": 1.0},
                termination="terminated"),
    ]
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={4: segs})
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s5_key_termination")

def _build_s6() -> Dict[str, Any]:
    ep = make_episode(ep_idx=5, T=8, is_terminated=True,
                      termination_proposals=("imbalance",))
    segs = [Segment(start=0, end=4, weight=1.0, mode=1.0),
            Segment(start=4, end=8, weight=1.0, mode=2.0)]
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={5: segs})
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s6_segment_mode")

def _build_s7() -> Dict[str, Any]:
    ep = make_episode(ep_idx=6, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    segs = [Segment(start=0, end=10, weight=2.5)]
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={6: segs})
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s7_segment_weight")

def _build_s8() -> Dict[str, Any]:
    ep0 = make_episode(ep_idx=7, T=8, is_terminated=True,
                       termination_proposals=("imbalance",))
    ep1 = make_episode(ep_idx=8, T=12, is_terminated=False,
                       termination_proposals=())
    ep2 = make_episode(ep_idx=9, T=6, is_terminated=True,
                       termination_proposals=("imbalance",))
    segs2 = [
        Segment(start=0, end=3, weight=1.0,
                key_weights={"r_a": 1.0},
                key_termination={"r_a": "truncated"}),
        Segment(start=3, end=6, weight=1.0,
                key_weights={"r_a": 1.0, "r_b": 1.0}),
    ]
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={9: segs2})
    return _build_buffer_and_capture([ep0, ep1, ep2], exp, (1.0, 0.5), "s8_multi_episode_mixed")

def _build_s9() -> Dict[str, Any]:
    ep = make_episode(ep_idx=10, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1",
                         v1_segments_override={10: [(0, 5, 1.0), (5, 10, 1.0)]})
    return _build_buffer_and_capture([ep], exp, (1.0, 1.0), "s9_v1_fallback")

def _build_s10() -> Dict[str, Any]:
    ep = make_episode(ep_idx=11, T=10, is_terminated=True,
                      termination_proposals=("imbalance",))
    exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1")
    return _build_buffer_and_capture([ep], exp, (1.0, 0.0), "s10_stage_weight_zero")


SCENARIO_BUILDERS: List[Tuple[str, callable, callable]] = [
    ("s1_single_terminated", _build_s1, lambda r: (
        r["buffer"]["ep_lengths"] == [10] and
        r["gae"]["bootstrap_indices"] == [] and
        all(all(v == 0.0 for v in r["gae"][f"last_values__{k}"]) for k in REWARD_KEYS)
    )),
    ("s2_single_truncated", _build_s2, lambda r: (
        r["gae"]["bootstrap_indices"] == [0] and
        all(r["gae"][f"last_values__{k}"][0] != 0.0 for k in REWARD_KEYS)
    )),
    ("s3_multi_segment_boundary", _build_s3, lambda r: (
        r["buffer"]["ep_lengths"] == [5, 5]
    )),
    ("s4_key_weights", _build_s4, lambda r: (
        r["buffer"]["key_seg_active__r_a"] == [True, False] and
        r["buffer"]["key_seg_active__r_b"] == [False, True]
    )),
    ("s5_key_termination", _build_s5, lambda r: (
        r["buffer"]["key_seg_terminated__r_a"] == [True, True] and
        r["buffer"]["key_seg_terminated__r_b"] == [False, True] and
        0 in r["gae"]["bootstrap_indices"] and
        r["gae"]["last_values__r_b"][0] != 0.0 and
        r["gae"]["last_values__r_a"][0] == 0.0
    )),
    ("s6_segment_mode", _build_s6, lambda r: (
        r["buffer"]["frame_modes"] is not None and
        r["buffer"]["frame_modes"][:4] == [1.0, 1.0, 1.0, 1.0] and
        r["buffer"]["frame_modes"][4:] == [2.0, 2.0, 2.0, 2.0]
    )),
    ("s7_segment_weight", _build_s7, lambda r: (
        all(w == 2.5 for w in r["buffer"]["sample_weights"])
    )),
    ("s8_multi_episode_mixed", _build_s8, lambda r: (
        len(r["buffer"]["ep_lengths"]) == 4
    )),
    ("s9_v1_fallback", _build_s9, lambda r: (
        len(r["buffer"]["ep_lengths"]) == 2
    )),
    ("s10_stage_weight_zero", _build_s10, lambda r: (
        any(v != 0.0 for v in r["gae"]["advs__r_b"])
    )),
]


class TestGoldenCapture:
    """Capture golden state for all scenarios and save to disk."""

    @pytest.mark.parametrize("label,builder,assert_fn", SCENARIO_BUILDERS)
    def test_capture_and_structure(self, label, builder, assert_fn):
        result = builder()
        assert assert_fn(result), f"Structural assertion failed for {label}"
        _save_golden(result)


# ---------------------------------------------------------------------------
# Golden save / load
# ---------------------------------------------------------------------------

def _save_golden(result: Dict[str, Any]) -> None:
    """Save golden state to disk (only if GOLDEN_UPDATE=1 env var is set)."""
    if os.environ.get("GOLDEN_UPDATE", "0") != "1":
        return
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    label = result["label"]
    path = GOLDEN_DIR / f"{label}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)


def _load_golden(label: str) -> Optional[Dict[str, Any]]:
    path = GOLDEN_DIR / f"{label}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Golden comparison tests (run when golden data exists)
# ---------------------------------------------------------------------------

class TestGoldenCompare:
    """Compare current output against saved golden data, bit-for-bit."""

    @pytest.mark.parametrize("label,builder,assert_fn", SCENARIO_BUILDERS)
    def test_golden_match(self, label, builder, assert_fn):
        golden = _load_golden(label)
        if golden is None:
            pytest.skip(f"No golden data for {label}. Run with GOLDEN_UPDATE=1 to generate.")

        result = builder()

        # Structural assertion
        assert assert_fn(result), f"Structural assertion failed for {label}"

        # Bit-for-bit comparison of all captured fields
        _assert_deep_equal(result["buffer"], golden["buffer"], f"{label}/buffer")
        _assert_deep_equal(result["gae"], golden["gae"], f"{label}/gae")


def _assert_deep_equal(actual: Any, expected: Any, path: str, rtol: float = 0.0) -> None:
    """Deep comparison of nested structures. Floats compared exactly (rtol=0)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys()), \
            f"{path}: key mismatch {set(actual.keys())} vs {set(expected.keys())}"
        for k in expected:
            _assert_deep_equal(actual[k], expected[k], f"{path}/{k}", rtol)
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(actual) == len(expected), f"{path}: len {len(actual)} vs {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_deep_equal(a, e, f"{path}[{i}]", rtol)
    elif isinstance(expected, (int, bool)):
        assert actual == expected, f"{path}: {actual} != {expected}"
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)), f"{path}: type {type(actual)} vs float"
        assert abs(actual - expected) < 1e-9, f"{path}: {actual} != {expected}"
    elif expected is None:
        assert actual is None, f"{path}: {actual} is not None"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
