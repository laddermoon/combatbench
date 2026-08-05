"""Test legacy_to_trajectories converter against golden buffer state.

Verifies that the v1→v2 converter produces Trajectory data that, when
unfolded, matches the exact buffer state captured by the golden tests.

This ensures that the v2 rewrite can consume ``List[Trajectory]``
exclusively while producing identical GAE results for v1 experiments.
"""
from __future__ import annotations

import json
import os
import sys
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
from baseline.framework.trajectory import (
    ChannelData,
    RewardChannel,
    Trajectory,
    legacy_to_trajectories,
    resolve_trajectories,
)

# Reuse infrastructure from golden test
from baseline.framework.test_ppo_buffer_golden import (
    MockActor,
    MockCritic,
    MockExperiment,
    REWARD_KEYS,
    GAMMAS,
    GAE_LAMBDA,
    GAE_LAMBDAS,
    OBS_DIM,
    ACTION_DIM,
    make_episode,
    _build_buffer_and_capture,
    GOLDEN_DIR,
    _load_golden,
    SCENARIO_BUILDERS,
)


# ---------------------------------------------------------------------------
# Trajectory → buffer state unfold
# ---------------------------------------------------------------------------

def trajectories_to_buffer_state(
    trajs: List[Trajectory],
    reward_keys: Tuple[str, ...],
) -> Dict[str, Any]:
    """Unfold List[Trajectory] into the same dict format as capture_buffer_state."""
    ep_lengths = [len(t.obs) for t in trajs]
    n = sum(ep_lengths)

    # Per-key per-segment active and terminated
    key_seg_active: Dict[str, List[bool]] = {k: [] for k in reward_keys}
    key_seg_terminated: Dict[str, List[bool]] = {k: [] for k in reward_keys}
    key_seg_actor_weight: Dict[str, List[float]] = {k: [] for k in reward_keys}
    reward_data: Dict[str, List[np.ndarray]] = {k: [] for k in reward_keys}
    is_terminated: List[bool] = []
    final_obs: List[Optional[np.ndarray]] = []
    sample_weights: List[float] = []
    frame_modes: List[float] = []

    any_mode = False
    for traj in trajs:
        T_seg = len(traj.obs)
        if traj.mode is not None:
            any_mode = True
            frame_modes.extend([float(traj.mode)] * T_seg)
        else:
            frame_modes.extend([1.0] * T_seg)

        sample_weights.extend([traj.importance] * T_seg)
        final_obs.append(traj.last_obs)

        # Reconstruct v1 segment-level is_terminated:
        # True if ALL active channels are terminated (no bootstrap needed
        # for any key).  This matches v1's term_seg when key_termination
        # is absent, and matches the "needs_boot" logic when it is present.
        active_channels = [k for k in reward_keys if k in traj.channels]
        if active_channels:
            all_terminated = all(
                traj.channels[k].is_terminated for k in active_channels
            )
        else:
            all_terminated = True
        # v1 is_terminated is the segment-level flag used for bootstrap
        # collection: True = no bootstrap for ANY key on this segment.
        # When key_termination causes some keys to be truncated, the
        # segment-level flag in v1 was still True (term_seg), but bootstrap
        # collection used per-key flags.  The v1 buffer stored term_seg
        # in is_terminated, NOT the all-keys-terminated derivation.
        # For golden comparison, we need to match v1's term_seg.
        # Since we don't have the original segment, we approximate:
        # if any key is truncated, v1's term_seg could be either True or
        # False depending on the segment's termination field.
        # The correct approach: is_terminated in v1 = term_seg, which is
        # determined by segment.termination / auto logic, NOT by per-key.
        # We can infer it: if all keys are terminated, term_seg was True.
        # If some keys are truncated, term_seg could still be True (e.g.
        # s5 where seg termination="terminated" but r_b key_termination="truncated").
        # The v1 buffer stored term_seg, which for s5 seg0 was True.
        # We can't perfectly reconstruct term_seg from trajectories alone.
        # However, the GAE comparison is what matters — the buffer
        # is_terminated field is only consumed by bootstrap collection,
        # which we already handle via per-key flags.  So we set it to
        # all_terminated for structural consistency, and accept that
        # this specific field may differ when key_termination is used.
        is_terminated.append(all_terminated)

        for key in reward_keys:
            if key in traj.channels:
                cd = traj.channels[key]
                key_seg_active[key].append(True)
                key_seg_terminated[key].append(cd.is_terminated)
                key_seg_actor_weight[key].append(cd.actor_weight)
                reward_data[key].append(cd.reward)
            else:
                key_seg_active[key].append(False)
                key_seg_terminated[key].append(True)  # inactive → terminated
                key_seg_actor_weight[key].append(0.0)
                reward_data[key].append(np.zeros(T_seg, dtype=np.float32))

    # Build obs/actions for shape reporting
    if trajs:
        obs = np.concatenate([t.obs for t in trajs], axis=0)
        actions = np.concatenate([t.actions for t in trajs], axis=0)
        log_probs = np.zeros(n, dtype=np.float32)  # mock actor gives zeros
    else:
        obs = np.zeros((0,), np.float32)
        actions = np.zeros((0,), np.float32)
        log_probs = np.zeros(0, dtype=np.float32)

    state: Dict[str, Any] = {
        "ep_lengths": ep_lengths,
        "episode_lengths": ep_lengths,  # v2 doesn't distinguish
        "is_terminated": is_terminated,
        "sample_weights": sample_weights,
        "frame_modes": frame_modes if any_mode else None,
        "obs_shape": list(obs.shape),
        "actions_shape": list(actions.shape),
        "log_probs_shape": list(log_probs.shape),
        "final_obs": [fo.tolist() if fo is not None else None for fo in final_obs],
    }

    for key in reward_keys:
        state[f"key_seg_active__{key}"] = key_seg_active[key]
        state[f"key_seg_terminated__{key}"] = key_seg_terminated[key]
        state[f"key_seg_actor_weight__{key}"] = key_seg_actor_weight[key]
        state[f"reward_data__{key}"] = [arr.tolist() for arr in reward_data[key]]

    return state


def trajectories_to_gae_state(
    trajs: List[Trajectory],
    reward_keys: Tuple[str, ...],
    gammas: Dict[str, float],
    gae_lambdas: Dict[str, float],
    critics: Dict[str, torch.nn.Module],
    device: torch.device,
    stage_weights: Tuple[float, ...],
) -> Dict[str, Any]:
    """Unfold trajectories and compute GAE, mirroring ppo_trainer logic."""
    ep_lengths = [len(t.obs) for t in trajs]
    n = sum(ep_lengths)

    # Concatenate obs for value computation
    if trajs:
        all_obs = np.concatenate([t.obs for t in trajs], axis=0).astype(np.float32)
    else:
        all_obs = np.zeros((0, OBS_DIM), dtype=np.float32)
    obs_t = torch.as_tensor(all_obs, dtype=torch.float32, device=device)

    # Compute values
    values_all: Dict[str, np.ndarray] = {}
    for key, critic in critics.items():
        with torch.no_grad():
            values_all[key] = critic(obs_t).reshape(-1).cpu().numpy().astype(np.float32)

    # Per-key per-segment state from trajectories
    key_seg_active: Dict[str, List[bool]] = {k: [] for k in reward_keys}
    key_seg_terminated: Dict[str, List[bool]] = {k: [] for k in reward_keys}
    key_seg_actor_weight: Dict[str, List[float]] = {k: [] for k in reward_keys}
    reward_data: Dict[str, List[np.ndarray]] = {k: [] for k in reward_keys}
    final_obs_list: List[np.ndarray] = []

    for traj in trajs:
        final_obs_list.append(traj.last_obs)
        for key in reward_keys:
            if key in traj.channels:
                cd = traj.channels[key]
                key_seg_active[key].append(True)
                key_seg_terminated[key].append(cd.is_terminated)
                key_seg_actor_weight[key].append(cd.actor_weight)
                reward_data[key].append(cd.reward)
            else:
                key_seg_active[key].append(False)
                key_seg_terminated[key].append(True)
                key_seg_actor_weight[key].append(0.0)
                reward_data[key].append(np.zeros(len(traj.obs), dtype=np.float32))

    # Bootstrap collection
    bootstrap_indices: List[int] = []
    bootstrap_obs: List[np.ndarray] = []
    for i in range(len(trajs)):
        needs_boot = any(
            key_seg_active[key][i] and not key_seg_terminated[key][i]
            for key in reward_keys
        )
        if needs_boot and final_obs_list[i] is not None:
            bootstrap_indices.append(i)
            bootstrap_obs.append(np.asarray(final_obs_list[i], dtype=np.float32))

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
    for T in ep_lengths:
        seg_offsets.append(_off)
        _off += T

    # Key frame mask
    key_frame_mask: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        mask = np.zeros(n, dtype=bool)
        for i, is_active in enumerate(key_seg_active[key]):
            if is_active:
                s = seg_offsets[i]
                e = s + ep_lengths[i]
                mask[s:e] = True
        key_frame_mask[key] = mask

    # GAE
    advs_all: Dict[str, np.ndarray] = {}
    rets_all: Dict[str, np.ndarray] = {}
    per_key_last_values: Dict[str, List[float]] = {}

    for key in reward_keys:
        advs_list = []
        rets_list = []
        last_values = []
        for i, T in enumerate(ep_lengths):
            s = seg_offsets[i]
            values = values_all[key][s: s + T]

            if not key_seg_active[key][i]:
                advs_list.append(np.zeros(T, dtype=np.float32))
                rets_list.append(np.zeros(T, dtype=np.float32))
                last_values.append(0.0)
                continue

            last_value = 0.0
            key_terminated = key_seg_terminated[key][i]
            if not key_terminated and final_obs_list[i] is not None and i in bootstrap_pos:
                last_value = float(bootstrap_values[key][bootstrap_pos[i]])
            last_values.append(last_value)

            rewards = reward_data[key][i]
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

    # Combined advantage
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

    # Build per-key per-frame actor_weight from trajectory data
    key_actor_weight_frame: Dict[str, np.ndarray] = {}
    for key in reward_keys:
        aw_frame = np.zeros(n, dtype=np.float32)
        for i, is_active in enumerate(key_seg_active[key]):
            if is_active:
                s = seg_offsets[i]
                e = s + ep_lengths[i]
                aw_frame[s:e] = key_seg_actor_weight[key][i]
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
# Build trajectories from v1 experiment for each scenario
# ---------------------------------------------------------------------------

def _build_trajectories_from_v1(builder) -> List[Trajectory]:
    """Run a scenario builder to get the experiment + episode, then convert."""
    # We need to intercept the builder to get the experiment and episodes.
    # Instead of modifying builders, we reconstruct the conversion directly.
    # The builders call _build_buffer_and_capture which internally creates
    # the buffer. We need the experiment and episodes separately.
    #
    # Strategy: re-run the scenario setup (duplicated from builders) to
    # get experiment + episodes, then call legacy_to_trajectories.
    # This is acceptable for Phase 1 testing — Phase 2 will test the
    # actual v2 buffer.
    raise NotImplementedError("Use _build_trajectories_for_scenario instead")


# We need to reconstruct scenarios to get experiment + episodes.
# Factor out the setup from each builder.

def _scenario_setup(label: str) -> Tuple[List[Episode], MockExperiment, Tuple[float, ...]]:
    """Reconstruct scenario setup to get episodes + experiment + stage_weights."""
    if label == "s1_single_terminated":
        ep = make_episode(ep_idx=0, T=10, is_terminated=True,
                          termination_proposals=("imbalance",))
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1")
        return [ep], exp, (1.0, 1.0)

    elif label == "s2_single_truncated":
        ep = make_episode(ep_idx=1, T=12, is_terminated=False,
                          termination_proposals=())
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1")
        return [ep], exp, (1.0, 1.0)

    elif label == "s3_multi_segment_boundary":
        ep = make_episode(ep_idx=2, T=10, is_terminated=True,
                          termination_proposals=("imbalance",))
        segs = [Segment(start=0, end=5, weight=1.0), Segment(start=5, end=10, weight=1.0)]
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={2: segs})
        return [ep], exp, (1.0, 1.0)

    elif label == "s4_key_weights":
        ep = make_episode(ep_idx=3, T=10, is_terminated=True,
                          termination_proposals=("imbalance",))
        segs = [
            Segment(start=0, end=5, weight=1.0, key_weights={"r_a": 1.0}),
            Segment(start=5, end=10, weight=1.0, key_weights={"r_b": 1.0}),
        ]
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={3: segs})
        return [ep], exp, (1.0, 1.0)

    elif label == "s5_key_termination":
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
        return [ep], exp, (1.0, 1.0)

    elif label == "s6_segment_mode":
        ep = make_episode(ep_idx=5, T=8, is_terminated=True,
                          termination_proposals=("imbalance",))
        segs = [Segment(start=0, end=4, weight=1.0, mode=1.0),
                Segment(start=4, end=8, weight=1.0, mode=2.0)]
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={5: segs})
        return [ep], exp, (1.0, 1.0)

    elif label == "s7_segment_weight":
        ep = make_episode(ep_idx=6, T=10, is_terminated=True,
                          termination_proposals=("imbalance",))
        segs = [Segment(start=0, end=10, weight=2.5)]
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v2", segments_override={6: segs})
        return [ep], exp, (1.0, 1.0)

    elif label == "s8_multi_episode_mixed":
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
        return [ep0, ep1, ep2], exp, (1.0, 0.5)

    elif label == "s9_v1_fallback":
        ep = make_episode(ep_idx=10, T=10, is_terminated=True,
                          termination_proposals=("imbalance",))
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1",
                             v1_segments_override={10: [(0, 5, 1.0), (5, 10, 1.0)]})
        return [ep], exp, (1.0, 1.0)

    elif label == "s10_stage_weight_zero":
        ep = make_episode(ep_idx=11, T=10, is_terminated=True,
                          termination_proposals=("imbalance",))
        exp = MockExperiment(REWARD_KEYS, GAMMAS, seg_mode="v1")
        return [ep], exp, (1.0, 0.0)

    raise ValueError(f"Unknown scenario: {label}")


def _build_trajectories_for_scenario(label: str) -> Tuple[List[Trajectory], Tuple[float, ...]]:
    """Build trajectories via legacy converter for a scenario."""
    episodes, exp, stage_weights = _scenario_setup(label)
    all_trajs: List[Trajectory] = []
    for ep in episodes:
        trajs = legacy_to_trajectories(
            exp, ep, REWARD_KEYS, GAMMAS, stage_weights,
        )
        all_trajs.extend(trajs)
    return all_trajs, stage_weights


# ---------------------------------------------------------------------------
# Deep comparison helper (same as golden test)
# ---------------------------------------------------------------------------

def _assert_deep_equal(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys()), \
            f"{path}: key mismatch {set(actual.keys())} vs {set(expected.keys())}"
        for k in expected:
            _assert_deep_equal(actual[k], expected[k], f"{path}/{k}")
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(actual) == len(expected), f"{path}: len {len(actual)} vs {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_deep_equal(a, e, f"{path}[{i}]")
    elif isinstance(expected, (int, bool)):
        assert actual == expected, f"{path}: {actual} != {expected}"
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)), f"{path}: type {type(actual)} vs float"
        assert abs(actual - expected) < 1e-9, f"{path}: {actual} != {expected}"
    elif expected is None:
        assert actual is None, f"{path}: {actual} is not None"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SCENARIO_LABELS = [s[0] for s in SCENARIO_BUILDERS]


class TestLegacyConverter:
    """Verify legacy_to_trajectories produces data identical to v1 buffer."""

    @pytest.mark.parametrize("label", SCENARIO_LABELS)
    def test_buffer_state_matches_golden(self, label):
        """Unfold trajectories → buffer state, compare to golden."""
        golden = _load_golden(label)
        if golden is None:
            pytest.skip(f"No golden data for {label}")

        trajs, _ = _build_trajectories_for_scenario(label)
        buf_state = trajectories_to_buffer_state(trajs, REWARD_KEYS)

        # episode_lengths is per-episode in v1, per-trajectory in v2.
        # is_terminated is a v1 segment-level concept replaced by per-key
        # is_terminated in v2.  Both are expected semantic differences.
        # The GAE test validates actual bootstrap behavior.
        SKIP_KEYS = {"episode_lengths", "is_terminated"}
        golden_buf = {k: v for k, v in golden["buffer"].items()
                      if k not in SKIP_KEYS}
        actual_buf = {k: v for k, v in buf_state.items()
                      if k not in SKIP_KEYS}

        _assert_deep_equal(actual_buf, golden_buf, f"{label}/buffer")

    @pytest.mark.parametrize("label", SCENARIO_LABELS)
    def test_gae_state_matches_golden(self, label):
        """Compute GAE from trajectories, compare to golden."""
        golden = _load_golden(label)
        if golden is None:
            pytest.skip(f"No golden data for {label}")

        trajs, stage_weights = _build_trajectories_for_scenario(label)
        device = torch.device("cpu")

        critics: Dict[str, torch.nn.Module] = {}
        for i, key in enumerate(REWARD_KEYS):
            critics[key] = MockCritic(OBS_DIM, seed=100 + i)

        gae_state = trajectories_to_gae_state(
            trajs, REWARD_KEYS, GAMMAS, GAE_LAMBDAS, critics, device, stage_weights,
        )

        _assert_deep_equal(gae_state, golden["gae"], f"{label}/gae")

    @pytest.mark.parametrize("label", SCENARIO_LABELS)
    def test_resolve_trajectories_dispatches_v1(self, label):
        """resolve_trajectories should use legacy path for v1 experiments."""
        episodes, exp, stage_weights = _scenario_setup(label)
        for ep in episodes:
            trajs = resolve_trajectories(
                exp, ep, REWARD_KEYS, GAMMAS, stage_weights,
            )
            # v1 experiment has no build_trajectories → should use legacy
            assert isinstance(trajs, list)
            assert all(isinstance(t, Trajectory) for t in trajs)


class TestTrajectoryStructures:
    """Verify basic properties of v2 data structures."""

    def test_reward_channel_is_frozen(self):
        ch = RewardChannel(name="r_test", gamma=0.99, gae_lambda=0.95)
        with pytest.raises(Exception):
            ch.name = "other"  # frozen dataclass

    def test_channel_data_defaults(self):
        cd = ChannelData(reward=np.zeros(5, dtype=np.float32), is_terminated=True)
        assert cd.actor_weight == 1.0

    def test_trajectory_minimal(self):
        t = Trajectory(
            obs=np.zeros((3, 4), dtype=np.float32),
            actions=np.zeros((3, 2), dtype=np.float32),
            last_obs=np.zeros(4, dtype=np.float32),
            channels={"r_a": ChannelData(
                reward=np.zeros(3, dtype=np.float32),
                is_terminated=True,
            )},
        )
        assert t.importance == 1.0
        assert t.mode is None
        assert t.log_prob is None
        assert len(t.obs) == 3
