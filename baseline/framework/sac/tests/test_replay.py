"""Unit tests for TaggedReplay buffer.

Tests:
- Trajectory slice → transition flattening correctness.
- Per-channel done semantics (terminated vs truncated).
- n-step return computation (reward accumulation, bootstrap, done truncation).
- Buffer capacity and circular overwrite.
- Buffer stats.
"""
from __future__ import annotations

import numpy as np
import torch

from baseline.framework.sac.replay import TaggedReplay
from baseline.framework.sac.experiment import TrajectorySlice


def _make_slice(
    T: int = 10,
    obs_dim: int = 4,
    action_dim: int = 2,
    channels: tuple = ("r_a", "r_b"),
    fell: bool = False,
    reward_a: float = 1.0,
    reward_b: float = 0.5,
    aw_a: float = 1.0,
    aw_b: float = 1.0,
) -> TrajectorySlice:
    """Create a simple test slice."""
    obs = np.random.randn(T, obs_dim).astype(np.float32)
    actions = np.random.randn(T, action_dim).astype(np.float32)
    last_obs = np.random.randn(obs_dim).astype(np.float32)

    rewards = {
        ch: np.full(T, reward_a if ch == "r_a" else reward_b, dtype=np.float32)
        for ch in channels
    }
    dones = {
        ch: np.zeros(T, dtype=bool) for ch in channels
    }
    if fell and T > 0:
        for ch in channels:
            dones[ch][-1] = True

    actor_weights = {
        ch: np.full(T, aw_a if ch == "r_a" else aw_b, dtype=np.float32)
        for ch in channels
    }

    return TrajectorySlice(
        obs=obs, actions=actions, last_obs=last_obs,
        rewards=rewards, dones=dones, actor_weights=actor_weights,
    )


def test_basic_insertion_and_sampling():
    """Test that slices are correctly flattened into transitions."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a", "r_b"),
    )
    assert buf.size == 0

    slice1 = _make_slice(T=10)
    n = buf.add_slices([slice1])
    assert n == 10
    assert buf.size == 10

    # Sample
    batch = buf.sample(5, torch.device("cpu"))
    assert batch["obs"].shape == (5, 4)
    assert batch["actions"].shape == (5, 2)
    assert batch["next_obs"].shape == (5, 4)
    assert batch["rewards_r_a"].shape == (5,)
    assert batch["rewards_r_b"].shape == (5,)
    assert batch["dones_r_a"].shape == (5,)
    assert batch["actor_weights_r_a"].shape == (5,)

    print("test_basic_insertion_and_sampling: PASS")


def test_next_obs_correctness():
    """Test that next_obs[t] = obs[t+1] for t < T-1, last_obs for t = T-1."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a",),
    )

    T = 5
    obs = np.arange(T * 4, dtype=np.float32).reshape(T, 4)
    actions = np.zeros((T, 2), dtype=np.float32)
    last_obs = np.full(4, 999.0, dtype=np.float32)

    slc = TrajectorySlice(
        obs=obs, actions=actions, last_obs=last_obs,
        rewards={"r_a": np.ones(T, dtype=np.float32)},
        dones={"r_a": np.zeros(T, dtype=bool)},
        actor_weights={"r_a": np.ones(T, dtype=np.float32)},
    )
    buf.add_slices([slc])

    # Check next_obs for each transition
    for t in range(T):
        if t < T - 1:
            expected = obs[t + 1]
        else:
            expected = last_obs
        actual = buf.next_obs[t]
        assert np.allclose(actual, expected), (
            f"next_obs[{t}] mismatch: expected {expected}, got {actual}"
        )

    print("test_next_obs_correctness: PASS")


def test_per_channel_done():
    """Test that per-channel done flags are stored correctly."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a", "r_b"),
    )

    T = 5
    slc = _make_slice(T=T, fell=True)
    buf.add_slices([slc])

    # Last step should be done for both channels
    assert buf.dones["r_a"][T - 1] == True
    assert buf.dones["r_b"][T - 1] == True
    # Other steps should not be done
    for t in range(T - 1):
        assert buf.dones["r_a"][t] == False
        assert buf.dones["r_b"][t] == False

    print("test_per_channel_done: PASS")


def test_truncated_no_done():
    """Test that truncated episodes have done=False at all steps."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a",),
    )

    T = 5
    slc = _make_slice(T=T, fell=False, channels=("r_a",))
    buf.add_slices([slc])

    for t in range(T):
        assert buf.dones["r_a"][t] == False

    print("test_truncated_no_done: PASS")


def test_nstep_sampling():
    """Test n-step return computation info."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a",),
    )

    T = 10
    obs = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        obs[t, 0] = float(t)
    actions = np.zeros((T, 2), dtype=np.float32)
    last_obs = np.full(4, 99.0, dtype=np.float32)

    # Reward = 1.0 per step
    slc = TrajectorySlice(
        obs=obs, actions=actions, last_obs=last_obs,
        rewards={"r_a": np.ones(T, dtype=np.float32)},
        dones={"r_a": np.zeros(T, dtype=bool)},
        actor_weights={"r_a": np.ones(T, dtype=np.float32)},
    )
    buf.add_slices([slc])

    # Sample with n_step=3
    batch = buf.sample_nstep(
        batch_size=1, device=torch.device("cpu"),
        n_steps={"r_a": 3},
    )

    # Check rewards shape
    assert batch["rewards_r_a"].shape == (1, 3)
    # Check valid_steps: for t=0, remaining=10, n=3, so valid=3
    # (but we sample randomly, so just check it's <= 3)
    vs = batch["valid_steps_r_a"].item()
    assert 1 <= vs <= 3, f"valid_steps should be 1-3, got {vs}"

    print("test_nstep_sampling: PASS")


def test_nstep_done_truncation():
    """Test that n-step rewards are truncated at done step."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a",),
    )

    T = 5
    obs = np.zeros((T, 4), dtype=np.float32)
    actions = np.zeros((T, 2), dtype=np.float32)
    last_obs = np.full(4, 99.0, dtype=np.float32)

    # Reward = 1.0 per step, done at step 2 (0-indexed)
    dones = np.zeros(T, dtype=bool)
    dones[2] = True

    slc = TrajectorySlice(
        obs=obs, actions=actions, last_obs=last_obs,
        rewards={"r_a": np.ones(T, dtype=np.float32)},
        dones={"r_a": dones},
        actor_weights={"r_a": np.ones(T, dtype=np.float32)},
    )
    buf.add_slices([slc])

    # Sample a large batch to ensure we hit all starting positions
    np.random.seed(0)
    batch = buf.sample_nstep(
        batch_size=500, device=torch.device("cpu"),
        n_steps={"r_a": 5},
    )

    indices = batch["indices"].cpu().numpy() if hasattr(batch["indices"], 'cpu') else batch["indices"]
    valid_steps = batch["valid_steps_r_a"].cpu().numpy()
    rewards = batch["rewards_r_a"].cpu().numpy()

    # For each starting position t (0-4), check valid_steps:
    # t=0: steps 0,1,2 → done at 2 → valid=3
    # t=1: steps 1,2 → done at 2 → valid=2
    # t=2: step 2 → done at 2 → valid=1
    # t=3: steps 3,4 → no done → valid=2
    # t=4: step 4 → no done → valid=1
    expected = {0: 3, 1: 2, 2: 1, 3: 2, 4: 1}

    for t, exp_vs in expected.items():
        mask = indices == t
        if not np.any(mask):
            continue
        actual_vs = valid_steps[mask][0]
        assert actual_vs == exp_vs, (
            f"Starting at t={t}: valid_steps={actual_vs}, expected {exp_vs}"
        )
        # Check rewards are 1.0 for valid steps, 0.0 after
        r = rewards[mask][0]
        for k in range(5):
            if k < exp_vs:
                assert r[k] == 1.0, f"t={t} k={k}: reward={r[k]}, expected 1.0"
            else:
                assert r[k] == 0.0, f"t={t} k={k}: reward={r[k]}, expected 0.0"

    print("test_nstep_done_truncation: PASS")


def test_circular_overwrite():
    """Test that buffer overwrites old data when full."""
    buf = TaggedReplay(
        capacity=15, obs_dim=4, action_dim=2,
        channel_names=("r_a",),
    )

    # Add 3 slices of 10 transitions each = 30 total, capacity=15
    for i in range(3):
        slc = _make_slice(T=10, channels=("r_a",), reward_a=float(i + 1))
        buf.add_slices([slc])

    assert buf.size == 15  # capped at capacity

    # The buffer should contain data from slices 1 and 2 (last 15 of 30)
    # Slice 0 (reward=1.0) should be overwritten
    # Slice 1 (reward=2.0) has 5 remaining, Slice 2 (reward=3.0) has 10
    rewards = buf.rewards["r_a"][:15]
    # Check that reward=1.0 is not present (overwritten)
    assert not np.any(rewards == 1.0), "Old data should be overwritten"
    # Check that reward=3.0 is present (most recent)
    assert np.any(rewards == 3.0), "New data should be present"

    print("test_circular_overwrite: PASS")


def test_buffer_stats():
    """Test buffer statistics computation."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a", "r_b"),
        tag_names=("phase",),
    )

    T = 10
    slc = _make_slice(T=T, channels=("r_a", "r_b"))
    slc.tags = {"phase": np.full(T, 1.0, dtype=np.float32)}
    buf.add_slices([slc])

    stats = buf.buffer_stats()
    assert stats["size"] == 10
    assert stats["capacity"] == 100
    assert stats["n_trajectories"] == 1
    assert stats["utilization"] == 0.1

    ch_stats = stats["per_channel"]["r_a"]
    assert ch_stats["reward_mean"] == 1.0
    assert ch_stats["aw_mean"] == 1.0

    tag_stats = stats["tag_stats"]["phase"]
    assert tag_stats["mean"] == 1.0

    print("test_buffer_stats: PASS")


def test_multiple_slices_trajectory_tracking():
    """Test that multiple slices get different traj_ids."""
    buf = TaggedReplay(
        capacity=100, obs_dim=4, action_dim=2,
        channel_names=("r_a",),
    )

    slc1 = _make_slice(T=5, channels=("r_a",))
    slc2 = _make_slice(T=7, channels=("r_a",))
    buf.add_slices([slc1, slc2])

    assert buf.size == 12
    # First 5 transitions should have traj_id=0
    assert all(buf.traj_ids[i] == 0 for i in range(5))
    # Next 7 should have traj_id=1
    assert all(buf.traj_ids[i] == 1 for i in range(5, 12))
    # traj_steps should be 0,1,2,3,4,0,1,2,3,4,5,6
    expected_steps = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]
    for i, expected in enumerate(expected_steps):
        assert buf.traj_steps[i] == expected, (
            f"traj_steps[{i}]={buf.traj_steps[i]} != expected {expected}"
        )

    print("test_multiple_slices_trajectory_tracking: PASS")


if __name__ == "__main__":
    test_basic_insertion_and_sampling()
    test_next_obs_correctness()
    test_per_channel_done()
    test_truncated_no_done()
    test_nstep_sampling()
    test_nstep_done_truncation()
    test_circular_overwrite()
    test_buffer_stats()
    test_multiple_slices_trajectory_tracking()
    print("\nAll tests passed!")
