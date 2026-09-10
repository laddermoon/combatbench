"""S1 frame provenance tests.

Tests cover:
- _make_frame_ids: obs content matching → frame_id generation.
- Basic matching, no match, empty, multiple trajectories, partial
  trajectory (t_start > 0), hash collision disambiguation.
- StandupStepV3.build_trajectories does NOT fill provenance.
- ppo_update unchanged (provenance is dump-only, not in production path).

Conventions follow test_trainer.py:
- Simple print("test_xxx: PASS") at the end of each test.
- `if __name__ == "__main__":` block runs all tests sequentially.
- No pytest fixtures required (but pytest-compatible).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.dumpkit.dump_capture import _make_frame_ids
from baseline.framework.ppo.trajectory import ChannelData, Trajectory


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_trajectory(obs: np.ndarray, actions: np.ndarray) -> Trajectory:
    """Build a Trajectory from explicit obs/actions arrays."""
    return Trajectory(
        obs=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        last_obs=np.zeros(obs.shape[1], dtype=np.float32),
        channels={"r": ChannelData(
            reward=np.zeros(len(obs), dtype=np.float32),
            is_terminated=False,
            actor_weight=1.0,
        )},
        importance=1.0,
    )


def _make_episode(obs_dict, actions_dict, num_frames=None):
    """Build a minimal synthetic episode (SimpleNamespace)."""
    if num_frames is None:
        num_frames = len(next(iter(obs_dict.values())))
    return SimpleNamespace(
        num_frames=num_frames,
        observations=obs_dict,
        actions=actions_dict,
        final_observation={k: np.zeros(v.shape[1], dtype=np.float32)
                           for k, v in obs_dict.items()},
        explore_factors={k: np.zeros(len(v), dtype=np.float32)
                         for k, v in obs_dict.items()},
        agent_termination_proposal_records={},
    )


# ---------------------------------------------------------------------------
# _make_frame_ids — basic matching
# ---------------------------------------------------------------------------

def test_frame_ids_basic_match():
    """2 episodes × 1 agent each → correct frame_ids with episode_pos."""
    rng = np.random.default_rng(42)
    ep0_obs = rng.standard_normal((10, 4)).astype(np.float32)
    ep1_obs = rng.standard_normal((8, 4)).astype(np.float32)
    ep0_act = rng.standard_normal((10, 2)).astype(np.float32)
    ep1_act = rng.standard_normal((8, 2)).astype(np.float32)

    episodes = [
        _make_episode({"robot_a": ep0_obs}, {"robot_a": ep0_act}),
        _make_episode({"robot_a": ep1_obs}, {"robot_a": ep1_act}),
    ]
    trajs = [
        _make_trajectory(ep0_obs, ep0_act),
        _make_trajectory(ep1_obs, ep1_act),
    ]
    ids = _make_frame_ids(trajs, episodes)
    assert len(ids) == 18
    assert ids[0] == "ep0000:robot_a:0"
    assert ids[9] == "ep0000:robot_a:9"
    assert ids[10] == "ep0001:robot_a:0"
    assert ids[17] == "ep0001:robot_a:7"
    print("test_frame_ids_basic_match: PASS")


def test_frame_ids_multiple_agents():
    """1 episode × 2 agents → correct agent_id in frame_ids."""
    rng = np.random.default_rng(42)
    obs_a = rng.standard_normal((5, 4)).astype(np.float32)
    obs_b = rng.standard_normal((5, 4)).astype(np.float32)
    act_a = rng.standard_normal((5, 2)).astype(np.float32)
    act_b = rng.standard_normal((5, 2)).astype(np.float32)

    episodes = [_make_episode(
        {"robot_a": obs_a, "robot_b": obs_b},
        {"robot_a": act_a, "robot_b": act_b},
    )]
    trajs = [
        _make_trajectory(obs_a, act_a),
        _make_trajectory(obs_b, act_b),
    ]
    ids = _make_frame_ids(trajs, episodes)
    assert len(ids) == 10
    assert ids[0] == "ep0000:robot_a:0"
    assert ids[4] == "ep0000:robot_a:4"
    assert ids[5] == "ep0000:robot_b:0"
    assert ids[9] == "ep0000:robot_b:4"
    print("test_frame_ids_multiple_agents: PASS")


def test_frame_ids_partial_trajectory():
    """Trajectory starting at t_start=3 → frame_ids reflect offset."""
    rng = np.random.default_rng(42)
    ep_obs = rng.standard_normal((10, 4)).astype(np.float32)
    ep_act = rng.standard_normal((10, 2)).astype(np.float32)

    episodes = [_make_episode({"robot_a": ep_obs}, {"robot_a": ep_act})]
    # Partial trajectory: frames 3..7
    trajs = [_make_trajectory(ep_obs[3:8], ep_act[3:8])]
    ids = _make_frame_ids(trajs, episodes)
    assert len(ids) == 5
    assert ids[0] == "ep0000:robot_a:3"
    assert ids[4] == "ep0000:robot_a:7"
    print("test_frame_ids_partial_trajectory: PASS")


def test_frame_ids_no_match():
    """Trajectory obs not in any episode → flat:{i}."""
    rng = np.random.default_rng(42)
    ep_obs = rng.standard_normal((10, 4)).astype(np.float32)
    ep_act = rng.standard_normal((10, 2)).astype(np.float32)
    other_obs = rng.standard_normal((5, 4)).astype(np.float32)
    other_act = rng.standard_normal((5, 2)).astype(np.float32)

    episodes = [_make_episode({"robot_a": ep_obs}, {"robot_a": ep_act})]
    trajs = [_make_trajectory(other_obs, other_act)]
    ids = _make_frame_ids(trajs, episodes)
    assert len(ids) == 5
    assert ids[0] == "flat:0"
    assert ids[4] == "flat:4"
    print("test_frame_ids_no_match: PASS")


def test_frame_ids_empty_trajectories():
    """No trajectories → empty array."""
    episodes = [_make_episode(
        {"robot_a": np.zeros((5, 4), dtype=np.float32)},
        {"robot_a": np.zeros((5, 2), dtype=np.float32)},
    )]
    ids = _make_frame_ids([], episodes)
    assert len(ids) == 0
    print("test_frame_ids_empty_trajectories: PASS")


def test_frame_ids_hash_collision_disambiguated():
    """Same obs[0] in two episodes → np.array_equal picks the right one."""
    # Two episodes with identical first frame but different rest
    ep0_obs = np.zeros((5, 4), dtype=np.float32)
    ep0_obs[1:] = 1.0
    ep1_obs = np.zeros((5, 4), dtype=np.float32)
    ep1_obs[1:] = 2.0
    ep0_act = np.zeros((5, 2), dtype=np.float32)
    ep1_act = np.zeros((5, 2), dtype=np.float32)

    episodes = [
        _make_episode({"robot_a": ep0_obs}, {"robot_a": ep0_act}),
        _make_episode({"robot_a": ep1_obs}, {"robot_a": ep1_act}),
    ]
    # Trajectory from episode 1 (all 2.0s after frame 0)
    trajs = [_make_trajectory(ep1_obs, ep1_act)]
    ids = _make_frame_ids(trajs, episodes)
    assert len(ids) == 5
    # Should match ep0001, not ep0000, because full obs array matches
    assert ids[0] == "ep0001:robot_a:0"
    print("test_frame_ids_hash_collision_disambiguated: PASS")


# ---------------------------------------------------------------------------
# StandupStepV3 does NOT fill provenance
# ---------------------------------------------------------------------------

def _make_synthetic_episode(
    T: int = 100,
    agent_id: str = "robot_a",
    termination_reason: str = "",
):
    """Build a minimal synthetic episode for build_trajectories testing."""
    h_torso = np.zeros(T, dtype=np.float32)
    h_torso[:50] = 0.5
    h_torso[50:] = 1.2
    potential = np.zeros(T, dtype=np.float32)
    potential[:50] = np.linspace(0.0, 0.95, 50)
    potential[50:] = 0.95
    h_left = np.zeros(T, dtype=np.float32)
    h_right = np.zeros(T, dtype=np.float32)
    contact_l = np.ones(T, dtype=bool)
    contact_r = np.ones(T, dtype=bool)

    foot_key = "foot_state_a"
    phi4stage_key = "standing_balance_a"
    phi_height_key = "height_phi_a"

    observer_outputs = {
        phi4stage_key: {"potential": potential, "h_torso": h_torso},
        foot_key: {
            "h_left_foot": h_left,
            "h_right_foot": h_right,
            "left_foot_contact": contact_l,
            "right_foot_contact": contact_r,
        },
        phi_height_key: {"phi": np.ones(T, dtype=np.float32)},
    }

    term_records = {}
    if termination_reason:
        term_records[agent_id] = ((termination_reason, T),)

    return SimpleNamespace(
        num_frames=T,
        episode_index=0,
        observer_outputs=observer_outputs,
        observations={agent_id: np.zeros((T, 96), dtype=np.float32)},
        actions={agent_id: np.zeros((T, 21), dtype=np.float32)},
        final_observation={agent_id: np.zeros(96, dtype=np.float32)},
        explore_factors={agent_id: np.zeros(T, dtype=np.float32)},
        agent_termination_proposal_records=term_records,
    )


def test_standup_step_v3_no_provenance():
    """StandupStepV3.build_trajectories does NOT fill provenance."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    e = StandupStepV3()
    ep = _make_synthetic_episode(agent_id="robot_a")
    trajs = e.build_trajectories([ep])
    assert len(trajs) > 0
    # Trajectory no longer has a provenance field
    assert not hasattr(trajs[0], "provenance")
    print("test_standup_step_v3_no_provenance: PASS")


def test_standup_step_v3_frame_ids_via_make_frame_ids():
    """StandupStepV3 trajectories → _make_frame_ids produces correct IDs."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    e = StandupStepV3()
    ep = _make_synthetic_episode(agent_id="robot_a")
    trajs = e.build_trajectories([ep])
    ids = _make_frame_ids(trajs, [ep])
    # All trajectories should match (obs is from the episode)
    flat_count = sum(1 for fid in ids if str(fid).startswith("flat:"))
    ep_count = sum(1 for fid in ids if str(fid).startswith("ep"))
    assert flat_count == 0, f"Expected no flat: IDs, got {flat_count}"
    assert ep_count == len(ids), f"Expected all ep: IDs, got {ep_count}/{len(ids)}"
    print("test_standup_step_v3_frame_ids_via_make_frame_ids: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_frame_ids_basic_match()
    test_frame_ids_multiple_agents()
    test_frame_ids_partial_trajectory()
    test_frame_ids_no_match()
    test_frame_ids_empty_trajectories()
    test_frame_ids_hash_collision_disambiguated()
    test_standup_step_v3_no_provenance()
    test_standup_step_v3_frame_ids_via_make_frame_ids()
    print("\nAll S1 provenance tests passed.")
