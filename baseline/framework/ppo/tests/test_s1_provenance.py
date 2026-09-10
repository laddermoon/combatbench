"""S1 frame provenance tests.

Tests cover:
- TrajectoryProvenance dataclass (frozen, defaults, fields).
- PPOBuffer.seg_provenance collection (with/without provenance, empty).
- PPOBuffer.frame_id (flat index → frame ID, with/without provenance).
- PPOBuffer.frame_ids (all-or-none strategy, empty buffer).
- PPOBuffer.find_frames (exact, wildcards, no provenance, invalid expr).
- StandupStepV3 fills provenance in build_trajectories.
- ppo_update unchanged with provenance (P1: no training dynamics change).

Conventions follow test_trainer.py:
- Simple print("test_xxx: PASS") at the end of each test.
- `if __name__ == "__main__":` block runs all tests sequentially.
- No pytest fixtures required (but pytest-compatible).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.trajectory import (
    ChannelData,
    RewardChannel,
    Trajectory,
    TrajectoryProvenance,
)
from baseline.framework.ppo.experiment import PPOParams
from baseline.framework.ppo.trainer import PPOBuffer


# ---------------------------------------------------------------------------
# Test helpers (minimal, from test_trainer.py)
# ---------------------------------------------------------------------------

class SimpleActor(nn.Module):
    """Minimal actor for PPOBuffer construction (implements evaluate_actions)."""

    def __init__(self, obs_dim: int = 8, action_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def evaluate_actions(self, obs, actions, *, want_stats=False, **kw):
        from baseline.framework.ppo.experiment import ActorEval
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        diff = actions - mean
        log_prob = -0.5 * ((diff / std) ** 2).sum(-1) - \
                   0.5 * np.log(2 * np.pi) * self.action_dim - \
                   self.log_std.sum()
        uncertainty = torch.full_like(log_prob, 0.5)
        return ActorEval(log_prob=log_prob, uncertainty=uncertainty, stats={})


def _make_trajectory(
    T: int = 5,
    obs_dim: int = 8,
    action_dim: int = 3,
    *,
    provenance: Optional[TrajectoryProvenance] = None,
    rng: np.random.Generator = None,
) -> Trajectory:
    """Build a random Trajectory with optional provenance."""
    if rng is None:
        rng = np.random.default_rng()
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32)
    last_obs = rng.standard_normal(obs_dim).astype(np.float32)
    channels = {
        "r_a": ChannelData(
            reward=rng.standard_normal(T).astype(np.float32),
            is_terminated=True,
            actor_weight=1.0,
        ),
    }
    return Trajectory(
        obs=obs,
        actions=actions,
        last_obs=last_obs,
        channels=channels,
        provenance=provenance,
    )


def _make_buffer(trajectories, obs_dim=8, action_dim=3):
    """Build a PPOBuffer with a fresh SimpleActor."""
    actor = SimpleActor(obs_dim, action_dim)
    return PPOBuffer(
        trajectories=trajectories,
        actor=actor,
        device=torch.device("cpu"),
        reward_keys=("r_a",),
    )


# ---------------------------------------------------------------------------
# TrajectoryProvenance dataclass
# ---------------------------------------------------------------------------

def test_trajectory_provenance_default_none():
    """Trajectory() without provenance → provenance is None."""
    traj = _make_trajectory()
    assert traj.provenance is None
    print("test_trajectory_provenance_default_none: PASS")


def test_trajectory_provenance_set():
    """Trajectory with provenance → fields correct."""
    prov = TrajectoryProvenance(
        episode_index=3, agent_id="robot_a",
        t_start=10, termination_reason="ko",
    )
    traj = _make_trajectory(provenance=prov)
    assert traj.provenance is not None
    assert traj.provenance.episode_index == 3
    assert traj.provenance.agent_id == "robot_a"
    assert traj.provenance.t_start == 10
    assert traj.provenance.termination_reason == "ko"
    print("test_trajectory_provenance_set: PASS")


def test_trajectory_provenance_frozen():
    """TrajectoryProvenance is frozen → cannot modify fields."""
    prov = TrajectoryProvenance(episode_index=0, agent_id="robot_a")
    with pytest.raises((AttributeError, Exception)):
        prov.episode_index = 5
    print("test_trajectory_provenance_frozen: PASS")


def test_trajectory_provenance_defaults():
    """TrajectoryProvenance defaults: t_start=0, termination_reason=''."""
    prov = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    assert prov.t_start == 0
    assert prov.termination_reason == ""
    print("test_trajectory_provenance_defaults: PASS")


# ---------------------------------------------------------------------------
# PPOBuffer.seg_provenance collection
# ---------------------------------------------------------------------------

def test_buffer_seg_provenance_collected():
    """2 trajectories with provenance → seg_provenance has 2 entries."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
    ]
    buf = _make_buffer(trajs)
    assert len(buf.seg_provenance) == 2
    assert buf.seg_provenance[0] is prov1
    assert buf.seg_provenance[1] is prov2
    print("test_buffer_seg_provenance_collected: PASS")


def test_buffer_seg_provenance_none_when_missing():
    """Trajectories without provenance → seg_provenance is [None, None]."""
    rng = np.random.default_rng(42)
    trajs = [_make_trajectory(T=5, rng=rng), _make_trajectory(T=3, rng=rng)]
    buf = _make_buffer(trajs)
    assert len(buf.seg_provenance) == 2
    assert buf.seg_provenance[0] is None
    assert buf.seg_provenance[1] is None
    print("test_buffer_seg_provenance_none_when_missing: PASS")


def test_buffer_seg_provenance_empty():
    """Empty trajectories → seg_provenance is []."""
    buf = _make_buffer([])
    assert buf.seg_provenance == []
    print("test_buffer_seg_provenance_empty: PASS")


# ---------------------------------------------------------------------------
# PPOBuffer.frame_id
# ---------------------------------------------------------------------------

def test_frame_id_with_provenance():
    """2 segments (ep3/robot_a/T=5, ep7/robot_b/T=3) → correct frame IDs."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
    ]
    buf = _make_buffer(trajs)
    assert buf.frame_id(0) == "ep0003:robot_a:0"
    assert buf.frame_id(4) == "ep0003:robot_a:4"
    assert buf.frame_id(5) == "ep0007:robot_b:0"
    assert buf.frame_id(7) == "ep0007:robot_b:2"
    print("test_frame_id_with_provenance: PASS")


def test_frame_id_without_provenance():
    """No provenance → frame_id returns 'flat:{i}'."""
    rng = np.random.default_rng(42)
    trajs = [_make_trajectory(T=5, rng=rng)]
    buf = _make_buffer(trajs)
    assert buf.frame_id(0) == "flat:0"
    assert buf.frame_id(4) == "flat:4"
    print("test_frame_id_without_provenance: PASS")


def test_frame_id_out_of_range():
    """frame_id(999) out of range → returns 'flat:999'."""
    rng = np.random.default_rng(42)
    prov = TrajectoryProvenance(episode_index=1, agent_id="robot_a")
    trajs = [_make_trajectory(T=5, provenance=prov, rng=rng)]
    buf = _make_buffer(trajs)
    assert buf.frame_id(999) == "flat:999"
    print("test_frame_id_out_of_range: PASS")


def test_frame_id_with_t_start():
    """Provenance with t_start=10 → frame_id reflects offset."""
    rng = np.random.default_rng(42)
    prov = TrajectoryProvenance(episode_index=1, agent_id="robot_a", t_start=10)
    trajs = [_make_trajectory(T=5, provenance=prov, rng=rng)]
    buf = _make_buffer(trajs)
    assert buf.frame_id(0) == "ep0001:robot_a:10"
    assert buf.frame_id(4) == "ep0001:robot_a:14"
    print("test_frame_id_with_t_start: PASS")


# ---------------------------------------------------------------------------
# PPOBuffer.frame_ids
# ---------------------------------------------------------------------------

def test_frame_ids_all_present():
    """All segments have provenance → returns (n,) string array."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
    ]
    buf = _make_buffer(trajs)
    ids = buf.frame_ids()
    assert ids is not None
    assert len(ids) == 8
    assert ids[0] == "ep0003:robot_a:0"
    assert ids[4] == "ep0003:robot_a:4"
    assert ids[5] == "ep0007:robot_b:0"
    assert ids[7] == "ep0007:robot_b:2"
    print("test_frame_ids_all_present: PASS")


def test_frame_ids_returns_none_when_missing():
    """Any segment without provenance → returns None."""
    rng = np.random.default_rng(42)
    prov = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    trajs = [
        _make_trajectory(T=5, provenance=prov, rng=rng),
        _make_trajectory(T=3, provenance=None, rng=rng),  # no provenance
    ]
    buf = _make_buffer(trajs)
    ids = buf.frame_ids()
    assert ids is None
    print("test_frame_ids_returns_none_when_missing: PASS")


def test_frame_ids_empty_buffer():
    """Empty buffer → frame_ids returns None."""
    buf = _make_buffer([])
    assert buf.frame_ids() is None
    print("test_frame_ids_empty_buffer: PASS")


# ---------------------------------------------------------------------------
# PPOBuffer.find_frames
# ---------------------------------------------------------------------------

def test_find_frames_exact():
    """find_frames('ep0003:robot_a:2') → returns that frame's flat index."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
    ]
    buf = _make_buffer(trajs)
    result = buf.find_frames("ep0003:robot_a:2")
    assert len(result) == 1
    assert result[0] == 2
    print("test_find_frames_exact: PASS")


def test_find_frames_wildcard_episode():
    """find_frames('*:robot_a:0') → all robot_a frame-0 indices."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    prov3 = TrajectoryProvenance(episode_index=9, agent_id="robot_a")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
        _make_trajectory(T=4, provenance=prov3, rng=rng),
    ]
    buf = _make_buffer(trajs)
    result = buf.find_frames("*:robot_a:0")
    # ep3 starts at flat 0, ep9 starts at flat 8
    assert len(result) == 2
    assert 0 in result
    assert 8 in result
    print("test_find_frames_wildcard_episode: PASS")


def test_find_frames_wildcard_agent():
    """find_frames('ep0003:*:*') → all frames from episode 3."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
    ]
    buf = _make_buffer(trajs)
    result = buf.find_frames("ep0003:*:*")
    assert len(result) == 5
    assert list(result) == [0, 1, 2, 3, 4]
    print("test_find_frames_wildcard_agent: PASS")


def test_find_frames_wildcard_time():
    """find_frames('*:*:0') → first frame of every segment."""
    rng = np.random.default_rng(42)
    prov1 = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    prov2 = TrajectoryProvenance(episode_index=7, agent_id="robot_b")
    trajs = [
        _make_trajectory(T=5, provenance=prov1, rng=rng),
        _make_trajectory(T=3, provenance=prov2, rng=rng),
    ]
    buf = _make_buffer(trajs)
    result = buf.find_frames("*:*:0")
    assert len(result) == 2
    assert 0 in result  # ep3 frame 0
    assert 5 in result  # ep7 frame 0
    print("test_find_frames_wildcard_time: PASS")


def test_find_frames_no_provenance():
    """No provenance → returns empty array + warning."""
    rng = np.random.default_rng(42)
    trajs = [_make_trajectory(T=5, rng=rng)]
    buf = _make_buffer(trajs)
    result = buf.find_frames("ep0003:robot_a:0")
    assert len(result) == 0
    print("test_find_frames_no_provenance: PASS")


def test_find_frames_invalid_expr():
    """Invalid expr (missing one segment) → ValueError."""
    rng = np.random.default_rng(42)
    prov = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    trajs = [_make_trajectory(T=5, provenance=prov, rng=rng)]
    buf = _make_buffer(trajs)
    with pytest.raises(ValueError):
        buf.find_frames("ep3:robot_a")  # missing :t
    print("test_find_frames_invalid_expr: PASS")


def test_find_frames_no_match():
    """find_frames('ep9999:robot_a:0') → empty array."""
    rng = np.random.default_rng(42)
    prov = TrajectoryProvenance(episode_index=3, agent_id="robot_a")
    trajs = [_make_trajectory(T=5, provenance=prov, rng=rng)]
    buf = _make_buffer(trajs)
    result = buf.find_frames("ep9999:robot_a:0")
    assert len(result) == 0
    print("test_find_frames_no_match: PASS")


# ---------------------------------------------------------------------------
# StandupStepV3 fills provenance
# ---------------------------------------------------------------------------

def _make_synthetic_episode(
    T: int = 100,
    episode_index: int = 3,
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
        episode_index=episode_index,
        observer_outputs=observer_outputs,
        observations={agent_id: np.zeros((T, 96), dtype=np.float32)},
        actions={agent_id: np.zeros((T, 21), dtype=np.float32)},
        final_observation={agent_id: np.zeros(96, dtype=np.float32)},
        explore_factors={agent_id: np.zeros(T, dtype=np.float32)},
        agent_termination_proposal_records=term_records,
    )


def test_standup_step_v3_fills_provenance():
    """StandupStepV3.build_trajectories fills provenance."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    e = StandupStepV3()
    ep = _make_synthetic_episode(episode_index=3, agent_id="robot_a")
    trajs = e.build_trajectories([ep])
    assert len(trajs) > 0
    for traj in trajs:
        assert traj.provenance is not None
        assert traj.provenance.episode_index == 3
    # At least one trajectory should be robot_a
    robot_a_trajs = [t for t in trajs if t.provenance.agent_id == "robot_a"]
    assert len(robot_a_trajs) > 0
    print("test_standup_step_v3_fills_provenance: PASS")


def test_standup_step_v3_provenance_termination_reason():
    """Episode with termination record → provenance.termination_reason set."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    e = StandupStepV3()
    ep = _make_synthetic_episode(
        episode_index=5, agent_id="robot_a", termination_reason="ko",
    )
    trajs = e.build_trajectories([ep])
    robot_a_trajs = [t for t in trajs if t.provenance.agent_id == "robot_a"]
    assert len(robot_a_trajs) > 0
    assert robot_a_trajs[0].provenance.termination_reason == "ko"
    print("test_standup_step_v3_provenance_termination_reason: PASS")


def test_standup_step_v3_provenance_no_termination():
    """Episode without termination record → termination_reason=''."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    e = StandupStepV3()
    ep = _make_synthetic_episode(
        episode_index=5, agent_id="robot_a", termination_reason="",
    )
    trajs = e.build_trajectories([ep])
    robot_a_trajs = [t for t in trajs if t.provenance.agent_id == "robot_a"]
    assert len(robot_a_trajs) > 0
    assert robot_a_trajs[0].provenance.termination_reason == ""
    print("test_standup_step_v3_provenance_no_termination: PASS")


# ---------------------------------------------------------------------------
# ppo_update unchanged with provenance (P1)
# ---------------------------------------------------------------------------

def test_ppo_update_unchanged_with_provenance():
    """PPOBuffer with provenance → same data arrays as without (P1).

    Provenance is pure metadata; it must not affect the buffer's
    computational contents (obs, actions, log_probs, rewards, etc.).
    This is sufficient to prove ppo_update is unchanged, since
    ppo_update only reads the buffer's data arrays, not provenance.
    """
    rng = np.random.default_rng(42)
    prov = TrajectoryProvenance(episode_index=1, agent_id="robot_a")

    # Build two identical trajectories, one with provenance, one without
    obs = rng.standard_normal((10, 8)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (10, 3)).astype(np.float32)
    last_obs = rng.standard_normal(8).astype(np.float32)
    reward = rng.standard_normal(10).astype(np.float32)
    channels = {"r_a": ChannelData(reward=reward, is_terminated=True,
                                     actor_weight=1.0)}

    traj_with = Trajectory(obs=obs.copy(), actions=actions.copy(),
                           last_obs=last_obs.copy(), channels=channels,
                           provenance=prov)
    traj_without = Trajectory(obs=obs.copy(), actions=actions.copy(),
                               last_obs=last_obs.copy(), channels=channels,
                               provenance=None)

    actor = SimpleActor(8, 3)
    buf_with = PPOBuffer([traj_with], actor, torch.device("cpu"), ("r_a",))
    buf_without = PPOBuffer([traj_without], actor, torch.device("cpu"), ("r_a",))

    # All computational arrays must be identical
    np.testing.assert_array_equal(buf_with.obs, buf_without.obs)
    np.testing.assert_array_equal(buf_with.actions, buf_without.actions)
    np.testing.assert_array_equal(buf_with.log_probs, buf_without.log_probs)
    np.testing.assert_array_equal(buf_with.sample_weights, buf_without.sample_weights)
    np.testing.assert_array_equal(buf_with.explore_factor, buf_without.explore_factor)
    np.testing.assert_array_equal(buf_with.floor_weight, buf_without.floor_weight)
    np.testing.assert_array_equal(
        buf_with.reward_data["r_a"][0], buf_without.reward_data["r_a"][0]
    )
    assert buf_with.ep_lengths == buf_without.ep_lengths

    # Only seg_provenance differs
    assert buf_with.seg_provenance[0] is prov
    assert buf_without.seg_provenance[0] is None

    print("test_ppo_update_unchanged_with_provenance: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_trajectory_provenance_default_none()
    test_trajectory_provenance_set()
    test_trajectory_provenance_frozen()
    test_trajectory_provenance_defaults()
    test_buffer_seg_provenance_collected()
    test_buffer_seg_provenance_none_when_missing()
    test_buffer_seg_provenance_empty()
    test_frame_id_with_provenance()
    test_frame_id_without_provenance()
    test_frame_id_out_of_range()
    test_frame_id_with_t_start()
    test_frame_ids_all_present()
    test_frame_ids_returns_none_when_missing()
    test_frame_ids_empty_buffer()
    test_find_frames_exact()
    test_find_frames_wildcard_episode()
    test_find_frames_wildcard_agent()
    test_find_frames_wildcard_time()
    test_find_frames_no_provenance()
    test_find_frames_invalid_expr()
    test_find_frames_no_match()
    test_standup_step_v3_fills_provenance()
    test_standup_step_v3_provenance_termination_reason()
    test_standup_step_v3_provenance_no_termination()
    test_ppo_update_unchanged_with_provenance()
    print("\nAll S1 provenance tests passed.")
