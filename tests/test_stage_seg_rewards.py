"""Regression tests for basic_balance_v2_stage_seg phase rewards / segments.

Guards the off-by-one class of bug where the boundary reward was placed on
the frame where the *new* phase was first observed instead of on the *last
frame of the ending run*, which made both transition rewards unreachable.

Run:
    PYTHONPATH=/data1/mono/things/combatbench python3 -m pytest tests/test_stage_seg_rewards.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from baseline.humanoid21.curriculum.experiments.exp_basic_balance_v2_stage_seg import (
    BasicBalanceV2StageSegConfig,
)


class FakeEpisode:
    """Minimal Episode stand-in carrying only what extract_rewards reads."""

    def __init__(self, phases: str, fell: bool):
        # phases: string of 'S' (stability) / 'X' (struggle), one char per frame
        self.num_frames = len(phases)
        self._is_struggle = [c == "X" for c in phases]
        self.termination_proposals = ["imbalance"] if fell else []

        T = self.num_frames
        self.observer_outputs = {
            "phase": {
                "is_struggle": list(self._is_struggle),
                # `transition` is deliberately left as all-"none": the reward
                # logic must derive boundaries from is_struggle runs alone.
                "transition": ["none"] * T,
            },
            "cross_support": np.zeros(T, dtype=np.float32),
            "posture": {
                "joint_deviation": np.zeros(T, dtype=np.float32),
                "joint_vel": np.zeros(T, dtype=np.float32),
                "torso_tilt": np.zeros(T, dtype=np.float32),
                "foot_height": np.zeros(T, dtype=np.float32),
            },
        }


@pytest.fixture
def exp():
    return BasicBalanceV2StageSegConfig()


# ---------------------------------------------------------------------------
# _phase_runs
# ---------------------------------------------------------------------------

def test_phase_runs_covers_episode_without_gaps(exp):
    ep = FakeEpisode("SSXXXSSXX", fell=True)
    runs = exp._phase_runs(ep)
    assert runs == [(0, 2, False), (2, 5, True), (5, 7, False), (7, 9, True)]
    # contiguous, gapless, full coverage
    assert runs[0][0] == 0
    assert runs[-1][1] == ep.num_frames
    for (_, prev_end, _), (next_start, _, _) in zip(runs, runs[1:]):
        assert prev_end == next_start


def test_phase_runs_empty_episode(exp):
    assert exp._phase_runs(FakeEpisode("", fell=False)) == []


# ---------------------------------------------------------------------------
# Boundary reward placement — the regression this file exists for
# ---------------------------------------------------------------------------

def test_recovery_bonus_lands_on_last_struggle_frame(exp):
    """Struggle run [0,3) recovers at frame 3 -> +1.0 must land on frame 2."""
    ep = FakeEpisode("XXXSSS", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]

    assert r[0] == pytest.approx(-0.01)
    assert r[1] == pytest.approx(-0.01)
    # -0.01 per-step plus the +1.0 recovery terminal
    assert r[2] == pytest.approx(-0.01 + 1.0)
    # stability frames: no per-step bonus, final run is a timeout -> no terminal
    assert r[3:] == pytest.approx(0.0)


def test_degradation_penalty_lands_on_last_stability_frame(exp):
    """Stability run [0,3) degrades at frame 3 -> -1.0 must land on frame 2."""
    ep = FakeEpisode("SSSXXX", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]

    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(-1.0)
    assert r[3:] == pytest.approx(-0.01)


def test_transition_rewards_are_actually_reachable(exp):
    """The original bug made both transition rewards dead code.

    An episode that recovers once must have a strictly positive frame.
    """
    ep = FakeEpisode("XXXSSS", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]
    assert r.max() > 0.5, "recovery bonus never fired"


def test_fall_penalty_on_final_frame(exp):
    ep = FakeEpisode("SSXX", fell=True)
    r = exp.extract_rewards(ep)["r_struggle"]
    # stability run [0,2) degrades -> -1.0 on frame 1
    assert r[1] == pytest.approx(-1.0)
    # final struggle run ends by falling -> -0.01 - 1.0 on last frame
    assert r[3] == pytest.approx(-0.01 - 1.0)


def test_timeout_gives_no_terminal_reward(exp):
    """A run that ends by timeout is bootstrapped, so no terminal reward."""
    ep = FakeEpisode("SSSS", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]
    assert r == pytest.approx(0.0)


def test_oscillation_is_never_profitable(exp):
    """A struggle/stability cycle must cost at least the per-step penalty.

    Otherwise the agent could farm recovery bonuses by flickering.
    """
    stable = exp.extract_rewards(FakeEpisode("S" * 12, fell=False))["r_struggle"].sum()
    oscil = exp.extract_rewards(FakeEpisode("SSXXSSXXSSXX", fell=False))["r_struggle"].sum()
    assert oscil < stable


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def test_segments_match_phase_runs(exp):
    ep = FakeEpisode("SSXXXSS", fell=False)
    runs = exp._phase_runs(ep)
    segs = exp.prepare_segments(ep)

    assert len(segs) == len(runs)
    for seg, (start, end, is_struggle) in zip(segs, runs):
        assert (seg.start, seg.end) == (start, end)
        if is_struggle:
            assert seg.key_weights == {"r_struggle": 1.0}
        else:
            assert seg.key_weights is None


def test_boundary_segments_are_terminated(exp):
    """Boundary reward is explicit, so bootstrapping would double-count."""
    ep = FakeEpisode("SSXXXSS", fell=False)
    segs = exp.prepare_segments(ep)
    for seg in segs[:-1]:
        assert seg.termination == "terminated"


def test_final_segment_termination_depends_on_fall(exp):
    fell = exp.prepare_segments(FakeEpisode("SSXX", fell=True))
    assert fell[-1].termination == "terminated"

    timeout = exp.prepare_segments(FakeEpisode("SSXX", fell=False))
    assert timeout[-1].termination == "truncated"


def test_segments_cover_all_frames_exactly_once(exp):
    ep = FakeEpisode("XSSXXSXS", fell=True)
    covered = np.zeros(ep.num_frames, dtype=int)
    for seg in exp.prepare_segments(ep):
        covered[seg.start:seg.end] += 1
    assert (covered == 1).all()


# ---------------------------------------------------------------------------
# Posture rewards must stay identical to basic_balance_v2
# ---------------------------------------------------------------------------

def test_posture_rewards_match_basic_balance_v2(exp):
    from baseline.humanoid21.curriculum.experiments.exp_basic_balance_v2 import (
        BasicBalanceV2Config,
    )

    base = BasicBalanceV2Config()
    ep = FakeEpisode("SSXXSS", fell=True)

    # give the posture observer non-trivial values
    ep.observer_outputs["posture"] = {
        "joint_deviation": np.array([0.05, 0.2, 0.3, 0.0, 0.15, 0.1], dtype=np.float32),
        "joint_vel": np.array([0.0, 0.5, 0.05, 0.3, 0.1, 0.2], dtype=np.float32),
        "torso_tilt": np.array([0.1, 0.4, 0.26, 0.5, 0.3, 0.0], dtype=np.float32),
        "foot_height": np.array([0.0, 0.2, 0.1, 0.05, 0.3, 0.12], dtype=np.float32),
    }

    got = exp.extract_rewards(ep)
    want = base.extract_rewards(ep)
    for key in ("r_cross", "r_joint", "r_vel", "r_tilt", "r_foot"):
        assert got[key] == pytest.approx(want[key]), key


# ---------------------------------------------------------------------------
# Episode metrics
# ---------------------------------------------------------------------------

def test_episode_metrics(exp):
    ep = FakeEpisode("XXXSSSSXX", fell=True)
    m = exp.compute_episode_metrics(ep)
    assert m["survived"] == 0.0
    assert m["struggle_steps"] == 5
    assert m["recoveries"] == 1.0
    assert m["longest_stable"] == 4.0
