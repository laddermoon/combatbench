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

    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(1.0)
    # final stability run ends by timeout -> no terminal, bootstrapped
    assert r[3:] == pytest.approx(0.0)


def test_degradation_penalty_lands_on_last_stability_frame(exp):
    """Stability run [0,3) degrades at frame 3 -> -1.0 must land on frame 2."""
    ep = FakeEpisode("SSSXXX", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]

    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(-1.0)
    assert r[3:] == pytest.approx(0.0)


def test_reward_is_terminal_only(exp):
    """No per-step term in either phase -- only run-boundary frames are nonzero."""
    ep = FakeEpisode("SSSSXXXXSSSS", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]

    nonzero = set(np.flatnonzero(r).tolist())
    # runs end at 4 and 8 -> terminal frames 3 and 7; the final run ends by
    # timeout so it contributes nothing.
    assert nonzero == {3, 7}


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
    # final struggle run ends by falling -> -1.0 on last frame
    assert r[3] == pytest.approx(-1.0)


def test_timeout_gives_no_terminal_reward(exp):
    """A run that ends by timeout is bootstrapped, so no terminal reward."""
    ep = FakeEpisode("SSSS", fell=False)
    r = exp.extract_rewards(ep)["r_struggle"]
    assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Return shape -- incentives must point the right way AND carry enough signal.
#
# These guard against reintroducing a per-step term.  A -0.01 per-step
# struggle penalty has an infinite-horizon discounted sum of
# -0.01/(1-0.99) = -1.0, exactly equal to the -1.0 fall terminal, so
#
#   G(k) = -0.01*(1-g^k)/(1-g) - g^(k-1) = -1 - 0.01*g^(k-1)
#
# which still increases in k but spans only 0.01 instead of 1.0 -- a 100x
# loss of signal that zeroed r_struggle's policy-gradient contribution.
# Monotonicity alone does NOT catch this; the spread test does.
# ---------------------------------------------------------------------------

GAMMA = 0.99


def _discounted_return(rewards: np.ndarray, gamma: float = GAMMA) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
    return g


def _struggle_return(exp, phases: str, fell: bool) -> float:
    r = exp.extract_rewards(FakeEpisode(phases, fell=fell))["r_struggle"]
    return _discounted_return(r)


@pytest.mark.parametrize("horizon", [5, 10, 20, 50])
def test_delaying_a_fall_is_rewarded(exp, horizon):
    """Falling later must be strictly better than falling earlier."""
    early = _struggle_return(exp, "XX", fell=True)
    late = _struggle_return(exp, "X" * horizon, fell=True)
    assert late > early


def test_fall_return_is_monotonically_increasing_in_duration(exp):
    returns = [_struggle_return(exp, "X" * k, fell=True) for k in range(1, 60)]
    assert all(b > a for a, b in zip(returns, returns[1:]))


def test_fall_return_spread_is_large_enough_to_learn_from(exp):
    """The duration signal must not be cancelled by a per-step term.

    Over the reachable horizon the return must vary by O(1), not O(0.01).
    A -0.01 per-step struggle penalty flattens this to 0.01.
    """
    returns = [_struggle_return(exp, "X" * k, fell=True) for k in range(1, 31)]
    spread = max(returns) - min(returns)
    assert spread > 0.2, (
        f"r_struggle return spread over k=1..30 is only {spread:.4f}; "
        "a per-step term is cancelling the terminal penalty"
    )


def test_stability_return_spread_is_large_enough_to_learn_from(exp):
    returns = [
        _struggle_return(exp, "S" * k + "X" * 3, fell=True) for k in range(1, 31)
    ]
    spread = max(returns) - min(returns)
    assert spread > 0.2


def test_recovering_sooner_is_rewarded(exp):
    fast = _struggle_return(exp, "XX" + "S" * 20, fell=False)
    slow = _struggle_return(exp, "X" * 15 + "S" * 20, fell=False)
    assert fast > slow


def test_longer_stability_is_rewarded(exp):
    short = _struggle_return(exp, "SS" + "X" * 5, fell=True)
    long_ = _struggle_return(exp, "S" * 20 + "X" * 5, fell=True)
    assert long_ > short


def test_surviving_beats_falling(exp):
    survive = _struggle_return(exp, "S" * 20, fell=False)
    fall = _struggle_return(exp, "S" * 10 + "X" * 10, fell=True)
    assert survive > fall


def test_recovering_beats_falling(exp):
    recover = _struggle_return(exp, "X" * 5 + "S" * 10, fell=False)
    fall = _struggle_return(exp, "X" * 5, fell=True)
    assert recover > fall


def test_oscillation_is_never_profitable(exp):
    """Flickering must not farm recovery bonuses."""
    stable = _struggle_return(exp, "S" * 12, fell=False)
    oscil = _struggle_return(exp, "SSXXSSXXSSXX", fell=False)
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
