"""Behavioral tests for the thin :class:`EpisodeRunner` contract.

Pins the post-refactor surface that had no coverage after the runner-layer
rewrite (see AUDIT.md — S8):

* ``run_episode`` returns ``None`` — episode data lives in recorders.
* ``post_termination_action="hold"`` replays the terminated agent's last
  action instead of calling ``act`` again; ``"policy"`` keeps calling it.
* ``want_extras=True`` forwards ``want_extra=True`` into ``policy.act`` and
  delivers the extras to recorders via ``action_extras``.
* Constructor duck-type check: a policy object without ``act`` is rejected.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import AGENT_IDS, EpisodeRunner
from envs.framework.plugin import BasePlugin
from envs.framework.policy import Policy
from envs.framework.recorder import PostActionRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _CountingPolicy(Policy):
    """Returns a distinct action value per call so held vs. fresh actions
    are distinguishable; records ``want_extra`` flags it was called with."""

    def __init__(self, action_dim: int = 21):
        self.action_dim = action_dim
        self.call_count = 0
        self.want_extra_seen: List[bool] = []

    def act(self, observation: Any, *, want_extra: bool = False):
        self.call_count += 1
        self.want_extra_seen.append(bool(want_extra))
        action = np.full(self.action_dim, float(self.call_count), dtype=np.float32)
        extra = {"logprob": -0.5, "call": self.call_count} if want_extra else None
        return action, extra


class _TerminateAAfterStep(BasePlugin):
    """Terminates only ``robot_a`` once ``episode_step`` reaches a threshold."""

    def __init__(self, at_step: int = 1):
        self._at_step = at_step

    @property
    def name(self) -> str:
        return "terminate_a"

    def on_post_action_step(self, ctx) -> None:
        if ctx.episode_step >= self._at_step:
            ctx.request_termination("ko_a", agent_id="robot_a")


class _TerminateAOnPreEpisode(BasePlugin):
    """Terminates ``robot_a`` before the episode starts."""

    @property
    def name(self) -> str:
        return "terminate_a_pre"

    def on_pre_episode(self, ctx) -> None:
        ctx.request_termination("ko_a_pre", agent_id="robot_a")


class _ActionCaptureRecorder(PostActionRecorder):
    """Snapshots the per-agent action mapping (and extras) each step."""

    def __init__(self) -> None:
        self.actions: List[Dict[str, np.ndarray]] = []
        self.extras: List[Any] = []

    def on_post_action_step(
        self, ctx, observation, action, observer_outputs, action_extras=None
    ) -> None:
        self.actions.append(
            {k: np.asarray(v).copy() for k, v in action.items()}
        )
        self.extras.append(action_extras)


def _build(
    mock_simulator,
    *,
    plugins=None,
    max_steps: int = 4,
    post_termination_action: str = "policy",
) -> tuple[EpisodeRunner, _CountingPolicy, _CountingPolicy, _ActionCaptureRecorder]:
    recorder = _ActionCaptureRecorder()
    runtime = EnvRuntime(
        simulator=mock_simulator,
        plugins=list(plugins or []),
        recorders=[recorder],
        max_steps=max_steps,
    )
    pa, pb = _CountingPolicy(), _CountingPolicy()
    runner = EpisodeRunner(
        runtime=runtime,
        policy_a=pa,
        policy_b=pb,
        post_termination_action=post_termination_action,
    )
    return runner, pa, pb, recorder


# ---------------------------------------------------------------------------
# Contract: run_episode returns None
# ---------------------------------------------------------------------------
class TestRunEpisodeReturnsNone:
    def test_returns_none(self, mock_simulator):
        runner, _, _, _ = _build(mock_simulator)
        assert runner.run_episode(seed=1) is None


# ---------------------------------------------------------------------------
# Duck-type policy validation
# ---------------------------------------------------------------------------
class TestDuckTypePolicyCheck:
    def test_policy_without_act_rejected(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator, max_steps=1)

        class NotAPolicy:
            pass

        with pytest.raises(TypeError, match="act"):
            EpisodeRunner(runtime=runtime, policy_a=NotAPolicy(), policy_b=_CountingPolicy())
        with pytest.raises(TypeError, match="act"):
            EpisodeRunner(runtime=runtime, policy_a=_CountingPolicy(), policy_b=NotAPolicy())

    def test_duck_typed_non_subclass_accepted(self, mock_simulator):
        """Self-contained exported policies inline a Policy stub — they are
        not ``Policy`` subclasses and must still be accepted (P0-6)."""
        runtime = EnvRuntime(simulator=mock_simulator, max_steps=1)

        class DuckPolicy:
            def act(self, observation, want_extra=False):
                return np.zeros(21, dtype=np.float32), None

        runner = EpisodeRunner(runtime=runtime, policy_a=DuckPolicy(), policy_b=DuckPolicy())
        assert runner.run_episode(seed=1) is None


# ---------------------------------------------------------------------------
# post_termination_action semantics
# ---------------------------------------------------------------------------
class TestPostTerminationAction:
    def test_hold_replays_last_action_and_stops_calling_act(self, mock_simulator):
        runner, pa, pb, rec = _build(
            mock_simulator,
            plugins=[_TerminateAAfterStep(at_step=1)],
            max_steps=4,
            post_termination_action="hold",
        )
        runner.run_episode(seed=1)

        # robot_a terminated after step 1 → act called exactly once.
        assert pa.call_count == 1
        # robot_b kept being driven by its policy every step.
        assert pb.call_count == 4
        # Steps 1..3 replayed the action produced at step 0 (value 1.0).
        for step_actions in rec.actions[1:]:
            assert np.all(step_actions["robot_a"] == 1.0)
        # robot_b actions kept changing (fresh policy calls).
        assert rec.actions[3]["robot_b"][0] == 4.0

    def test_policy_mode_keeps_calling_act(self, mock_simulator):
        runner, pa, pb, rec = _build(
            mock_simulator,
            plugins=[_TerminateAAfterStep(at_step=1)],
            max_steps=4,
            post_termination_action="policy",
        )
        runner.run_episode(seed=1)

        # Both policies keep being asked for actions despite robot_a's
        # termination at step 1.
        assert pa.call_count == 4
        assert pb.call_count == 4

    def test_pre_episode_terminated_agent_still_gets_one_act_call(
        self, mock_simulator
    ):
        """``a_active`` starts ``True`` unconditionally, so an agent that was
        terminated in ``on_pre_episode`` is still asked for one action before
        hold-mode kicks in — the ``last_action_a is None`` RuntimeError in
        the loop is therefore unreachable by design. Pin this contract."""
        runner, pa, pb, rec = _build(
            mock_simulator,
            plugins=[_TerminateAOnPreEpisode()],
            max_steps=3,
            post_termination_action="hold",
        )
        runner.run_episode(seed=1)

        assert pa.call_count == 1
        for step_actions in rec.actions[1:]:
            assert np.all(step_actions["robot_a"] == 1.0)


# ---------------------------------------------------------------------------
# want_extras forwarding
# ---------------------------------------------------------------------------
class TestWantExtras:
    def test_want_extra_true_forwarded_to_act(self, mock_simulator):
        runner, pa, pb, _ = _build(mock_simulator, max_steps=2)
        runner.run_episode(seed=1, want_extras=True)
        assert pa.want_extra_seen == [True, True]
        assert pb.want_extra_seen == [True, True]

    def test_extras_reach_recorder(self, mock_simulator):
        runner, pa, pb, rec = _build(mock_simulator, max_steps=2)
        runner.run_episode(seed=1, want_extras=True)
        assert len(rec.extras) == 2
        first = rec.extras[0]
        assert first["robot_a"] == {"logprob": -0.5, "call": 1}
        assert first["robot_b"] == {"logprob": -0.5, "call": 1}

    def test_default_is_no_extras(self, mock_simulator):
        runner, pa, pb, rec = _build(mock_simulator, max_steps=2)
        runner.run_episode(seed=1)
        assert pa.want_extra_seen == [False, False]
        for step_extras in rec.extras:
            assert step_extras["robot_a"] is None
            assert step_extras["robot_b"] is None


# ---------------------------------------------------------------------------
# Runner does not own runtime lifecycle
# ---------------------------------------------------------------------------
class TestLifecycleOwnership:
    def test_episode_runner_close_does_not_close_runtime(self, mock_simulator):
        """``EpisodeRunner.close`` releases policies, not the runtime —
        the caller owns the simulator/runtime lifecycle."""
        runner, pa, pb, _ = _build(mock_simulator, max_steps=1)
        runner.run_episode(seed=1)
        runner.close()
        # Runtime is still usable for another episode.
        runner2 = EpisodeRunner(
            runtime=runner.runtime,
            policy_a=_CountingPolicy(),
            policy_b=_CountingPolicy(),
        )
        assert runner2.run_episode(seed=2) is None
